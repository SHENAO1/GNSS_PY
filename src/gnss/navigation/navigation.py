# src/gnss/navigation/navigation.py

from types import SimpleNamespace
from typing import Sequence, Dict, Tuple, List

import numpy as np

from gnss.navigation.nav_msg import find_preambles
from gnss.navigation.pseudorange import calculate_pseudoranges

# 卫星星历与时间修正相关
from gnss.navigation.ephemeris.ephemeris import (
    ephemeris,   # 解析广播星历
    check_t,     # 周首/周末时间跳变修正
)

# 卫星位置与钟差
from gnss.navigation.ephemeris.satpos import satpos

# 坐标转换相关函数
from gnss.utils.geo_functions import (
    cart2geo,
    find_utm_zone,
    cart2utm,
)
# 位置解算相关函数
from gnss.navigation.positioning import least_square_pos


def _get_field(tr, name):
    """兼容 struct 对象 / dict 的字段访问。"""
    if hasattr(tr, name):
        return getattr(tr, name)
    if isinstance(tr, dict) and name in tr:
        return tr[name]
    raise AttributeError(f"trackResults 中缺少字段 {name!r}")


def _make_empty_nav(n_ch: int, num_epochs: int = 0):
    """辅助函数：创建空的 nav 结构"""
    nav = SimpleNamespace()
    nav.channel = SimpleNamespace()

    nav.channel.PRN        = np.zeros((n_ch, num_epochs), dtype=int)
    nav.channel.rawP       = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.correctedP = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.az         = np.zeros((n_ch, num_epochs), dtype=float)
    nav.channel.el         = np.zeros((n_ch, num_epochs), dtype=float)

    nav.DOP       = np.zeros((5, num_epochs), dtype=float)
    nav.X         = np.full(num_epochs, np.nan, dtype=float)
    nav.Y         = np.full(num_epochs, np.nan, dtype=float)
    nav.Z         = np.full(num_epochs, np.nan, dtype=float)
    nav.dt        = np.full(num_epochs, np.nan, dtype=float)
    nav.latitude  = np.full(num_epochs, np.nan, dtype=float)
    nav.longitude = np.full(num_epochs, np.nan, dtype=float)
    nav.height    = np.full(num_epochs, np.nan, dtype=float)
    nav.E         = np.full(num_epochs, np.nan, dtype=float)
    nav.N         = np.full(num_epochs, np.nan, dtype=float)
    nav.U         = np.full(num_epochs, np.nan, dtype=float)
    nav.utmZone   = None
    return nav


def post_navigation(
    track_results: Sequence,
    settings,
) -> Tuple[SimpleNamespace, Dict[int, object]]:

    n_ch = settings.numberOfChannels
    
    # ---------- 0. 检查数据长度 & 跟踪卫星数量 ----------
    num_tracked = sum(1 for tr in track_results if _get_field(tr, "status") != "-")
    print(f"[NAV DEBUG] msToProcess={settings.msToProcess}, num_tracked={num_tracked}")

    if settings.msToProcess < 36000 or num_tracked < 4:
        print("记录时间太短或跟踪到的卫星太少。正在退出！")
        return _make_empty_nav(n_ch, 0), {}

    # ---------- 1. 寻找前导码起始位置 ----------
    sub_frame_start, active_chn_list = find_preambles(track_results, settings)

    print("[DEBUG] sub_frame_start =", sub_frame_start)
    print("[DEBUG] active_chn_list(before ephemeris) =", active_chn_list)

    # ---------- 2. 解码星历 ----------
    eph: Dict[int, object] = {}
    TOW = None  # Time Of Week

    # 复制一份列表用于遍历，因为会在循环中修改 active_chn_list
    for ch in list(active_chn_list):
        idx = ch - 1
        tr = track_results[idx]

        # I_P 为每毫秒积分后的同相分量
        I_P = np.asarray(_get_field(tr, "I_P"), dtype=float)

        # 确保数据足够覆盖 5 个子帧 (1500 bits = 30000 ms)
        # 从前导码起点往前 20 ms，往后 1500*20-1 ms
        start_ms = int(sub_frame_start[idx]) - 20
        end_ms   = int(sub_frame_start[idx]) + 1500 * 20 - 1

        if start_ms < 1 or end_ms > len(I_P):
            print(f"[post_navigation] 通道 {ch} 数据长度不足以提取 5 个子帧，剔除。")
            active_chn_list.remove(ch)
            continue

        # 取出这一段数据
        seg = I_P[start_ms - 1 : end_ms].copy()

        # 按 20ms 一列 reshape 成 20 x N 的矩阵，然后按行求和做符号判决
        seg_mat       = seg.reshape(-1, 20).T
        nav_bits_soft = np.sum(seg_mat, axis=0)

        # 门限判决为 0/1（bool 数组）
        nav_bits = nav_bits_soft > 0

        # 取 1500 bit 用于星历解析
        bits_for_ephem = nav_bits[1:1501]
        
        prn = int(_get_field(tr, "PRN"))

        try:
            # ephemeris 函数现在会尝试自动处理相位反转
            eph_prn, TOW_new = ephemeris(bits_for_ephem, None)
        except Exception as e:
            print(f"[post_navigation] 通道 {ch} (PRN {prn}) 解析异常: {e}")
            active_chn_list.remove(ch)
            continue

        if eph_prn is None:
            print(f"[post_navigation] 通道 {ch} (PRN {prn}) 星历无效，剔除。")
            active_chn_list.remove(ch)
            continue

        # 首次拿到一个 TOW 就记住
        if TOW is None:
            TOW = TOW_new

        eph[prn] = eph_prn
        print(f"[NAV DEBUG] PRN {prn} ephemeris decoded OK.")

    # ---------- 3. 再次检查卫星数量 ----------
    print("[DEBUG] active_chn_list(after ephemeris) =", active_chn_list)
    print("[DEBUG] Valid Ephemeris Keys:", list(eph.keys()))

    if len(eph) < 4 or TOW is None:
        print("拥有有效星历的卫星不足 4 颗，无法定位。正在退出！")
        return _make_empty_nav(n_ch, 0), eph

    # ---------- 4. 初始化解算结果结构 ----------
    max_start   = int(np.max(sub_frame_start))
    num_epochs  = int((settings.msToProcess - max_start) // settings.navSolPeriod)
    
    if num_epochs <= 0:
        print("可用测量历元数为 0。正在退出！")
        return _make_empty_nav(n_ch, 0), eph

    nav = _make_empty_nav(n_ch, num_epochs)

    sat_elev = np.full(n_ch, np.inf, dtype=float)
    ready_chn_list: List[int] = list(active_chn_list)

    transmit_time = float(TOW)

    print(f"开始位置解算，共 {num_epochs} 个历元...")

    # ---------- 5. 按历元循环解算 ----------
    for epoch_idx in range(num_epochs):
        curr_meas_nr = epoch_idx + 1

        # 1. 筛选高度角满足要求的卫星
        above_mask = [i + 1 for i in range(n_ch) if sat_elev[i] >= settings.elevationMask]
        # 2. 取交集：既有星历(ready_chn_list) 又在高度角以上
        potential_sats = sorted(set(above_mask).intersection(ready_chn_list))
        
        # 3. 严格检查：确保这些卫星真的在 eph 字典里（双重保险）
        active_now = [ch for ch in potential_sats if int(_get_field(track_results[ch - 1], "PRN")) in eph]

        # 记录 PRN
        for ch in active_now:
            nav.channel.PRN[ch - 1, epoch_idx] = int(_get_field(track_results[ch - 1], "PRN"))

        # 准备数据
        ms_of_signal = sub_frame_start + settings.navSolPeriod * epoch_idx
        
        # 计算伪距
        # 注意：这里 calculate_pseudoranges 内部可能会返回 NaN，如果信号不好
        raw_p = calculate_pseudoranges(
            track_results,
            ms_of_signal,
            active_now,
            settings,
        )
        nav.channel.rawP[:, epoch_idx] = raw_p

        # 计算卫星位置
        prn_list = [int(_get_field(track_results[ch - 1], "PRN")) for ch in active_now]
        
        # 这里的 eph 必须包含 prn_list 里的所有 PRN
        sat_positions, sat_clk_corr = satpos(transmit_time, prn_list, eph, settings)

        # 检查 sat_positions 是否包含 NaN
        valid_indices = []
        if sat_positions is not None:
             for i in range(len(active_now)):
                 if not np.any(np.isnan(sat_positions[i, :])):
                     valid_indices.append(i)
        
        # 只有当有效卫星（位置非NaN）数量 > 3 时才解算
        if len(valid_indices) > 3:
            # 筛选出真正有效的子集
            final_active_chns = [active_now[i] for i in valid_indices]
            final_raw_p = nav.channel.rawP[[ch - 1 for ch in final_active_chns], epoch_idx]
            final_sat_pos = sat_positions[valid_indices, :]
            final_sat_clk = sat_clk_corr[valid_indices]

            # 调用最小二乘
            try:
                xyzdt, el, az, DOP = least_square_pos(
                    final_sat_pos,
                    final_raw_p + final_sat_clk * settings.c,
                    settings,
                )
                
                # 存结果
                nav.X[epoch_idx]  = xyzdt[0]
                nav.Y[epoch_idx]  = xyzdt[1]
                nav.Z[epoch_idx]  = xyzdt[2]
                nav.dt[epoch_idx] = xyzdt[3]

                # 映射回 active channels
                for i, ch_idx in enumerate(valid_indices):
                    ch = active_now[ch_idx] # 原通道号
                    nav.channel.el[ch - 1, epoch_idx] = el[i]
                    nav.channel.az[ch - 1, epoch_idx] = az[i]
                    # 更新高度角用于下一次循环
                    sat_elev[ch - 1] = el[i] * 180 / np.pi # 转回角度存 sat_elev? 
                    # 注意：least_square_pos 返回的是 弧度，但 settings.elevationMask 通常是 角度(10度)
                    # 请确认你的 elevationMask 单位。通常需要转换。这里假设 settings 用角度。
                
                nav.DOP[:, epoch_idx] = DOP

                # 存 correctedP
                for i, ch_idx in enumerate(valid_indices):
                    ch = active_now[ch_idx]
                    nav.channel.correctedP[ch - 1, epoch_idx] = (
                        final_raw_p[i] + final_sat_clk[i] * settings.c + nav.dt[epoch_idx]
                    )

                # 经纬度转换
                lat, lon, hgt = cart2geo(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], 5)
                nav.latitude[epoch_idx]  = lat
                nav.longitude[epoch_idx] = lon
                nav.height[epoch_idx]    = hgt

                utm_zone  = find_utm_zone(lat, lon)
                nav.utmZone = utm_zone
                E, N, U = cart2utm(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], utm_zone)
                nav.E[epoch_idx] = E
                nav.N[epoch_idx] = N
                nav.U[epoch_idx] = U

            except np.linalg.LinAlgError:
                # 捕获 SVD 不收敛，不崩溃，跳过该历元
                print(f"[NAV WARN] Epoch {epoch_idx}: SVD 不收敛 (可能是矩阵奇异)")
            except Exception as e:
                print(f"[NAV ERROR] Epoch {epoch_idx}: 未知计算错误 {e}")

        else:
            # print(f"  测量历元 No. {curr_meas_nr}: 有效卫星不足 ({len(valid_indices)} < 4)。")
            # 保持 NaN
            pass

        transmit_time += settings.navSolPeriod / 1000.0

    return nav, eph