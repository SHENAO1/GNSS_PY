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
    # 如果 ephemeris.py 里还有别的工具函数，也可以一并导入
)

# 卫星位置与钟差
from gnss.navigation.ephemeris.satpos import satpos

# 坐标转换相关函数，现在放在 utils/geo_functions.py 里
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


def post_navigation(
    track_results: Sequence,
    settings,
) -> Tuple[SimpleNamespace, Dict[int, object]]:

    n_ch = settings.numberOfChannels

    # 在函数内部加一个小工具，用于构造“空 nav”
    def _make_empty_nav(num_epochs: int = 0):
        nav = SimpleNamespace()
        nav.channel = SimpleNamespace()

        nav.channel.PRN        = np.zeros((n_ch, num_epochs), dtype=int)
        nav.channel.rawP       = np.zeros((n_ch, num_epochs), dtype=float)
        nav.channel.correctedP = np.zeros((n_ch, num_epochs), dtype=float)
        nav.channel.az         = np.zeros((n_ch, num_epochs), dtype=float)
        nav.channel.el         = np.zeros((n_ch, num_epochs), dtype=float)

        nav.DOP       = np.zeros((5, num_epochs), dtype=float)
        nav.X         = np.zeros(num_epochs, dtype=float)
        nav.Y         = np.zeros(num_epochs, dtype=float)
        nav.Z         = np.zeros(num_epochs, dtype=float)
        nav.dt        = np.zeros(num_epochs, dtype=float)
        nav.latitude  = np.zeros(num_epochs, dtype=float)
        nav.longitude = np.zeros(num_epochs, dtype=float)
        nav.height    = np.zeros(num_epochs, dtype=float)
        nav.E         = np.zeros(num_epochs, dtype=float)
        nav.N         = np.zeros(num_epochs, dtype=float)
        nav.U         = np.zeros(num_epochs, dtype=float)
        nav.utmZone   = None
        return nav

    # ---------- 0. 检查数据长度 & 跟踪卫星数量 ----------
    num_tracked = sum(1 for tr in track_results if _get_field(tr, "status") != "-")
    print(f"[NAV DEBUG] msToProcess={settings.msToProcess}, num_tracked={num_tracked}")

    if settings.msToProcess < 36000 or num_tracked < 4:
        print("记录时间太短或跟踪到的卫星太少。正在退出！")
        nav_empty = _make_empty_nav(0)
        return nav_empty, {}

    # ---------- 1. 寻找前导码起始位置 ----------
    sub_frame_start, active_chn_list = find_preambles(track_results, settings)

    print("[DEBUG] sub_frame_start =", sub_frame_start)
    print("[DEBUG] active_chn_list(before ephemeris) =", active_chn_list)

    # ---------- 2. 解码星历 ----------
    eph: Dict[int, object] = {}
    TOW = None  # Time Of Week

    active_chn_list = list(active_chn_list)

    for ch in list(active_chn_list):
        idx = ch - 1
        tr = track_results[idx]

        # I_P 为每毫秒积分后的同相分量
        I_P = np.asarray(_get_field(tr, "I_P"), dtype=float)

        # 从前导码起点往前 20 ms，往后 1500*20-1 ms，刚好覆盖 5 个子帧
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

        # 🔍 调试：打印当前通道前 50 bit，确认极性和结构
        print(f"[NAV DEBUG] [CH {ch}] nav_bits[0:50] = {nav_bits[0:50].astype(int)}")

        # 取 1500 bit（对应 30000 ms）用于星历解析
        bits_for_ephem = nav_bits[1:1501]         # 对应 MATLAB 的 navBitsBin(2:1501)'
        # D30Star：前一字的第30位（MATLAB navBitsBin(1)），这里也转成 '0'/'1'
        tlm_last_bit   = '1' if nav_bits[0] else '0'

        bits_for_ephem_str = ''.join('1' if b else '0' for b in bits_for_ephem)

        prn = int(_get_field(tr, "PRN"))

        try:
            eph_prn, TOW_new = ephemeris(bits_for_ephem_str, tlm_last_bit)
        except Exception as e:
            print(f"[post_navigation] 通道 {ch} (PRN {prn}) 星历解析失败: {e}")
            active_chn_list.remove(ch)
            continue

        if eph_prn is None:
            print(f"[post_navigation] 通道 {ch} (PRN {prn}) 星历为空，剔除。")
            active_chn_list.remove(ch)
            continue

        # 首次拿到一个 TOW 就记住，后续如果不一致再说
        if TOW is None:
            TOW = TOW_new
        else:
            # 这里简单检查一下是否有明显不一致（可选）
            if abs(TOW_new - TOW) > 30:
                print(f"[WARN] 通道 {ch} (PRN {prn}) 的 TOW 与之前不一致: {TOW_new} vs {TOW}")

        eph[prn] = eph_prn

        # （仅打印一下星历结构里有哪些字段，方便你之后核对）
        if hasattr(eph_prn, "__dict__"):
            print(f"[NAV DEBUG] PRN {prn} ephemeris fields:", list(eph_prn.__dict__.keys()))
        elif isinstance(eph_prn, dict):
            print(f"[NAV DEBUG] PRN {prn} ephemeris keys:", list(eph_prn.keys()))

        # ⚠️ 先不按 IODC/IODE 剔除，等确认字段名后再加质量判断
        # iodc  = getattr(eph_prn, "IODC", None)
        # iode2 = getattr(eph_prn, "IODE_sf2", None)
        # iode3 = getattr(eph_prn, "IODE_sf3", None)
        # if iodc is None or iode2 is None or iode3 is None:
        #     active_chn_list.remove(ch)

    # ---------- 3. 再次检查卫星数量 ----------
    print("[DEBUG] active_chn_list(after ephemeris) =", active_chn_list)

    if len(active_chn_list) < 4 or TOW is None:
        print("拥有星历数据的卫星太少，无法进行位置计算。正在退出！")
        nav_empty = _make_empty_nav(0)
        return nav_empty, eph

    # ---------- 4. 初始化解算结果结构 ----------
    max_start   = int(np.max(sub_frame_start))
    num_epochs  = int((settings.msToProcess - max_start) // settings.navSolPeriod)
    if num_epochs <= 0:
        print("可用测量历元数为 0。正在退出！")
        nav_empty = _make_empty_nav(0)
        return nav_empty, eph

    nav = SimpleNamespace()
    nav.channel = SimpleNamespace()

    nav.channel.PRN        = np.zeros((n_ch, num_epochs), dtype=int)
    nav.channel.rawP       = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.correctedP = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.az         = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.el         = np.full((n_ch, num_epochs), np.nan, dtype=float)

    nav.DOP = np.zeros((5, num_epochs), dtype=float)

    nav.X  = np.full(num_epochs, np.nan, dtype=float)
    nav.Y  = np.full(num_epochs, np.nan, dtype=float)
    nav.Z  = np.full(num_epochs, np.nan, dtype=float)
    nav.dt = np.full(num_epochs, np.nan, dtype=float)

    nav.latitude  = np.full(num_epochs, np.nan, dtype=float)
    nav.longitude = np.full(num_epochs, np.nan, dtype=float)
    nav.height    = np.full(num_epochs, np.nan, dtype=float)
    nav.E         = np.full(num_epochs, np.nan, dtype=float)
    nav.N         = np.full(num_epochs, np.nan, dtype=float)
    nav.U         = np.full(num_epochs, np.nan, dtype=float)
    nav.utmZone   = None

    sat_elev = np.full(n_ch, np.inf, dtype=float)
    ready_chn_list: List[int] = list(active_chn_list)

    transmit_time = float(TOW)

    # ---------- 5. 按历元循环解算 ----------
    for epoch_idx in range(num_epochs):
        curr_meas_nr = epoch_idx + 1

        above_mask = [i + 1 for i in range(n_ch) if sat_elev[i] >= settings.elevationMask]
        active_now = sorted(set(above_mask).intersection(ready_chn_list))

        for ch in active_now:
            nav.channel.PRN[ch - 1, epoch_idx] = int(_get_field(track_results[ch - 1], "PRN"))

        nav.channel.az[:, epoch_idx] = np.nan
        nav.channel.el[:, epoch_idx] = np.nan

        ms_of_signal = sub_frame_start + settings.navSolPeriod * epoch_idx
        raw_p = calculate_pseudoranges(
            track_results,
            ms_of_signal,
            active_now,
            settings,
        )
        nav.channel.rawP[:, epoch_idx] = raw_p

        prn_list = [int(_get_field(track_results[ch - 1], "PRN")) for ch in active_now]
        sat_positions, sat_clk_corr = satpos(transmit_time, prn_list, eph, settings)

        if len(active_now) > 3:
            raw_p_used = nav.channel.rawP[[ch - 1 for ch in active_now], epoch_idx]

            xyzdt, el, az, DOP = least_square_pos(
                sat_positions,
                raw_p_used + sat_clk_corr * settings.c,
                settings,
            )

            nav.X[epoch_idx]  = xyzdt[0]
            nav.Y[epoch_idx]  = xyzdt[1]
            nav.Z[epoch_idx]  = xyzdt[2]
            nav.dt[epoch_idx] = xyzdt[3]

            nav.channel.el[[ch - 1 for ch in active_now], epoch_idx] = el
            nav.channel.az[[ch - 1 for ch in active_now], epoch_idx] = az
            nav.DOP[:, epoch_idx] = DOP

            sat_elev = nav.channel.el[:, epoch_idx]

            nav.channel.correctedP[[ch - 1 for ch in active_now], epoch_idx] = (
                raw_p_used + sat_clk_corr * settings.c + nav.dt[epoch_idx]
            )

            lat, lon, hgt = cart2geo(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], 5)
            nav.latitude[epoch_idx]  = lat
            nav.longitude[epoch_idx] = lon
            nav.height[epoch_idx]    = hgt

            utm_zone   = find_utm_zone(lat, lon)
            nav.utmZone = utm_zone

            E, N, U = cart2utm(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], utm_zone)
            nav.E[epoch_idx] = E
            nav.N[epoch_idx] = N
            nav.U[epoch_idx] = U

        else:
            print(f"   测量历元 No. {curr_meas_nr}: 信息不足，无法进行位置解算。")

            nav.X[epoch_idx]  = np.nan
            nav.Y[epoch_idx]  = np.nan
            nav.Z[epoch_idx]  = np.nan
            nav.dt[epoch_idx] = np.nan
            nav.DOP[:, epoch_idx] = 0.0
            nav.latitude[epoch_idx]  = np.nan
            nav.longitude[epoch_idx] = np.nan
            nav.height[epoch_idx]    = np.nan
            nav.E[epoch_idx] = np.nan
            nav.N[epoch_idx] = np.nan
            nav.U[epoch_idx] = np.nan

            nav.channel.az[[ch - 1 for ch in active_now], epoch_idx] = np.nan
            nav.channel.el[[ch - 1 for ch in active_now], epoch_idx] = np.nan

        transmit_time += settings.navSolPeriod / 1000.0

    return nav, eph



