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
    """
    Python 版 postNavigation.m

    参数
    ----
    track_results : 序列
        tracking 模块的输出，每个元素对应一个通道。
        需要包含：
            - status : 字符，'-' 表示未跟踪
            - PRN
            - I_P
    settings : Settings
        需要字段：
            - msToProcess
            - numberOfChannels
            - navSolPeriod
            - elevationMask
            - c

    返回
    ----
    nav_solutions : SimpleNamespace
        包含 E/N/U、ECEF、钟差、DOP、各通道相关信息等
    eph : dict[int -> Ephemeris]
        按 PRN 索引的星历对象字典
    """

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
    # sub_frame_start: np.ndarray, len = numberOfChannels, 每个元素是 1-based 毫秒索引

    # ---------- 2. 解码星历 ----------
    eph: Dict[int, object] = {}
    TOW = None  # Time Of Week

    # 复制一份通道列表，后面会删掉数据不完整的卫星
    active_chn_list = list(active_chn_list)

    for ch in list(active_chn_list):  # 用 list() 是为了循环中安全删除
        idx = ch - 1
        tr = track_results[idx]

        I_P = np.asarray(_get_field(tr, "I_P"), dtype=float)

        start_ms = int(sub_frame_start[idx]) - 20
        end_ms = int(sub_frame_start[idx]) + 1500 * 20 - 1

        if start_ms < 1 or end_ms > len(I_P):
            print(f"[post_navigation] 通道 {ch} 数据长度不足以提取 5 个子帧，剔除。")
            active_chn_list.remove(ch)
            continue

        # MATLAB: I_P(start_ms : end_ms)' -> 注意 1-based
        seg = I_P[start_ms - 1 : end_ms].copy()

        # 重塑为 (20, Nbit) 并按列求和
        seg_mat = seg.reshape(-1, 20).T  # 形状 (20, Nbit)
        nav_bits_soft = np.sum(seg_mat, axis=0)

        # 门限判决为 0/1
        nav_bits = nav_bits_soft > 0  # bool 数组

        # MATLAB: navBitsBin = dec2bin(navBits);
        # 这里直接用 0/1 数组传给 ephemeris 即可
        bits_for_ephem = nav_bits[1:1501]   # navBitsBin(2:1501)'
        tlm_last_bit = int(nav_bits[0])     # navBitsBin(1)

        prn = int(_get_field(tr, "PRN"))

        eph_prn, TOW = ephemeris(bits_for_ephem, tlm_last_bit)
        eph[prn] = eph_prn

        # 检查星历是否完整
        iodc = getattr(eph_prn, "IODC", None)
        iode2 = getattr(eph_prn, "IODE_sf2", None)
        iode3 = getattr(eph_prn, "IODE_sf3", None)

        if iodc is None or iode2 is None or iode3 is None:
            active_chn_list.remove(ch)

    # ---------- 3. 再次检查卫星数量 ----------
    if len(active_chn_list) < 4 or TOW is None:
        print("拥有星历数据的卫星太少，无法进行位置计算。正在退出！")
        nav_empty = _make_empty_nav(0)
        return nav_empty, eph


    # ---------- 4. 初始化解算结果结构 ----------
    max_start = int(np.max(sub_frame_start))
    num_epochs = int((settings.msToProcess - max_start) // settings.navSolPeriod)
    if num_epochs <= 0:
        print("可用测量历元数为 0。正在退出！")
        nav_empty = _make_empty_nav(0)
        return nav_empty, eph


    nav = SimpleNamespace()
    nav.channel = SimpleNamespace()

    # 通道相关量（维度：n_ch × num_epochs）
    nav.channel.PRN = np.zeros((n_ch, num_epochs), dtype=int)
    nav.channel.rawP = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.correctedP = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.az = np.full((n_ch, num_epochs), np.nan, dtype=float)
    nav.channel.el = np.full((n_ch, num_epochs), np.nan, dtype=float)

    # DOP (5 × num_epochs)
    nav.DOP = np.zeros((5, num_epochs), dtype=float)

    # 接收机位置 & 钟差（1 × num_epochs）
    nav.X = np.full(num_epochs, np.nan, dtype=float)
    nav.Y = np.full(num_epochs, np.nan, dtype=float)
    nav.Z = np.full(num_epochs, np.nan, dtype=float)
    nav.dt = np.full(num_epochs, np.nan, dtype=float)

    nav.latitude = np.full(num_epochs, np.nan, dtype=float)
    nav.longitude = np.full(num_epochs, np.nan, dtype=float)
    nav.height = np.full(num_epochs, np.nan, dtype=float)
    nav.E = np.full(num_epochs, np.nan, dtype=float)
    nav.N = np.full(num_epochs, np.nan, dtype=float)
    nav.U = np.full(num_epochs, np.nan, dtype=float)
    nav.utmZone = None  # 文本，最后一次更新的 UTM 分区

    # 卫星仰角初始化为 +inf（这样第一次不会被 elevationMask 剔除）
    sat_elev = np.full(n_ch, np.inf, dtype=float)
    ready_chn_list: List[int] = list(active_chn_list)

    transmit_time = float(TOW)  # [s]

    # ---------- 5. 按历元循环解算 ----------
    for epoch_idx in range(num_epochs):
        curr_meas_nr = epoch_idx + 1  # 仅用于输出时展示

        # elevationMask 过滤 + 只保留 ready 里的通道
        above_mask = [i + 1 for i in range(n_ch) if sat_elev[i] >= settings.elevationMask]
        active_now = sorted(set(above_mask).intersection(ready_chn_list))

        # 记录本历元可用卫星的 PRN
        for ch in active_now:
            nav.channel.PRN[ch - 1, epoch_idx] = int(_get_field(track_results[ch - 1], "PRN"))

        # 先把所有通道的 az/el 设为 NaN（防止 skyplot 出现 0 点跳变）
        nav.channel.az[:, epoch_idx] = np.nan
        nav.channel.el[:, epoch_idx] = np.nan

        # ---------- 5.1 计算原始伪距 ----------
        ms_of_signal = sub_frame_start + settings.navSolPeriod * epoch_idx
        raw_p = calculate_pseudoranges(
            track_results,
            ms_of_signal,
            active_now,
            settings,
        )
        nav.channel.rawP[:, epoch_idx] = raw_p

        # ---------- 5.2 计算卫星位置 & 卫星钟差 ----------
        prn_list = [int(_get_field(track_results[ch - 1], "PRN")) for ch in active_now]
        sat_positions, sat_clk_corr = satpos(transmit_time, prn_list, eph, settings)

        # ---------- 5.3 计算接收机位置 ----------
        if len(active_now) > 3:
            # 取出本历元、且通过 elevationMask 的伪距，并加上卫星钟差修正
            raw_p_used = nav.channel.rawP[[ch - 1 for ch in active_now], epoch_idx]
            # least_square_pos 里自己决定 sat_positions 的维度约定
            xyzdt, el, az, DOP = least_square_pos(
                sat_positions,
                raw_p_used + sat_clk_corr * settings.c,
                settings,
            )

            nav.X[epoch_idx] = xyzdt[0]
            nav.Y[epoch_idx] = xyzdt[1]
            nav.Z[epoch_idx] = xyzdt[2]
            nav.dt[epoch_idx] = xyzdt[3]

            # 更新当前历元的卫星仰角 & 方位角
            nav.channel.el[[ch - 1 for ch in active_now], epoch_idx] = el
            nav.channel.az[[ch - 1 for ch in active_now], epoch_idx] = az
            nav.DOP[:, epoch_idx] = DOP

            # 用当前仰角更新 sat_elev（供下一历元使用 elevationMask）
            sat_elev = nav.channel.el[:, epoch_idx]

            # ---------- 5.4 根据接收机钟差 & 卫星钟差校正伪距 ----------
            nav.channel.correctedP[[ch - 1 for ch in active_now], epoch_idx] = (
                raw_p_used + sat_clk_corr * settings.c + nav.dt[epoch_idx]
            )

            # ---------- 5.5 坐标转换 ----------
            lat, lon, hgt = cart2geo(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], 5)
            nav.latitude[epoch_idx] = lat
            nav.longitude[epoch_idx] = lon
            nav.height[epoch_idx] = hgt

            utm_zone = find_utm_zone(lat, lon)
            nav.utmZone = utm_zone

            E, N, U = cart2utm(nav.X[epoch_idx], nav.Y[epoch_idx], nav.Z[epoch_idx], utm_zone)
            nav.E[epoch_idx] = E
            nav.N[epoch_idx] = N
            nav.U[epoch_idx] = U

        else:
            # 可用卫星不足 4 颗，当前位置解算失败
            print(f"   测量历元 No. {curr_meas_nr}: 信息不足，无法进行位置解算。")

            nav.X[epoch_idx] = np.nan
            nav.Y[epoch_idx] = np.nan
            nav.Z[epoch_idx] = np.nan
            nav.dt[epoch_idx] = np.nan
            nav.DOP[:, epoch_idx] = 0.0
            nav.latitude[epoch_idx] = np.nan
            nav.longitude[epoch_idx] = np.nan
            nav.height[epoch_idx] = np.nan
            nav.E[epoch_idx] = np.nan
            nav.N[epoch_idx] = np.nan
            nav.U[epoch_idx] = np.nan

            nav.channel.az[[ch - 1 for ch in active_now], epoch_idx] = np.nan
            nav.channel.el[[ch - 1 for ch in active_now], epoch_idx] = np.nan

        # ---------- 5.6 更新时间（发射时间 / 测量时间） ----------
        transmit_time += settings.navSolPeriod / 1000.0

    return nav, eph
