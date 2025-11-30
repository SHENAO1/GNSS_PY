# src/gnss/navigation/nav_msg.py

from typing import Sequence, List, Tuple
import numpy as np

from .ephemeris.nav_party_chk import nav_party_chk


def find_preambles(
    track_results: Sequence,
    settings,
) -> Tuple[np.ndarray, List[int]]:
    """
    寻找各通道第一个有效 GPS 前导码位置 (TLM preamble)。

    尽量 1:1 复刻 SoftGNSS MATLAB 版：
    1. 用互相关找到所有疑似前导码起点；
    2. 检查 6000 ms 间隔的重复（1 个子帧）；
    3. 从 (idx-40) 开始取 1240 ms，合成 62 个 nav bits；
    4. 对 TLM 和 HOW 两个字分别做奇偶校验 (nav_party_chk)，两个都通过才接受。
    """

    search_start_offset = 0
    n_ch = settings.numberOfChannels

    # first_sub_frame[i] 存的是第 i 个通道“第一帧前导码”的 ms 起点 (1-based)
    first_sub_frame = np.zeros(n_ch, dtype=int)

    # GPS L1 C/A 前导码: 1 0 0 0 1 0 1 1  -> 用 ±1 表示
    preamble_bits = np.array([1, -1, -1, -1, 1, -1, 1, 1], dtype=float)
    # 每个比特持续 20 ms
    preamble_ms = np.kron(preamble_bits, np.ones(20, dtype=float))

    # 先把所有“有跟踪结果的通道”筛出来（status != '-'）
    active_chn_list_all: List[int] = []
    for ch in range(1, n_ch + 1):
        tr = track_results[ch - 1]
        status = getattr(tr, "status", "-")
        if status != "-":
            active_chn_list_all.append(ch)

    valid_chn_list: List[int] = []

    print("-" * 60)
    print("[find_preambles] 开始搜索前导码 (SoftGNSS-compatible 模式)...")

    for channel_nr in active_chn_list_all:
        idx_ch = channel_nr - 1
        tr = track_results[idx_ch]

        # 提取 I_P
        if hasattr(tr, "I_P"):
            bits = np.asarray(tr.I_P, dtype=float)
        elif isinstance(tr, dict) and "I_P" in tr:
            bits = np.asarray(tr["I_P"], dtype=float)
        else:
            continue

        # 从 search_start_offset 之后开始搜索
        bits = bits[search_start_offset:]

        if bits.size < 2000:
            print(f"    Ch {channel_nr}: 数据太短 (<2000 ms)，跳过。")
            continue

        # --- 互相关找峰值（完全类比 MATLAB xcorr(bits, preamble_ms)） ---
        bits_hard = np.where(bits > 0.0, 1.0, -1.0)
        tlm_xcorr_result = np.correlate(bits_hard, preamble_ms, mode="full")

        L = tlm_xcorr_result.size
        xcorr_length = (L + 1) // 2
        # 取“后半部分”（对应 MATLAB xcorrLength : 2*xcorrLength-1）
        pos_half = tlm_xcorr_result[xcorr_length - 1:]

        # 阈值 153.0 是 SoftGNSS 的经验值
        indices_rel = np.nonzero(np.abs(pos_half) > 153.0)[0]

        # 转成 1-based + 全局 offset
        index = indices_rel + 1 + search_start_offset

        if index.size == 0:
            print(f"    Ch {channel_nr}: 未找到峰值。")
            continue

        found_for_this_channel = False

        # === 遍历每一个疑似前导码起点 ===
        for idx_i in index:
            # idx_i 是疑似起点 (1-based, 单位 ms)

            # 1. 峰值间距检查 (6000ms)，与 MATLAB 完全一致
            index2 = index - idx_i
            if 6000 not in index2:
                continue

            # 2. 提取用于 parity 校验的 1240 ms 数据
            start_ms = idx_i - 40
            end_ms   = idx_i + 20 * 60 - 1  # 共 1240 ms

            if start_ms < 1 or end_ms > bits.size:
                continue

            seg = bits[start_ms - 1: end_ms].copy()
            if seg.size != 1240:
                continue

            # --- 合成 62 个 nav bits（20ms -> 1bit），完全对应 MATLAB ---
            # MATLAB: reshape(bits, 20, (size(bits,1)/20)); bits = sum(bits);
            seg_mat = np.reshape(seg, (20, -1), order="F")  # (20, 62)
            bits_soft = np.sum(seg_mat, axis=0)             # (62,)

            # 门限判决成 ±1
            bits_pm = np.where(bits_soft > 0.0, 1, -1)      # (62,)

            # --- TLM & HOW 两个字分别做 nav_party_chk ---
            w1 = bits_pm[0:32]   # bits(1:32) in MATLAB
            w2 = bits_pm[30:62]  # bits(31:62) in MATLAB

            c1 = nav_party_chk(w1)
            c2 = nav_party_chk(w2)

            if c1 != 0 and c2 != 0:
                first_sub_frame[idx_ch] = int(idx_i)
                found_for_this_channel = True
                print(
                    f"    Ch {channel_nr} preamble locked at {idx_i} ms, "
                    f"parity(TLM)={c1}, parity(HOW)={c2}"
                )
                break

        if found_for_this_channel:
            valid_chn_list.append(channel_nr)
        else:
            print(f"    Ch {channel_nr}: 虽有峰值，但未通过 TLM+HOW 奇偶校验。")

    print("-" * 60)
    return first_sub_frame, valid_chn_list
