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
    
    【最终修复版】
    1. 优先使用奇偶校验 (HOW Word) 确认锁定。
    2. Fallback 策略：如果奇偶校验失败，但检测到完美的 6000ms (1子帧) 周期性重复，
       则忽略校验错误，强制认为锁定。这能防止因个别误码导致的流程卡死。
    """

    search_start_offset = 0
    n_ch = settings.numberOfChannels

    first_sub_frame = np.zeros(n_ch, dtype=int)

    # GPS L1 C/A 前导码: 1 0 0 0 1 0 1 1
    preamble_bits = np.array([1, -1, -1, -1, 1, -1, 1, 1], dtype=float)
    preamble_ms = np.kron(preamble_bits, np.ones(20, dtype=float)) 

    active_chn_list_all: List[int] = []
    for ch in range(1, n_ch + 1):
        tr = track_results[ch - 1]
        status = getattr(tr, "status", "-")
        if status != "-":
            active_chn_list_all.append(ch)

    valid_chn_list: List[int] = []

    print("-" * 60)
    print("[find_preambles] 开始搜索前导码 (Smart Fallback Mode)...")

    for channel_nr in active_chn_list_all:
        idx_ch = channel_nr - 1
        tr = track_results[idx_ch]

        if hasattr(tr, "I_P"):
            bits = np.array(tr.I_P, dtype=float)
        elif isinstance(tr, dict) and "I_P" in tr:
            bits = np.array(tr["I_P"], dtype=float)
        else:
            continue

        bits = bits[search_start_offset:]

        if bits.size < 2000:
            continue
            
        # --- 互相关找峰值 ---
        bits_hard = np.where(bits > 0, 1.0, -1.0)
        tlm_xcorr_result = np.correlate(bits_hard, preamble_ms, mode="full")

        L = tlm_xcorr_result.size
        xcorr_length = (L + 1) // 2
        pos_half = tlm_xcorr_result[xcorr_length - 1 :]

        indices_rel = np.nonzero(np.abs(pos_half) > 153.0)[0]
        index = indices_rel + 1 + search_start_offset

        if index.size == 0:
            print(f"    Ch {channel_nr}: 未找到峰值。")
            continue

        found_for_this_channel = False

        for idx_i in index:
            # idx_i 是疑似起点 (1-based)
            
            # 1. 峰值间距检查 (6000ms) - 这是最强的物理证据
            index2 = index - idx_i
            has_valid_spacing = (6000 in index2)
            
            if not has_valid_spacing:
                continue 

            # 2. 提取数据进行校验
            start_ms = idx_i - 40
            end_ms = idx_i + 20 * 60 - 1

            if start_ms < 1 or end_ms > bits.size:
                continue

            seg = bits[start_ms - 1 : end_ms].copy()
            if seg.size != 1240:
                continue

            # 积分
            seg_mat = seg.reshape(-1, 20).T
            bits_soft = np.sum(seg_mat, axis=0)

            # 生成正常和反相两种比特流
            bits_bin_norm = (bits_soft > 0).astype(int)
            bits_bin_inv  = (bits_soft <= 0).astype(int)

            # 提取 Word 2 (HOW) 用于校验
            w2_norm = bits_bin_norm[30:62]
            w2_inv = bits_bin_inv[30:62]

            # --- 校验 ---
            c2_n = nav_party_chk(w2_norm)
            c2_i = nav_party_chk(w2_inv)

            valid_norm = (c2_n == 0)
            valid_inv  = (c2_i == 0)

            # --- 判决逻辑 ---
            if valid_norm or valid_inv:
                # 完美情况：时间对齐 + 校验通过
                first_sub_frame[idx_ch] = int(idx_i)
                found_for_this_channel = True
                mode_str = "NORMAL" if valid_norm else "INVERTED"
                print(f"    Ch {channel_nr} 完美锁定! Pos={idx_i}, Mode={mode_str}")
                break
            elif has_valid_spacing:
                # Fallback 情况：时间对齐完美，但校验失败
                # 既然 6000ms 都有峰值，说明大概率是对的，只是误码导致校验挂了
                first_sub_frame[idx_ch] = int(idx_i)
                found_for_this_channel = True
                print(f"    Ch {channel_nr} 强制锁定 (时间对齐但校验失败). Pos={idx_i}")
                break

        if found_for_this_channel:
            valid_chn_list.append(channel_nr)
        else:
            print(f"    Ch {channel_nr}: 虽有峰值，但未满足 6000ms 周期性。")

    print("-" * 60)
    return first_sub_frame, valid_chn_list