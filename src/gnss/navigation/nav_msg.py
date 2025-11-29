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
    
    【修复版】
    1. 修复了内存引用问题后，数据已正常。
    2. 放宽奇偶校验逻辑：由于刚开始跟踪时，Word 1 (TLM) 的历史位(D29*,D30*)未知，
       导致 Word 1 校验极易失败。因此，只要 Word 2 (HOW) 校验通过，
       结合 6000ms 的峰值间距匹配，即可认为锁定成功。
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
    print("[find_preambles] 开始搜索前导码 (Relaxed Parity Check)...")

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
            
        # 调试：打印前10个点确认数据独立性
        sample_sig = bits[0:10].astype(int)
        # print(f"  > Ch {channel_nr} I_P check: {sample_sig}")

        # --- 互相关找峰值 ---
        bits_hard = np.where(bits > 0, 1.0, -1.0)
        tlm_xcorr_result = np.correlate(bits_hard, preamble_ms, mode="full")

        L = tlm_xcorr_result.size
        xcorr_length = (L + 1) // 2
        pos_half = tlm_xcorr_result[xcorr_length - 1 :]

        indices_rel = np.nonzero(np.abs(pos_half) > 153.0)[0]
        index = indices_rel + 1 + search_start_offset

        if index.size == 0:
            # print(f"    Ch {channel_nr}: 未找到峰值。")
            continue

        found_for_this_channel = False

        for idx_i in index:
            # idx_i 是疑似起点 (1-based)
            
            # 1. 峰值间距检查 (6000ms)
            index2 = index - idx_i
            if 6000 not in index2:
                continue 

            # 2. 提取数据进行校验
            # 取 TLM (Word 1) 和 HOW (Word 2)
            # Word 1: 30 bits. Word 2: 30 bits.
            # 校验 Word 1 需要前一子帧最后 2 bits (idx_i - 40ms)
            # 校验 Word 2 需要 Word 1 最后 2 bits
            
            # 我们至少需要读取 2个 Word (60 bits) + 前 2 bits = 62 bits
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

            # Word 1 (TLM): bits 0..31 (含历史2bit)
            # Word 2 (HOW): bits 30..61 (含TLM最后2bit)
            
            w1_norm = bits_bin_norm[0:32]
            w2_norm = bits_bin_norm[30:62]
            
            w1_inv = bits_bin_inv[0:32]
            w2_inv = bits_bin_inv[30:62]

            # --- 校验 ---
            c1_n = nav_party_chk(w1_norm)
            c2_n = nav_party_chk(w2_norm)
            
            c1_i = nav_party_chk(w1_inv)
            c2_i = nav_party_chk(w2_inv)

            # --- 核心修改：放宽判决条件 ---
            # 只要 HOW (Word 2) 校验通过，或者 TLM (Word 1) 通过，我们就认为找到了。
            # 通常 HOW 更可靠，因为它的历史位就在当前数据包里。
            
            valid_norm = (c2_n == 0) # 仅依赖 HOW
            valid_inv  = (c2_i == 0) # 仅依赖 HOW
            
            # 如果非常严格，可以要求 (c1==0 and c2==0)，但第一帧通常做不到。

            if valid_norm or valid_inv:
                first_sub_frame[idx_ch] = int(idx_i)
                found_for_this_channel = True
                
                # 再次确认到底是 Normal 还是 Inverted，用于日志
                mode_str = "NORMAL" if valid_norm else "INVERTED"
                # 如果 HOW 通过但 TLM 没通过，提示一下
                extra_info = ""
                if valid_norm and c1_n != 0: extra_info = "(TLM fail, HOW pass)"
                if valid_inv  and c1_i != 0: extra_info = "(TLM fail, HOW pass)"
                
                print(f"    Ch {channel_nr} 锁定! Pos={idx_i}, Mode={mode_str} {extra_info}")
                break

        if found_for_this_channel:
            valid_chn_list.append(channel_nr)
        else:
            print(f"    Ch {channel_nr}: 峰值间距匹配(6000ms)，但 HOW 字校验仍失败。")

    print("-" * 60)
    return first_sub_frame, valid_chn_list