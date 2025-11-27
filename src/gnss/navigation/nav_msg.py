"""导航电文相关工具，对应 navPartyChk.m、findPreambles.m 等 (占位)。"""

# src/gnss/navigation/nav_msg.py

from typing import Sequence, List, Tuple
import numpy as np

from .ephemeris.nav_party_chk import nav_party_chk  # 需要你实现这个函数


def find_preambles(
    track_results: Sequence,
    settings,
) -> Tuple[np.ndarray, List[int]]:
    """
    在各通道跟踪输出比特流中寻找第一个有效 GPS 前导码 (TLM preamble) 的位置，
    并返回具有有效前导码的活动通道列表。

    参数
    ----
    track_results : 序列（list / tuple）
        跟踪结果，每个元素对应一个通道。
        每个元素需要包含：
            - status : 字符，'-' 表示未跟踪，其它表示正在跟踪
            - I_P    : 一维数组，按 1 ms 抽取的 I_P 积分结果（导航比特流，单位：ms）
    settings : 配置对象
        需要字段：
            - numberOfChannels : 通道总数
            - 其它字段此处未使用

    返回
    ----
    first_sub_frame : np.ndarray, shape (numberOfChannels,)
        每个通道第一个前导码的位置（从跟踪开始计数的毫秒数，1-based）。
        未找到则为 0。
    active_chn_list : List[int]
        包含有效前导码的通道号列表（1-based）。
    """

    # 可以推迟前导码搜索到跟踪后期，这里和原代码一样从 0 开始
    search_start_offset = 0

    n_ch = settings.numberOfChannels

    # --- 初始化 firstSubFrame 数组 ---------------------------------------
    first_sub_frame = np.zeros(n_ch, dtype=int)

    # --- 生成前导码模式 --------------------------------------------------
    # GPS L1 C/A 导航电文前导码比特：10001011
    # MATLAB 中用 +1/-1 形式；这里沿用
    preamble_bits = np.array([1, -1, -1, -1, 1, -1, 1, 1], dtype=float)

    # 每比特持续 20 ms（跟踪结果每 1 ms 一个样本）
    preamble_ms = np.kron(preamble_bits, np.ones(20, dtype=float))  # 长度 160

    # --- 创建“正在跟踪”的通道列表（1-based） ----------------------------
    active_chn_list_all: List[int] = []
    for ch in range(1, n_ch + 1):
        tr = track_results[ch - 1]
        status = getattr(tr, "status", "-")
        if status != "-":
            active_chn_list_all.append(ch)

    # 用来记录最终“仍然有效”的通道
    valid_chn_list: List[int] = []

    # === 对所有正在跟踪的通道进行处理 =====================================
    for channel_nr in active_chn_list_all:
        idx_ch = channel_nr - 1  # Python 0-based

        # 读取该通道的 I_P 导航比特流（ms 级）
        tr = track_results[idx_ch]
        if hasattr(tr, "I_P"):
            bits = np.array(tr.I_P, dtype=float)
        elif isinstance(tr, dict) and "I_P" in tr:
            bits = np.array(tr["I_P"], dtype=float)
        else:
            print(f"[find_preambles] 通道 {channel_nr} 中没有 I_P 字段，跳过。")
            continue

        # 跳过前面的 offset（避免跟踪瞬态）
        bits = bits[search_start_offset:]

        if bits.size < 2000:  # 太短就没必要找前导码
            print(f"[find_preambles] 通道 {channel_nr} 比特流太短，跳过。")
            continue

        # --- 硬判决成 ±1 ------------------------------------------------
        bits_hard = np.where(bits > 0, 1.0, -1.0)

        # --- 互相关：对应 MATLAB xcorr(bits, preamble_ms) ----------------
        # numpy.correlate(full) 与 MATLAB xcorr 在“整体峰值位置检测”上足够对齐
        tlm_xcorr_result = np.correlate(bits_hard, preamble_ms, mode="full")

        # xcorrLength = (length(tlmXcorrResult) + 1) / 2
        L = tlm_xcorr_result.size
        xcorr_length = (L + 1) // 2  # 中心（包含零延迟）位置（1-based -> 0-based 后要减 1）

        # MATLAB:
        # tlmXcorrResult(xcorrLength : xcorrLength*2-1)
        # -> Python: tlm_xcorr_result[xcorr_length-1 : ]
        pos_half = tlm_xcorr_result[xcorr_length - 1 :]

        # --- 寻找所有疑似前导码起始点（阈值 153） -----------------------
        # find(abs(pos_half) > 153)' + searchStartOffset
        indices_rel = np.nonzero(np.abs(pos_half) > 153.0)[0]  # 0-based
        # lag 对应 0,1,2,... -> 累加 +1（MATLAB 1-based）再加 search_start_offset
        index = indices_rel + 1 + search_start_offset  # 仍然是 1-based 毫秒索引

        if index.size == 0:
            # 后面统一打印“未找到有效前导码”
            print(f"在通道 {channel_nr} 中未找到疑似前导码峰值。")
            continue

        # === 分析每个疑似前导码位置 ======================================
        found_for_this_channel = False

        for idx_i in index:
            # idx_i 是当前疑似前导码起始位置（毫秒，1-based）

            # 计算与其它疑似位置的时间差，寻找相差 6000 ms 的另一个前导码
            index2 = index - idx_i

            if 6000 not in index2:
                continue  # 没有恰好差 6000ms 的位置，则继续看下一个峰

            # === 重新读取比特值进行前导码验证 =============================
            # 需要从 I_P 里取：
            #   前一子帧最后 2bit + 当前子帧前 60bit = 共 62bit
            # 每 bit 为 20ms -> 总 1240 个 1ms 样本
            start_ms = idx_i - 40           # (2 bit * 20ms = 40ms)
            end_ms = idx_i + 20 * 60 - 1    # 共 1240 个

            # 边界检查
            if start_ms < 1 or end_ms > bits.size:
                # 原 code 没太多边界判断，这里多一道保险
                continue

            # 映射到 Python 0-based 索引：
            # MATLAB start_ms .. end_ms (inclusive)
            # -> Python [start_ms-1 : end_ms)
            seg = bits[start_ms - 1 : end_ms].copy()

            if seg.size != 1240:
                # 理论上应为 62 * 20 = 1240
                continue

            # 将样本按 20ms 一组形成 (20, N) 矩阵，每列对应一个比特
            # MATLAB: reshape(bits, 20, size(bits,1)/20) (column-major)
            # Python 中可以：先按 (Nbit, 20) 再转置
            seg_mat = seg.reshape(-1, 20).T  # 形状 (20, Nbit)

            # 对每列求和 -> “积分与倾泄”
            bits_soft = np.sum(seg_mat, axis=0)

            # 再硬判决成 ±1 比特
            bits_hard2 = np.where(bits_soft > 0, 1, -1)

            if bits_hard2.size < 62:
                continue

            # --- 奇偶校验：nav_party_chk ---------------------------------
            # 第一个字：前一字的最后 2bit + TLM 字 30bit -> 共 32bit
            word1 = bits_hard2[0:32]
            # 第二个字：TLM 字最后 2bit + HOW 字 30bit -> 共 32bit
            word2 = bits_hard2[30:62]

            if nav_party_chk(word1) != 0 and nav_party_chk(word2) != 0:
                # 通过奇偶校验，确认是真实前导码
                first_sub_frame[idx_ch] = int(idx_i)
                found_for_this_channel = True
                break  # 跳出此通道的峰值循环

        if found_for_this_channel:
            valid_chn_list.append(channel_nr)
        else:
            print(f"在通道 {channel_nr} 中未找到有效的前导码!")

    return first_sub_frame, valid_chn_list
