# src/gnss/navigation/ephemeris/ephemeris.py

"""
GPS Ephemeris 解码 (支持 MATLAB 完全等价 + 自动对齐)

- 默认：严格 MATLAB 模式（auto_align=False）
- 可选：启用自动对齐（auto_align=True）
"""

from typing import Tuple, Dict, Union, Sequence, Optional, List
import numpy as np

from gnss.utils.twos_comp import twos_comp2dec


gpsPi = 3.1415926535898


# --------------------- Bits 转换 --------------------- #

def _bits_to_str(bits: Union[str, Sequence[int]]) -> str:
    """统一转换为 '0'/'1' 字符串"""
    if isinstance(bits, str):
        return bits
    arr = np.asarray(bits).astype(int).flatten()
    return "".join("1" if b else "0" for b in arr)


def _invert_bits(bits_str: str) -> str:
    return "".join("0" if b == "1" else "1" for b in bits_str)


# --------------------- 自动对齐评分 --------------------- #

def _score_alignment(bits_str: str, start: int) -> int:
    """在 bits_str[start:] 内尝试识别 5 个子帧的 ID，返回落在 1~5 的数量"""
    score = 0
    for i in range(5):
        s0 = start + 300 * i
        s1 = s0 + 300
        if s1 > len(bits_str):
            break
        sf = bits_str[s0:s1]
        if len(sf) >= 52:
            sid = int(sf[49:52], 2)   # MATLAB (50:52)
            if 1 <= sid <= 5:
                score += 1
    return score


# ===============================================================
# 🔥 正确 MATLAB checkPhase 版本（你刚提供的）
# ===============================================================

def _checkPhase(word30: str, D30Star: Optional[str]) -> str:
    """
    正确的 MATLAB 极性校正：

    - 如果上一个字的 D30Star == '1'
      则将当前 word(1:24) 反相（奇偶校验 25~30 保持不变）

    word30：长度 30 的 '0'/'1'
    """
    if D30Star != '1':
        return word30

    # 反相前 24 bit
    data = word30[:24]
    parity = word30[24:30]

    data_inv = "".join("0" if b == "1" else "1" for b in data)

    return data_inv + parity


# ===============================================================
# 🔥 核心：MATLAB 等价的 1500bit 解码
# ===============================================================

def _decode_ephemeris_1500(bits_1500: str, D30Star: Optional[str]):
    """
    输入已经对齐的 1500bit（5 个子帧）
    执行：
      - 正确的逐 word 极性校正
      - 子帧 ID 解码
      - 星历字段解码（完全复制 MATLAB bit 范围）

    返回：
      eph: dict | None
      TOW: int  | None
      newD30Star
      found_ids
      id_bits
    """

    # 为了可编辑，把字符串变为 list
    bit_list = list(bits_1500)

    eph: Dict[str, float] = {}
    found_ids: List[int] = []
    id_bits: List[str] = []

    # ==== 依次处理 5 个子帧 ====
    for i in range(5):
        s0 = 300 * i
        s1 = s0 + 300

        subframe = bit_list[s0:s1]     # list
        subframe_str = "".join(subframe)

        # ---- 逐 word 极性校正 ----
        for w in range(10):
            w0 = 30 * w
            w1 = w0 + 30
            word = subframe_str[w0:w1]

            # MATLAB 等价 checkPhase
            corrected = _checkPhase(word, D30Star)

            # 写回
            for k in range(30):
                subframe[w0 + k] = corrected[k]

            # 更新 D30Star
            D30Star = corrected[-1]

        # 更新整个 bit_list
        bit_list[s0:s1] = subframe
        subframe_str = "".join(subframe)

        # ---- 解子帧 ID ----
        sid_bits = subframe_str[49:52]      # MATLAB (50:52)
        sid = int(sid_bits, 2)
        found_ids.append(sid)
        id_bits.append(sid_bits)

        # ---- 解析子帧内容（完全按 MATLAB bit 范围） ----
        if sid == 1:
            eph["weekNumber"] = int(subframe_str[60:70], 2) + 1024
            eph["accuracy"] = int(subframe_str[72:76], 2)
            eph["health"] = int(subframe_str[76:82], 2)

            eph["T_GD"] = twos_comp2dec(subframe_str[196:204]) * 2 ** (-31)
            eph["IODC"] = int(subframe_str[82:84] + subframe_str[196:204], 2)

            eph["t_oc"] = int(subframe_str[218:234], 2) * 2 ** 4
            eph["a_f2"] = twos_comp2dec(subframe_str[240:248]) * 2 ** (-55)
            eph["a_f1"] = twos_comp2dec(subframe_str[248:264]) * 2 ** (-43)
            eph["a_f0"] = twos_comp2dec(subframe_str[270:292]) * 2 ** (-31)

        elif sid == 2:
            eph["IODE_sf2"] = int(subframe_str[60:68], 2)
            eph["C_rs"] = twos_comp2dec(subframe_str[68:84]) * 2 ** (-5)
            eph["deltan"] = twos_comp2dec(subframe_str[90:106]) * 2 ** (-43) * gpsPi

            eph["M_0"] = (
                twos_comp2dec(subframe_str[106:114] + subframe_str[120:144])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_uc"] = twos_comp2dec(subframe_str[150:166]) * 2 ** (-29)
            eph["e"] = int(subframe_str[166:174] + subframe_str[180:204], 2) * 2 ** (-33)
            eph["C_us"] = twos_comp2dec(subframe_str[210:226]) * 2 ** (-29)

            eph["sqrtA"] = int(subframe_str[226:234] + subframe_str[240:264], 2) * 2 ** (-19)
            eph["t_oe"] = int(subframe_str[270:286], 2) * 2 ** 4

        elif sid == 3:
            eph["C_ic"] = twos_comp2dec(subframe_str[60:76]) * 2 ** (-29)

            eph["omega_0"] = (
                twos_comp2dec(subframe_str[76:84] + subframe_str[90:114])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_is"] = twos_comp2dec(subframe_str[120:136]) * 2 ** (-29)

            eph["i_0"] = (
                twos_comp2dec(subframe_str[136:144] + subframe_str[150:174])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_rc"] = twos_comp2dec(subframe_str[180:196]) * 2 ** (-5)

            eph["omega"] = (
                twos_comp2dec(subframe_str[196:204] + subframe_str[210:234])
                * 2 ** (-31)
                * gpsPi
            )

            eph["omegaDot"] = twos_comp2dec(subframe_str[240:264]) * 2 ** (-43) * gpsPi

            eph["IODE_sf3"] = int(subframe_str[270:278], 2)
            eph["iDot"] = twos_comp2dec(subframe_str[278:292]) * 2 ** (-43) * gpsPi

    # ---- 检查必要字段 ----
    essential = ["sqrtA", "t_oe", "M_0", "e"]
    if any(k not in eph for k in essential):
        return None, None, D30Star, found_ids, id_bits

    # ---- 用第 5 个子帧计算 TOW ----
    sf5 = "".join(bit_list[1200:1500])
    TOW = int(sf5[30:47], 2) * 6 - 30

    return eph, TOW, D30Star, found_ids, id_bits


# ===============================================================
# 🔥 对外主函数：ephemeris()
# ===============================================================

def ephemeris(bits: Union[str, Sequence[int]],
              D30Star: Optional[str] = None,
              auto_align: bool = False):
    """
    解码 GPS 广播星历（支持 MATLAB 等价模式 + 自动对齐模式）

    bits      : '0'/'1' 或 数组
    D30Star   : 上一 word 的第 30 位
    auto_align: 是否启用自动对齐
    """
    bits_str = _bits_to_str(bits)
    L = len(bits_str)

    if L < 1500:
        print(f"[EPH] 输入比特不足 1500 (len={L})")
        return None, None

    # ----------------------------------------------------
    # 模式 A：严格 MATLAB（不自动对齐）
    # ----------------------------------------------------
    if not auto_align:
        sub = bits_str[:1500]
        eph, TOW, _, ids, _ = _decode_ephemeris_1500(sub, D30Star)
        if eph is None:
            print(f"[EPH FAIL] (MATLAB mode) IDs={ids}")
        return eph, TOW

    # ----------------------------------------------------
    # 模式 B：自动对齐 + 自动反相
    # ----------------------------------------------------
    inv = _invert_bits(bits_str)
    candidates = [("Normal", bits_str), ("Inverted", inv)]

    best_mode = None
    best_start = None
    best_score = -1
    best_bits = None

    search_limit = min(L - 1500, 600)

    # --- 搜索最佳起点 ---
    for mode, bstr in candidates:
        for s in range(search_limit + 1):
            sc = _score_alignment(bstr, s)
            if sc > best_score:
                best_score = sc
                best_mode = mode
                best_start = s
                best_bits = bstr
            if sc == 5:
                break

    if best_score <= 0:
        print("[EPH FAIL] Auto-align failed.")
        return None, None

    aligned = best_bits[best_start:best_start + 1500]
    eph, TOW, _, ids, id_bits = _decode_ephemeris_1500(aligned, D30Star)

    if eph is None:
        print(f"[EPH FAIL] AutoAlign IDs={ids}, bits={id_bits}")
    return eph, TOW


# ===============================================================
# check_t (你的版本保持不变)
# ===============================================================

def check_t(time: float) -> float:
    half_week = 302400
    if time > half_week:
        return time - 2 * half_week
    if time < -half_week:
        return time + 2 * half_week
    return time
