# src/gnss/navigation/ephemeris/ephemeris.py

"""
ephemeris: 从给定的 GPS 导航电文比特流中解码出星历(ephemeris)和周内时(TOW)。
"""

from typing import Tuple, Dict, Union, Sequence, Optional
import numpy as np

from gnss.utils.twos_comp import twos_comp2dec

gpsPi = 3.1415926535898  # GPS 坐标系中使用的 Pi


def _bits_to_str(bits: Union[str, Sequence[int]]) -> str:
    """把输入的 bits 统一转换为 '0'/'1' 字符串。"""
    if isinstance(bits, str):
        return bits
    arr = np.asarray(bits).astype(int).flatten()
    return "".join("1" if b != 0 else "0" for b in arr)


def _invert_bits_str(bits_str: str) -> str:
    """将 0/1 字符串按位取反（解决 BPSK 180度相位模糊）"""
    return "".join("0" if b == "1" else "1" for b in bits_str)


def _score_alignment(bits_str: str, start: int) -> int:
    """
    给定一个候选起点 start，看看从这里开始的 5 个子帧
    （每个 300bit）里，有多少个子帧 ID 落在 [1,5]。
    """
    score = 0
    for i in range(5):
        sf_start = start + 300 * i
        sf_end = sf_start + 300
        if sf_end > len(bits_str):
            break
        sf = bits_str[sf_start:sf_end]
        if len(sf) < 52:
            continue
        # 子帧 ID：位 49-52 (Python index 49, 50, 51)
        sid = int(sf[49:52], 2)
        if 1 <= sid <= 5:
            score += 1
    return score


def ephemeris(bits: Union[str, Sequence[int]], D30Star: str = None) -> Tuple[Optional[Dict[str, float]], Optional[int]]:
    """
    Python 版 ephemeris 解码函数。
    包含详细的 ID 比特流 Debug 功能。
    """
    raw_bits_str = _bits_to_str(bits)

    if len(raw_bits_str) < 1500:
        print(f"[EPH ERROR] 输入比特长度不足 ({len(raw_bits_str)} < 1500)")
        return None, None

    # 准备两套比特流：正常 & 反相
    inverted_bits_str = _invert_bits_str(raw_bits_str)
    
    candidates = [
        ("Normal", raw_bits_str),
        ("Inverted", inverted_bits_str)
    ]

    best_mode = "Normal"
    best_bits = raw_bits_str
    best_start = 0
    max_score = -1

    # 搜索限制
    search_limit = min(len(raw_bits_str) - 1500, 600)

    # 1) 双向搜索最佳对齐点
    for mode, b_str in candidates:
        current_best_start = 0
        current_max_score = -1
        
        for s in range(search_limit + 1):
            score = _score_alignment(b_str, s)
            if score > current_max_score:
                current_max_score = score
                current_best_start = s
            if score == 5:
                break
        
        # 优先选择得分高的
        if current_max_score > max_score:
            max_score = current_max_score
            best_start = current_best_start
            best_bits = b_str
            best_mode = mode

    if max_score <= 0:
        # print(f"[EPH DEBUG] 无法对齐子帧 (Max Score=0)。")
        return None, None

    # 2) 截取并解析
    valid_bits = best_bits[best_start : best_start + 1500]

    eph: Dict[str, float] = {}
    found_ids = [] # 记录找到的 ID (整数)
    debug_id_bits = [] # 记录找到的 ID (原始比特串)

    for i in range(5):
        start = 300 * i
        end = 300 * (i + 1)
        subframe = valid_bits[start:end]
        
        # 提取 ID (第 49, 50, 51 位)
        id_str_local = subframe[49:52]
        subframeID = int(id_str_local, 2)
        
        found_ids.append(subframeID)
        debug_id_bits.append(id_str_local) # <--- 存下来

        if subframeID == 1:
            eph["weekNumber"] = int(subframe[60:70], 2) + 1024
            eph["accuracy"] = int(subframe[72:76], 2)
            eph["health"] = int(subframe[76:82], 2)
            eph["T_GD"] = twos_comp2dec(subframe[196:204]) * 2 ** (-31)
            eph["IODC"] = int(subframe[82:84] + subframe[196:204], 2)
            eph["t_oc"] = int(subframe[218:234], 2) * 2 ** 4
            eph["a_f2"] = twos_comp2dec(subframe[240:248]) * 2 ** (-55)
            eph["a_f1"] = twos_comp2dec(subframe[248:264]) * 2 ** (-43)
            eph["a_f0"] = twos_comp2dec(subframe[270:292],) * 2 ** (-31)

        elif subframeID == 2:
            eph["IODE_sf2"] = int(subframe[60:68], 2)
            eph["C_rs"] = twos_comp2dec(subframe[68:84]) * 2 ** (-5)
            eph["deltan"] = twos_comp2dec(subframe[90:106]) * 2 ** (-43) * gpsPi
            eph["M_0"] = (twos_comp2dec(subframe[106:114] + subframe[120:144]) * 2 ** (-31) * gpsPi)
            eph["C_uc"] = twos_comp2dec(subframe[150:166]) * 2 ** (-29)
            eph["e"] = int(subframe[166:174] + subframe[180:204], 2) * 2 ** (-33)
            eph["C_us"] = twos_comp2dec(subframe[210:226]) * 2 ** (-29)
            eph["sqrtA"] = int(subframe[226:234] + subframe[240:264], 2) * 2 ** (-19)
            eph["t_oe"] = int(subframe[270:286], 2) * 2 ** 4

        elif subframeID == 3:
            eph["C_ic"] = twos_comp2dec(subframe[60:76]) * 2 ** (-29)
            eph["omega_0"] = (twos_comp2dec(subframe[76:84] + subframe[90:114]) * 2 ** (-31) * gpsPi)
            eph["C_is"] = twos_comp2dec(subframe[120:136]) * 2 ** (-29)
            eph["i_0"] = (twos_comp2dec(subframe[136:144] + subframe[150:174]) * 2 ** (-31) * gpsPi)
            eph["C_rc"] = twos_comp2dec(subframe[180:196]) * 2 ** (-5)
            eph["omega"] = (twos_comp2dec(subframe[196:204] + subframe[210:234]) * 2 ** (-31) * gpsPi)
            eph["omegaDot"] = twos_comp2dec(subframe[240:264]) * 2 ** (-43) * gpsPi
            eph["IODE_sf3"] = int(subframe[270:278], 2)
            eph["iDot"] = twos_comp2dec(subframe[278:292]) * 2 ** (-43) * gpsPi

    # 检查完整性
    required_keys = ['sqrtA', 't_oe', 'M_0', 'e']
    missing = [k for k in required_keys if k not in eph]
    
    if missing:
        # === 详细调试信息 ===
        print(f"[EPH FAIL] Mode={best_mode}, Start={best_start}, Score={max_score}")
        print(f"           IDs found: {found_ids}")
        print(f"           ID bits  : {debug_id_bits}") 
        return None, None

    # 计算 TOW (使用第一个子帧)
    first_sf = valid_bits[0:300]
    TOW = int(first_sf[30:47], 2) * 6 - 30

    return eph, TOW


def check_t(time: float) -> float:
    """处理 GPS 时间的周内跳变"""
    half_week = 302400.0
    corr_time = time
    if time > half_week:
        corr_time = time - 2.0 * half_week
    elif time < -half_week:
        corr_time = time + 2.0 * half_week
    return corr_time