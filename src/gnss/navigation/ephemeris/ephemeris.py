"""
ephemeris: 从给定的 GPS 导航电文比特流中解码出星历(ephemeris)和周内时(TOW)。

该函数从给定的GPS导航电文比特流中解码出星历(ephemeris)和周内时(TOW)。
输入参数 BITS 必须包含1500个比特（即5个连续的子帧）。
数组的第一个元素必须是某个子帧的第一个比特。第一个子帧的ID是几并不重要。

注意：本函数当前版本 **不执行** 奇偶/极性校正（假设 nav_bits 进入时已经是正确极性）。

[用法]
    eph, TOW = ephemeris(bits, D30Star)

    输入:
        bits        - 导航电文的比特流 (包含5个子帧)。
                      可以是字符串(只含 '0'/'1')，也可以是 0/1 的 list/numpy 数组。
        D30Star     - 前一个导航字(word)的最后一个比特。此版本中不会用到。

    输出:
        TOW         - 比特流中第一个子帧的周内时 (单位：秒)。
        eph         - 解码出的卫星星历参数（使用 dict 存储，字段名与 MATLAB 结构体相同）。
"""

from typing import Tuple, Dict, Union, Sequence
import numpy as np

from gnss.utils.twos_comp import twos_comp2dec

gpsPi = 3.1415926535898  # GPS 坐标系中使用的 Pi


def _bits_to_str(bits: Union[str, Sequence[int]]) -> str:
    """
    把输入的 bits 统一转换为 '0'/'1' 字符串：
    - 如果本来就是 str，直接返回
    - 如果是 list / numpy array，则逐个元素转成 '0' 或 '1'
    """
    if isinstance(bits, str):
        return bits
    arr = np.asarray(bits).astype(int).flatten()
    return "".join("1" if b != 0 else "0" for b in arr)


def _score_alignment(bits_str: str, start: int) -> int:
    """
    给定一个候选起点 start，看看从这里开始的 5 个子帧
    （每个 300bit）里，有多少个子帧 ID 落在 [1,5]。
    用来挑选“最靠谱”的子帧起点。
    """
    score = 0
    for i in range(5):
        sf_start = start + 300 * i
        sf_end = sf_start + 300
        if sf_end > len(bits_str):
            break
        sf = bits_str[sf_start:sf_end]
        # 子帧 ID：位 50-52 （MATLAB 1-based）
        if len(sf) < 52:
            continue
        sid = int(sf[49:52], 2)
        if 1 <= sid <= 5:
            score += 1
    return score


def ephemeris(bits: Union[str, Sequence[int]], D30Star: str) -> Tuple[Dict[str, float], int]:
    """
    Python 版 ephemeris 解码函数（不做极性校正 + 自动对子帧起点对齐）
    """
    bits_str = _bits_to_str(bits)

    if len(bits_str) < 1500:
        raise ValueError("输入参数 bits 必须至少包含 1500 个比特!")

    # ========== 1) 自动搜索子帧起点（防止 nav_bits 没对齐） ==========
    max_start = len(bits_str) - 1500
    best_start = 0
    best_score = -1

    # 限制一下搜索范围，最多看前 600bit，防止 bits 特别长
    search_limit = min(max_start, 600)
    for s in range(0, search_limit + 1):
        sc = _score_alignment(bits_str, s)
        if sc > best_score:
            best_score = sc
            best_start = s

    if best_score <= 0:
        print(f"[EPH DEBUG] 未找到明显的子帧对齐点，使用 start=0（可能导致 ID 异常）。")
        best_start = 0
    else:
        print(f"[EPH DEBUG] 自动对子帧起点对齐: start={best_start}, score={best_score}/5")

    # 从最佳起点截取 5 个子帧（1500bit）
    bits_str = bits_str[best_start : best_start + 1500]

    # ========== 2) 开始按 SoftGNSS 逻辑解 5 个子帧 ==========
    eph: Dict[str, float] = {}
    last_subframe: str = ""

    for i in range(5):
        start = 300 * i
        end = 300 * (i + 1)
        subframe = bits_str[start:end]
        last_subframe = subframe

        if len(subframe) < 292:
            print(f"[EPH DEBUG] 子帧 #{i+1} 长度不足 292，比特: len={len(subframe)}，跳过。")
            continue

        # 子帧 ID：位 50-52
        subframeID = int(subframe[49:52], 2)
        print(f"[EPH DEBUG] subframe #{i+1}, ID = {subframeID}")

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

            eph["M_0"] = (
                twos_comp2dec(subframe[106:114] + subframe[120:144])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_uc"] = twos_comp2dec(subframe[150:166]) * 2 ** (-29)

            eph["e"] = int(subframe[166:174] + subframe[180:204], 2) * 2 ** (-33)

            eph["C_us"] = twos_comp2dec(subframe[210:226]) * 2 ** (-29)

            eph["sqrtA"] = int(subframe[226:234] + subframe[240:264], 2) * 2 ** (-19)

            eph["t_oe"] = int(subframe[270:286], 2) * 2 ** 4

        elif subframeID == 3:
            eph["C_ic"] = twos_comp2dec(subframe[60:76]) * 2 ** (-29)

            eph["omega_0"] = (
                twos_comp2dec(subframe[76:84] + subframe[90:114])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_is"] = twos_comp2dec(subframe[120:136]) * 2 ** (-29)

            eph["i_0"] = (
                twos_comp2dec(subframe[136:144] + subframe[150:174])
                * 2 ** (-31)
                * gpsPi
            )

            eph["C_rc"] = twos_comp2dec(subframe[180:196]) * 2 ** (-5)

            eph["omega"] = (
                twos_comp2dec(subframe[196:204] + subframe[210:234])
                * 2 ** (-31)
                * gpsPi
            )

            eph["omegaDot"] = twos_comp2dec(subframe[240:264]) * 2 ** (-43) * gpsPi

            eph["IODE_sf3"] = int(subframe[270:278], 2)

            eph["iDot"] = twos_comp2dec(subframe[278:292]) * 2 ** (-43) * gpsPi

        elif subframeID in (4, 5):
            # 暂时不解 4/5
            pass
        else:
            # 其他 ID（0、6、7）一般说明没对齐，这里只打印一下
            print(f"[EPH DEBUG] 子帧 #{i+1} 的 ID = {subframeID} (非常规 1~5)，未解码其内容。")

    # === 计算 TOW ===
    if not last_subframe:
        raise RuntimeError("ephemeris: 未能解析任何子帧。")

    TOW = int(last_subframe[30:47], 2) * 6 - 30

    return eph, TOW




def check_t(time: float) -> float:
    """
    SoftGNSS: check_t.m 的 Python 实现。
    处理 GPS 时间的周内跳变（week crossover）：
    如果时间差超过 ±302400 秒（半个 GPS 周），就需要折回。
    """
    half_week = 302400.0  # seconds

    corr_time = time

    if time > half_week:
        corr_time = time - 2.0 * half_week
    elif time < -half_week:
        corr_time = time + 2.0 * half_week

    return corr_time

# %%
