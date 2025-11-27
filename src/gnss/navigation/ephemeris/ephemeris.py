"""
ephemeris: 从给定的 GPS 导航电文比特流中解码出星历(ephemeris)和周内时(TOW)。

该函数从给定的GPS导航电文比特流中解码出星历(ephemeris)和周内时(TOW)。
输入参数 BITS 必须包含1500个比特（即5个连续的子帧）。
数组的第一个元素必须是某个子帧的第一个比特。第一个子帧的ID是几并不重要。

注意：本函数不执行奇偶校验！

[用法]
    eph, TOW = ephemeris(bits, D30Star)

    输入:
        bits        - 导航电文的比特流 (包含5个子帧)。
                      类型为字符串，且只应包含 '0' 或 '1'。
        D30Star     - 前一个导航字(word)的最后一个比特。用于数据比特的极性校正。
                      更多细节请参考GPS接口控制文件 ICD (IS-GPS-200D)。
                      参数类型为单个字符，且只应包含 '0' 或 '1'。

    输出:
        TOW         - 比特流中第一个子帧的周内时 (单位：秒)。
        eph         - 解码出的卫星星历参数（使用 dict 存储，字段名与 MATLAB 结构体相同）。
"""

import numpy as np
from gnss.navigation.ephemeris.nav_party_chk import check_t  # 对应 MATLAB 的 check_t


from typing import Tuple, Dict

from gnss.utils.signal_utils import check_phase          # 对应 MATLAB 的 checkPhase
from gnss.utils.twos_comp import twos_comp2dec          # 对应 MATLAB 的 twosComp2dec


# GPS 坐标系中使用的 Pi 值
gpsPi = 3.1415926535898


def ephemeris(bits: str, D30Star: str) -> Tuple[Dict[str, float], int]:
    """
    Python 版 ephemeris 解码函数
    """

    # %% 检查输入数据是否足够 ==========================================
    if len(bits) < 1500:
        raise ValueError("输入参数 bits 必须包含 1500 个比特!")

    # %% 检查输入参数类型 ============================================
    if not isinstance(bits, str):
        raise TypeError("输入参数 bits 必须是字符串!")

    if not isinstance(D30Star, str):
        raise TypeError("输入参数 D30Star 必须是字符!")

    # 星历结果使用 dict 保存（等价于 MATLAB 的结构体 eph）
    eph: Dict[str, float] = {}

    # %% 解码全部 5 个子帧 ==============================================
    for i in range(5):

        # --- "切割"出当前子帧的 300 个比特 --------------------------------
        start = 300 * i
        end = 300 * (i + 1)
        # 转成 list 方便就地修改单个比特
        subframe_list = list(bits[start:end])

        # --- 对全部 10 个字(word)中的数据比特进行极性校正 ----------------
        # GPS 信号解调可能存在 180 度相位模糊，导致比特流反相。
        # checkPhase 函数利用前一个字的最后一个比特(D30Star)来检测并纠正当前字的极性。
        for j in range(10):
            w_start = 30 * j
            w_end = 30 * (j + 1)
            word_str = "".join(subframe_list[w_start:w_end])

            # 调用 check_phase 函数对每个字（30 比特）进行处理
            word_str = check_phase(word_str, D30Star)

            # 更新子帧中的 30 个比特
            subframe_list[w_start:w_end] = list(word_str)

            # 更新 D30Star，用于下一个字的极性校正
            D30Star = word_str[-1]

        # 校正后的整个子帧
        subframe = "".join(subframe_list)

        # --- 解码子帧 ID ----------------------------------------------
        # 子帧 ID 位于第 2 个字（交接字 HOW）的第 2-4 位，
        # 即整个子帧的第 50-52 位（MATLAB: 50:52）。
        # Python 切片注意：位置 n (1-based) -> 索引 n-1 (0-based)，末端索引为 b (不含)。
        subframeID = int(subframe[49:52], 2)

        # --- 根据子帧 ID 解码子帧内容 --------------------------
        # 任务是选择必要的比特位，并将它们转换为十进制数。
        # 所有参数的比特位置、编码方式（有/无符号）和缩放因子
        # 都严格遵循 IS-GPS-200D 文档。
        if subframeID == 1:
            # --- 这是子帧 1 -------------------------------------
            # 包含：GPS 周数(WN), 卫星时钟校正参数, 卫星健康状态和精度。

            # GPS 周数 (WN)。+1024 是为了处理 1999 年 8 月发生的周数翻转问题。
            eph["weekNumber"] = int(subframe[60:70], 2) + 1024

            # 用户测距精度(URA)的指数
            eph["accuracy"] = int(subframe[72:76], 2)

            # 卫星健康状态 (6 比特)
            eph["health"] = int(subframe[76:82], 2)

            # 卫星设备群延迟差 (T_GD)，单位：秒
            eph["T_GD"] = twos_comp2dec(subframe[196:204]) * 2 ** (-31)

            # 数据发布期数（时钟）
            eph["IODC"] = int(subframe[82:84] + subframe[196:204], 2)

            # 时钟数据参考时刻 (t_oc)，单位：秒
            eph["t_oc"] = int(subframe[218:234], 2) * 2 ** 4

            # 卫星时钟漂移率 (a_f2)，单位：秒/秒^2
            eph["a_f2"] = twos_comp2dec(subframe[240:248]) * 2 ** (-55)

            # 卫星时钟漂移 (a_f1)，单位：秒/秒
            eph["a_f1"] = twos_comp2dec(subframe[248:264]) * 2 ** (-43)

            # 卫星时钟偏差 (a_f0)，单位：秒
            eph["a_f0"] = twos_comp2dec(subframe[270:292]) * 2 ** (-31)

        elif subframeID == 2:
            # --- 这是子帧 2 -------------------------------------
            # 包含第一部分星历参数

            # 数据发布期数（星历），子帧 2
            eph["IODE_sf2"] = int(subframe[60:68], 2)

            # 轨道半径正弦谐波校正振幅 (C_rs)，单位：米
            eph["C_rs"] = twos_comp2dec(subframe[68:84]) * 2 ** (-5)

            # 平近点角与计算值之差的校正量 (delta n)，单位：弧度/秒
            eph["deltan"] = twos_comp2dec(subframe[90:106]) * 2 ** (-43) * gpsPi

            # 参考时刻的平近点角 (M_0)，单位：弧度
            eph["M_0"] = (
                twos_comp2dec(subframe[106:114] + subframe[120:144])
                * 2 ** (-31)
                * gpsPi
            )

            # 纬度幅角余弦谐波校正振幅 (C_uc)，单位：弧度
            eph["C_uc"] = twos_comp2dec(subframe[150:166]) * 2 ** (-29)

            # 轨道偏心率 (e)，无单位
            eph["e"] = int(subframe[166:174] + subframe[180:204], 2) * 2 ** (-33)

            # 纬度幅角正弦谐波校正振幅 (C_us)，单位：弧度
            eph["C_us"] = twos_comp2dec(subframe[210:226]) * 2 ** (-29)

            # 轨道长半轴的平方根 (sqrt(A))，单位：米^(1/2)
            eph["sqrtA"] = int(subframe[226:234] + subframe[240:264], 2) * 2 ** (-19)

            # 星历参考时刻 (t_oe)，单位：秒
            eph["t_oe"] = int(subframe[270:286], 2) * 2 ** 4

        elif subframeID == 3:
            # --- 这是子帧 3 -------------------------------------
            # 包含第二部分星历参数

            # 轨道倾角余弦谐波校正振幅 (C_ic)，单位：弧度
            eph["C_ic"] = twos_comp2dec(subframe[60:76]) * 2 ** (-29)

            # GPS 周开始时刻的升交点赤经 (Omega_0)，单位：弧度
            eph["omega_0"] = (
                twos_comp2dec(subframe[76:84] + subframe[90:114])
                * 2 ** (-31)
                * gpsPi
            )

            # 轨道倾角正弦谐波校正振幅 (C_is)，单位：弧度
            eph["C_is"] = twos_comp2dec(subframe[120:136]) * 2 ** (-29)

            # 参考时刻的轨道倾角 (i_0)，单位：弧度
            eph["i_0"] = (
                twos_comp2dec(subframe[136:144] + subframe[150:174])
                * 2 ** (-31)
                * gpsPi
            )

            # 轨道半径余弦谐波校正振幅 (C_rc)，单位：米
            eph["C_rc"] = twos_comp2dec(subframe[180:196]) * 2 ** (-5)

            # 近地点角距 (omega)，单位：弧度
            eph["omega"] = (
                twos_comp2dec(subframe[196:204] + subframe[210:234])
                * 2 ** (-31)
                * gpsPi
            )

            # 升交点赤经变化率 (Omega Dot)，单位：弧度/秒
            eph["omegaDot"] = twos_comp2dec(subframe[240:264]) * 2 ** (-43) * gpsPi

            # 数据发布期数（星历），子帧 3
            eph["IODE_sf3"] = int(subframe[270:278], 2)

            # 轨道倾角变化率 (i Dot)，单位：弧度/秒
            eph["iDot"] = twos_comp2dec(subframe[278:292]) * 2 ** (-43) * gpsPi

        elif subframeID == 4:
            # --- 这是子帧 4 -------------------------------------
            # 包含：历书、电离层模型、UTC 参数、部分卫星健康状态（PRN 25-32）等。
            # 本代码目前未解码。
            pass

        elif subframeID == 5:
            # --- 这是子帧 5 -------------------------------------
            # 包含：大部分卫星的历书和健康状态（PRN 1-24）、历书参考周数和时刻。
            # 本代码目前未解码。
            pass

        # end if subframeID

    # end for i in range(5)

    # %% 计算数据块中第一个子帧的周内时 (TOW) ============================
    # 此时，变量 subframe 仍然是最后一个子帧（第 5 个子帧）的比特。
    # HOW 字中的 TOW 字段(位于子帧的 31-47 比特)给出的是 *下一个* 子帧的起始时刻。
    # 每个子帧传输 6 秒，因此该 TOW 值乘以 6 得到以秒为单位的时间。
    # 为了得到本数据块 *第一个* 子帧的起始时刻，需要从最后一个子帧解码出的 TOW 中，
    # 减去 5 个子帧的总时长 (5 * 6 = 30 秒)。
    TOW = int(subframe[30:47], 2) * 6 - 30

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
