"""
navPartyChk(ndat)
------------------------------------------------------------
功能:
    该函数用于计算和检查 GPS 导航电文“字”(word) 的奇偶校验位。
    其算法基于 GPS-SPS 信号规范第二版中的图 2-10 流程。

调用格式:
    status = nav_party_chk(ndat)

输入:
    ndat    - 一个 1×32 的数组，代表一个 GPS 导航字。
              包含 30 个当前字比特和 2 个来自前一字的比特。
              ndat 的结构为:
                [D29*, D30*, d1, d2, ..., d24, D25, D26, ..., D30]

              注意：ndat 必须已经转换为 “±1 表示法”：
                     -1 代表 bit = 0
                     +1 代表 bit = 1
              这是为了使用乘法模拟 XOR（异或）。

输出:
    status  - 校验状态:
                +1: 数据极性正确
                -1: 数据极性需要反相
                 0: 奇偶校验失败（数据损坏）
------------------------------------------------------------
"""

import numpy as np


def nav_party_chk(ndat: np.ndarray) -> int:
    """
    Python 版 navPartyChk

    参数:
        ndat: numpy 数组，长度 32，元素为 ±1
              ndat[0] = D29*
              ndat[1] = D30*
              ndat[2:26] = d1..d24
              ndat[26:32] = D25..D30

    返回:
        status: +1, -1, 或 0
    """

    # --- 异或(XOR)运算的乘法实现 -----------------------------------------
    # 使用 -1 表示 '0'，+1 表示 '1'
    #
    # a   b   xor    |   (-1=0, +1=1 表示法)
    # -----------------------------------------
    #  0   0    1    |   -1 * -1 = +1
    #  0   1    0    |   -1 * +1 = -1
    #  1   0    0    |   +1 * -1 = -1
    #  1   1    1    |   +1 * +1 = +1
    #
    # 因此 “乘法” 可以直接模拟 XOR。
    # -----------------------------------------------------------------------

    ndat = ndat.copy()  # 避免修改输入

    # --- 检查数据位是否需要反相 (用于内部计算) -----------------------------
    # GPS 规范规定：d1~d24 在计算奇偶校验前需要与 D30* 异或。
    #
    # MATLAB 代码: if (ndat(2) ~= 1)
    # 其中 ndat(2)=1 代表 bit=1，ndat(2)=-1 代表 bit=0。
    #
    # 所以：
    #   D30* = 0 → ndat[1] = -1 → 需要内部反相
    #   D30* = 1 → ndat[1] = +1 → 不反相
    # -----------------------------------------------------------------------
    if ndat[1] != 1:  # D30* 是 0 (即 ndat[1] == -1)
        ndat[2:26] = -ndat[2:26]

    # --- 根据 ICD-200C 表 20-XIV 计算 6 个奇偶校验位 -----------------------

    parity = np.zeros(6)

    # Python 下标注意：ndat[0] = D29*, ndat[1] = D30*, ndat[2]=d1
    parity[0] = np.prod([
        ndat[0], ndat[2], ndat[3], ndat[4], ndat[6],
        ndat[7], ndat[11], ndat[12], ndat[13], ndat[14],
        ndat[15], ndat[18], ndat[19], ndat[21], ndat[24]
    ])

    parity[1] = np.prod([
        ndat[1], ndat[3], ndat[4], ndat[5], ndat[7],
        ndat[8], ndat[12], ndat[13], ndat[14], ndat[15],
        ndat[16], ndat[19], ndat[20], ndat[22], ndat[25]
    ])

    parity[2] = np.prod([
        ndat[0], ndat[2], ndat[4], ndat[5], ndat[6],
        ndat[8], ndat[9], ndat[13], ndat[14], ndat[15],
        ndat[16], ndat[17], ndat[20], ndat[21], ndat[23]
    ])

    parity[3] = np.prod([
        ndat[1], ndat[3], ndat[5], ndat[6], ndat[7],
        ndat[9], ndat[10], ndat[14], ndat[15], ndat[16],
        ndat[17], ndat[18], ndat[21], ndat[22], ndat[24]
    ])

    parity[4] = np.prod([
        ndat[1], ndat[2], ndat[4], ndat[6], ndat[7],
        ndat[8], ndat[10], ndat[11], ndat[15], ndat[16],
        ndat[17], ndat[18], ndat[19], ndat[22], ndat[23], ndat[25]
    ])

    parity[5] = np.prod([
        ndat[0], ndat[4], ndat[6], ndat[7], ndat[9],
        ndat[10], ndat[11], ndat[12], ndat[14], ndat[16],
        ndat[20], ndat[23], ndat[24], ndat[25]
    ])

    # --- 与接收到的奇偶校验位比较 -----------------------------------------
    recv_parity = ndat[26:32]

    if np.sum(parity == recv_parity) == 6:
        # 奇偶校验通过
        #
        # 输出规则（与 MATLAB 一致）：
        #   如果 D30* = 1 (ndat[1]= +1) → status = -1 → 调用者需要反相
        #   如果 D30* = 0 (ndat[1]= -1) → status = +1 → 数据极性正确
        status = -1 * ndat[1]
    else:
        # 奇偶校验失败
        status = 0

    return int(status)


def check_t(time):
    """
    对应 MATLAB 的 check_t.m
    功能：修正周首 / 周末交叉处的时间（GPS 周滚动）

    输入:
        time : 标量或 numpy 数组，单位为秒 (s)

    输出:
        corr_time : 修正后的时间（同维度）
    """
    half_week = 302400.0  # 一周的一半，单位秒

    # 兼容标量和数组
    t = np.asarray(time, dtype=float)
    corr = t.copy()

    corr[corr > half_week] -= 2 * half_week
    corr[corr < -half_week] += 2 * half_week

    # 如果输入是标量，就返回标量，尽量模仿 MATLAB 使用习惯
    if np.isscalar(time):
        return float(corr)
    return corr
