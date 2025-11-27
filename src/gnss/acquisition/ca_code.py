"""
generateCAcode(PRN)
生成指定 GPS 卫星（PRN 1–32）的 C/A 码序列。

该函数对应 MATLAB 文件 generateCAcode.m，
并完全保持原始注释与逻辑。

输入:
    PRN : 卫星 PRN 号（1–32）

输出:
    CAcode : numpy 数组，包含长度 1023 的 C/A 码（取值为 -1 或 +1）
"""

import numpy as np


def generate_ca_code(PRN: int):
    """
    Python 版 GPS C/A 码生成函数
    """

    # --- 创建码偏移数组。这个偏移量取决于输入的PRN号 ---
    # g2s 向量存储了用于生成特定 C/A 码时，G2 码所需的相位偏移量。
    # 例如：对于 19 号卫星(SV#19)，需要使用 g2s[18] = 471 的偏移量（Python 下标从0开始）。
    g2s = [
          5,   6,   7,   8,  17,  18, 139, 140, 141, 251,   # PRN 1-10
        252, 254, 255, 256, 257, 258, 469, 470, 471, 472,   # PRN 11-20
        473, 474, 509, 512, 513, 514, 515, 516, 859, 860,   # PRN 21-30
        861, 862                                            # PRN 31-32
        # 以下是增强型系统（WAAS/EGNOS）PRN 映射，但此处不使用
    ]

    # --- 根据给定的 PRN 编号选择偏移量 ---
    g2shift = g2s[PRN - 1]    # Python 下标从 0 开始

    # ==============================================================
    #                      生成 G1 码
    # ==============================================================

    # G1 是 10 位 LFSR，反馈多项式：
    #       G1(x) = 1 + x^3 + x^10

    g1 = np.zeros(1023)
    reg = -1 * np.ones(10)     # 初始寄存器全为1，用 -1 表示双极性

    for i in range(1023):
        g1[i] = reg[9]         # 输出 reg(10)
        saveBit = reg[2] * reg[9]   # 第 3、10 位（0-based: 2, 9）相乘
        reg[1:10] = reg[0:9]        # 右移
        reg[0] = saveBit

    # ==============================================================
    #                      生成 G2 码
    # ==============================================================

    # G2 的反馈多项式：
    # G2(x) = 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10

    g2 = np.zeros(1023)
    reg = -1 * np.ones(10)

    for i in range(1023):
        g2[i] = reg[9]
        saveBit = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9]
        reg[1:10] = reg[0:9]
        reg[0] = saveBit

    # ==============================================================
    #                   对 G2 码进行循环移位
    # ==============================================================

    # MATLAB: g2 = [g2(1023-g2shift+1:1023), g2(1:1023-g2shift)]
    g2 = np.concatenate((g2[1023 - g2shift:], g2[:1023 - g2shift]))

    # ==============================================================
    #            C/A 码 = - ( G1 .* G2 ) （逐元素相乘）
    # ==============================================================
    CAcode = -(g1 * g2)

    return CAcode


""" from gnss.acquisition.ca_code import generate_ca_code
import matplotlib.pyplot as plt

code = generate_ca_code(1)  # PRN 1

plt.plot(code)
plt.title("C/A Code for PRN 1")
plt.show() """



# make_ca_table(settings) - 根据 settings 生成所有卫星的数字化 C/A 码表

import numpy as np
from typing import Any

from gnss.acquisition.ca_code import generate_ca_code  # 如果就在同文件，可省略或改为相对导入


def _get_setting(settings: Any, name: str):
    """
    小工具：既支持 settings.xxx 也支持 settings['xxx'] 两种访问方式
    """
    if hasattr(settings, name):
        return getattr(settings, name)
    try:
        return settings[name]
    except (TypeError, KeyError):
        raise AttributeError(f"settings 中缺少字段: {name}")


def make_ca_table(settings):
    """
    功能: 根据提供的 "settings" 结构体中的参数，为所有32颗卫星生成C/A码。
          生成的C/A码会根据 "settings" 中指定的采样频率进行数字化。
          在输出的 "caCodesTable" 矩阵中，每一行代表一个卫星的C/A码，行号即为其PRN号。

    用法: caCodesTable = make_ca_table(settings)

        输入:
            settings        - 包含接收机设置的结构体/对象，需要包含以下字段：
                              - settings.samplingFreq: 接收机的采样频率 (Hz)
                              - settings.codeFreqBasis: C/A码的码片速率
                                (对于GPS C/A码，通常是 1.023e6 Hz)
                              - settings.codeLength: C/A码的长度 (对于GPS C/A码，是 1023)

        输出:
            caCodesTable    - 一个矩阵，包含了所有卫星PRN的数字化C/A码。
                              维度为 (32, samplesPerCode)。
    """

    # --- 计算每个C/A码周期内的采样点数 ------------------------------------
    # C/A码的重复频率 = 码片速率 / 码长 (1.023e6 / 1023 = 1000 Hz)
    # 每个码周期的采样点数 = 采样频率 / C/A码重复频率
    samplingFreq = _get_setting(settings, "samplingFreq")
    codeFreqBasis = _get_setting(settings, "codeFreqBasis")
    codeLength = _get_setting(settings, "codeLength")

    samplesPerCode = int(
        round(
            samplingFreq
            / (codeFreqBasis / codeLength)
        )
    )

    # --- 预分配输出矩阵内存，以提高函数执行速度 ---------------------------
    # 创建一个 32 行 x samplesPerCode 列的全零矩阵
    # 每一行将用来存储一颗卫星 (PRN 1-32) 的数字化C/A码
    caCodesTable = np.zeros((32, samplesPerCode))

    # --- 计算相关的时间常数 -----------------------------------------------
    ts = 1.0 / samplingFreq      # ADC的采样周期 (秒)，即两个采样点之间的时间间隔
    tc = 1.0 / codeFreqBasis     # C/A码单个码片的持续时间 (秒)

    # === 遍历所有32个卫星的PRN号 ...
    for PRN in range(1, 33):
        # --- 为给定的PRN号生成原始的C/A码 -----------------------------------
        # 调用一个外部函数来生成标准的1023个码片的C/A码序列
        caCode = generate_ca_code(PRN)    # 长度 1023，值为 -1 / +1

        # === 开始进行数字化/上采样过程 =======================================

        # --- 创建一个索引数组，用于从原始C/A码中读取码片值 -----------------
        # 这个索引数组的长度取决于采样频率，即一个C/A码周期(1毫秒)内的采样点数。
        # 核心思想：对于每个采样点，计算出它在时间上属于原始C/A码的第几个码片。
        # (ts * (1:samplesPerCode)) 计算出每个采样点的绝对时间戳。
        # ... / tc 将时间戳转换为以“码片周期”为单位的相对位置。
        # ceil(...) 向上取整，将每个采样点映射到其所属的码片索引。
        n = np.arange(1, samplesPerCode + 1)
        codeValueIndex = np.ceil((ts * n) / tc).astype(int)

        # --- 修正最后一个索引值（处理因浮点数舍入误差导致的问题）-----------
        # 由于计算误差，最后一个采样点的索引可能被计算为1024，这会超出
        # caCode 数组的边界(1-1023)。这里强制将其设为1023以避免错误。
        codeValueIndex[-1] = 1023

        # Python 下标从 0 开始，因此这里要减 1
        caCodesTable[PRN - 1, :] = caCode[codeValueIndex - 1]

    # end for PRN = 1:32

    return caCodesTable
