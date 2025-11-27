# src/gnss/acquisition/ca_code.py

"""
generate_ca_code(PRN)
生成指定 GPS 卫星（PRN 1–32）的 C/A 码序列。

对应 MATLAB 的 generateCAcode.m。
"""

import numpy as np
from typing import Any


def generate_ca_code(PRN: int) -> np.ndarray:
    """
    Python 版 GPS C/A 码生成函数
    返回: 长度 1023、取值为 -1 / +1 的 numpy 数组
    """

    # --- 创建 G2 码偏移数组 (PRN 1–32) ---
    g2s = [
          5,   6,   7,   8,  17,  18, 139, 140, 141, 251,   # PRN 1-10
        252, 254, 255, 256, 257, 258, 469, 470, 471, 472,   # PRN 11-20
        473, 474, 509, 512, 513, 514, 515, 516, 859, 860,   # PRN 21-30
        861, 862                                            # PRN 31-32
    ]

    if PRN < 1 or PRN > 32:
        raise ValueError("PRN must be between 1 and 32")

    g2shift = g2s[PRN - 1]

    # ==================== 生成 G1 ====================
    g1 = np.zeros(1023)
    reg = -1 * np.ones(10)  # 用 -1 表示“1”

    for i in range(1023):
        g1[i] = reg[9]
        saveBit = reg[2] * reg[9]      # taps 3,10
        reg[1:10] = reg[0:9]
        reg[0] = saveBit

    # ==================== 生成 G2 ====================
    g2 = np.zeros(1023)
    reg = -1 * np.ones(10)

    for i in range(1023):
        g2[i] = reg[9]
        saveBit = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9]  # taps 2,3,6,8,9,10
        reg[1:10] = reg[0:9]
        reg[0] = saveBit

    # ==================== 循环移位 G2 ====================
    g2 = np.concatenate((g2[1023 - g2shift :], g2[: 1023 - g2shift]))

    # ==================== C/A 码 ====================
    CAcode = -(g1 * g2)
    return CAcode


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


def make_ca_table(settings) -> np.ndarray:
    """
    根据 settings 生成所有 32 颗卫星的数字化 C/A 码表。

    返回: caCodesTable, shape = (32, samplesPerCode)
    每一行对应一个 PRN (1..32)。
    """

    samplingFreq = _get_setting(settings, "samplingFreq")
    codeFreqBasis = _get_setting(settings, "codeFreqBasis")
    codeLength = _get_setting(settings, "codeLength")

    samplesPerCode = int(
        round(
            samplingFreq
            / (codeFreqBasis / codeLength)
        )
    )

    caCodesTable = np.zeros((32, samplesPerCode))

    ts = 1.0 / samplingFreq
    tc = 1.0 / codeFreqBasis

    for PRN in range(1, 33):
        caCode = generate_ca_code(PRN)  # 长度 1023，-1/+1

        n = np.arange(1, samplesPerCode + 1)
        codeValueIndex = np.ceil((ts * n) / tc).astype(int)
        codeValueIndex[-1] = 1023  # 防止 1024 越界

        caCodesTable[PRN - 1, :] = caCode[codeValueIndex - 1]

    return caCodesTable
