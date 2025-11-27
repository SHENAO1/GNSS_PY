# src/gnss/settings.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class TruePosition:
    """天线真值位置（UTM），等价于 MATLAB 的 settings.truePosition.E/N/U"""
    E: float = math.nan
    N: float = math.nan
    U: float = math.nan


@dataclass
class Settings:
    """
    SoftGNSS 风格的配置，字段名尽量保持与 MATLAB 版本一致（驼峰命名），
    这样你在其他 Python 模块里可以直接用 settings.samplingFreq 等，
    和原 MATLAB 代码几乎一模一样。
    """

    # ===== 处理设置 ======================================================
    msToProcess: int = 37000          # 要处理的数据时长 [ms]
    numberOfChannels: int = 8         # 通道数
    skipNumberOfBytes: int = 0        # 从文件开头跳过的字节数

    # ===== 原始信号文件及相关参数 ========================================
    # 注意：这里直接保留你 MATLAB 里的绝对路径，你可以视情况自行修改
    """     
    fileName: str = (
        r"E:\BaiduSyncdisk\CUC\Code_Online\MATLAB_code_Gongwei\GNSS\data"
        r"\GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin"
    ) """

    fileName: str = (
        r"E:\BaiduSyncdisk\CUC\Code_Online\MATLAB_code_Gongwei\GNSS\data"
        r"\GPSdata-DiscreteComponents-fs38_192-if9_55.bin"
    )
    dataType: str = "int8"            # 文件中单个采样点的数据类型

    # 中频、采样频率和码率
    IF: float = 9.548e6               # 中频 [Hz]
    samplingFreq: float = 38.192e6    # 采样频率 [Hz]
    codeFreqBasis: float = 1.023e6    # C/A 码码片速率 [Hz]
    codeLength: int = 1023            # C/A 码长度 [chip]

    # ===== 捕获设置 ======================================================
    skipAcquisition: int = 0          # 1 -> 跳过捕获阶段
    acqSatelliteList: List[int] = field(
        default_factory=lambda: list(range(1, 33))
    )                                  # 要搜索的 PRN 列表
    acqSearchBand: float = 14.0        # 捕获搜索带宽（单边）[kHz]
    acqThreshold: float = 2.5          # 峰值比判决门限

    # ===== 跟踪环路设置 ==================================================
    # DLL
    dllDampingRatio: float = 0.7
    dllNoiseBandwidth: float = 2.0     # [Hz]
    dllCorrelatorSpacing: float = 0.5  # [chip]

    # PLL
    pllDampingRatio: float = 0.7
    pllNoiseBandwidth: float = 25.0    # [Hz]

    # ===== 导航解算设置 ==================================================
    navSolPeriod: int = 500            # 导航解算周期 [ms]
    elevationMask: float = 10.0        # 仰角掩码 [deg]
    useTropCorr: int = 1               # 是否使用对流层改正 (0/1)

    # 天线真值位置（如果已知），未知则为 NaN
    truePosition: TruePosition = field(default_factory=TruePosition)

    # ===== 绘图设置 ======================================================
    plotTracking: int = 1              # 是否绘制跟踪结果图 (0/1)

    # ===== 常量 =========================================================
    c: float = 299_792_458.0           # 光速 [m/s]
    startOffset: float = 68.802        # 初始传播时间估计 [ms]


def init_settings() -> Settings:
    """
    Python 版 initSettings，对应 MATLAB 的 initSettings.m

    用法:
        from gnss.settings import init_settings
        settings = init_settings()
    """
    return Settings()
