"""
preRun(acqResults, settings)

功能：
    初始化所有跟踪通道（Tracking Channels）。
    根据捕获(acquisition)结果，将最强的卫星信号分配到各跟踪通道中，
    为后续 DLL/PLL 跟踪环路做好准备。

输入：
    acqResults:
        - acqResults.peakMetric: 每个 PRN 的峰值度量（强弱排序依据）
        - acqResults.carrFreq:   捕获得到的载波频率
        - acqResults.codePhase:  捕获得到的码相位

    settings:
        - settings.numberOfChannels: 跟踪通道数量（如 12 或 32）

输出：
    channels:
        一个列表，每个元素为一个 dict，表示一个通道的状态：
        {
            "PRN": int,          # 跟踪的卫星号
            "acquiredFreq": float,
            "codePhase": int,
            "status": "T" 或 "-"
        }
"""

import numpy as np
from typing import List, Dict, Any


def pre_run(acqResults: Any, settings: Any) -> List[Dict]:
    """
    Python 版 preRun() —— 初始化跟踪通道
    """

    # ===============================================================
    # 1. 创建所有通道的数据结构（初始为空闲状态）
    # ===============================================================
    channels = []

    # 单个通道的模板
    empty_channel = {
        "PRN": 0,             # 分配到的卫星 PRN 号
        "acquiredFreq": 0.0,  # 捕获载波频率（作为 NCO 起始值）
        "codePhase": 0,       # 捕获码相位（Tracking 起点）
        "status": "-"         # "-" 表示空闲；"T" 表示 Tracking 状态
    }

    # 复制模板，创建 numberOfChannels 个通道
    for _ in range(settings.numberOfChannels):
        channels.append(empty_channel.copy())

    # ===============================================================
    # 2. 根据 peakMetric 对卫星信号强度降序排序
    # ===============================================================
    # argsort 默认升序，这里加负号做降序
    PRN_indexes_sorted = np.argsort(-np.array(acqResults.peakMetric))

    # 成功捕获的卫星数量（频率为 0 的代表捕获失败）
    acquired_sat_count = np.sum(np.array(acqResults.carrFreq) > 0)

    # 实际可分配通道数量 = min(可用通道数, 捕获到的卫星数)
    usable_channels = min(settings.numberOfChannels, acquired_sat_count)

    # ===============================================================
    # 3. 将最强的卫星分配给前几个 Tracking 通道
    # ===============================================================
    for ii in range(usable_channels):
        prn = int(PRN_indexes_sorted[ii]) + 1    # +1 将索引转成 PRN 号（MATLAB 从1开始）

        channels[ii]["PRN"] = prn
        channels[ii]["acquiredFreq"] = float(acqResults.carrFreq[prn - 1])
        channels[ii]["codePhase"] = int(acqResults.codePhase[prn - 1])
        channels[ii]["status"] = "T"     # 激活通道，进入 Tracking 状态

    # ===============================================================
    # 未分配的通道仍为 "-" 状态
    # ===============================================================

    return channels
