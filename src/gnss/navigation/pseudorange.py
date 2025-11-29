"""伪距计算，对应 Matlab calculatePseudoranges.m。"""

# src/gnss/navigation/pseudorange.py

from typing import Sequence
import numpy as np


def calculate_pseudoranges(
    track_results: Sequence,
    ms_of_the_signal: Sequence[int],
    channel_list: Sequence[int],
    settings,
):
    """
    计算在给定毫秒采样点处，各通道的相对伪距（包含接收机时钟偏移）。

    参数说明略……
    """

    # ---------- 初始化传播时间 ----------
    n_ch = settings.numberOfChannels
    travel_time = np.full(n_ch, np.inf, dtype=float)

    # 每个 1 ms C/A 码的采样点数
    samples_per_code = int(
        round(settings.samplingFreq / (settings.codeFreqBasis / settings.codeLength))
    )

    # ---------- 修复 NumPy 2.0 报错 ----------
    # 原：channel_list = np.array(channel_list, copy=False).ravel()
    # 改：
    channel_list = np.asarray(channel_list).ravel()

    # ---------- 为每个通道计算传播时间 ----------
    for channel_nr in channel_list:
        idx = int(channel_nr) - 1  # MATLAB → Python 索引转换

        if idx < 0 or idx >= n_ch:
            continue  # 非法通道号直接跳过

        tr = track_results[idx]

        # 兼容多种字段名形式
        if hasattr(tr, "absoluteSample"):
            absolute_sample = tr.absoluteSample
        elif hasattr(tr, "absolute_sample"):
            absolute_sample = tr.absolute_sample
        elif isinstance(tr, dict) and "absoluteSample" in tr:
            absolute_sample = tr["absoluteSample"]
        else:
            raise AttributeError(
                f"track_results[{idx}] 缺少 absoluteSample 字段"
            )

        # 毫秒索引：ms_of_the_signal 按通道存储
        ms_idx = int(ms_of_the_signal[idx])

        # MATLAB 是 1-based，因此这里减 1
        sample_index = absolute_sample[ms_idx - 1]

        # 传播时间 = 样本索引 / 每码周期样本数
        travel_time[idx] = sample_index / samples_per_code

    # ---------- 去除 inf，寻找最小传播时间 ----------
    finite_mask = np.isfinite(travel_time)
    if finite_mask.any():
        minimum = np.floor(travel_time[finite_mask].min())
    else:
        minimum = 0.0  # 所有通道都未跟踪

    # ---------- 相对传播时间 ----------
    travel_time = travel_time - minimum + settings.startOffset

    # ---------- 转换为伪距（米） ----------
    # settings.c 单位是 m/s，因此除以 1000 得到 m/ms
    pseudoranges = travel_time * (settings.c / 1000.0)

    return pseudoranges
