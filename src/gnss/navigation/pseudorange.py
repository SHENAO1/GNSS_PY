"""伪距计算，对应 Matlab calculatePseudoranges.m (占位)。"""

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

    参数
    ----
    track_results : 序列（list / tuple）
        跟踪结果，每个元素对应一个通道。
        每个元素需包含属性/字段：
            - absoluteSample: 一维数组，给出某毫秒对应的“绝对采样点索引”。
    ms_of_the_signal : 序列
        长度为 numberOfChannels 的整数数组，第 ch 通道在第多少毫秒处取伪距。
        （即 MATLAB 的 msOfTheSignal(channelNr)）
    channel_list : 序列
        要参与伪距计算的通道号列表（MATLAB 中是 1~N，这里会自动减一）
    settings : 配置对象
        需要以下字段：
            - samplingFreq
            - codeFreqBasis
            - codeLength
            - numberOfChannels
            - startOffset
            - c      （光速，单位 m/s）

    返回
    ----
    pseudoranges : np.ndarray, shape (numberOfChannels,)
        各通道的相对伪距，单位：米。
        未跟踪的通道对应的位置会很大（由于 travelTime 初始化为 inf，会被忽略）
    """

    # --- 设置初始传播时间为无穷大 ------------------------------------
    n_ch = settings.numberOfChannels
    travel_time = np.full(n_ch, np.inf, dtype=float)

    # 每个 1 ms C/A 码的采样点数
    samples_per_code = int(
        round(
            settings.samplingFreq
            / (settings.codeFreqBasis / settings.codeLength)
        )
    )

    # 把 channel_list 拉成一维行向量（类似 MATLAB 的 channelList(:).')
    channel_list = np.array(channel_list, copy=False).ravel()

    # --- 对列表中的所有通道，计算传播时间 -----------------------------
    for channel_nr in channel_list:
        # MATLAB 通道号是 1-based，这里转成 Python 0-based 索引
        idx = int(channel_nr) - 1
        if idx < 0 or idx >= n_ch:
            continue  # 防御性检查

        tr = track_results[idx]

        # 取出 absoluteSample 向量（兼容 attribute / dict 两种写法）
        if hasattr(tr, "absoluteSample"):
            absolute_sample = getattr(tr, "absoluteSample")
        elif hasattr(tr, "absolute_sample"):
            absolute_sample = getattr(tr, "absolute_sample")
        elif isinstance(tr, dict) and "absoluteSample" in tr:
            absolute_sample = tr["absoluteSample"]
        else:
            raise AttributeError(
                f"track_results[{idx}] 中找不到 absoluteSample / absolute_sample 字段"
            )

        # ms_of_the_signal 与 MATLAB 一样，按通道号索引
        ms_idx = ms_of_the_signal[idx]

        # MATLAB: trackResults(channelNr).absoluteSample(msOfTheSignal(channelNr))
        # 注意 MATLAB 是 1-based，所以这里减 1
        sample_index = absolute_sample[ms_idx - 1]

        # 传播时间 = 绝对采样点索引 / 每码周期采样点数
        travel_time[idx] = sample_index / samples_per_code

    # --- 截断传播时间并计算伪距 --------------------------------------
    # 只对有限值取最小值，忽略 inf（未跟踪通道）
    finite_mask = np.isfinite(travel_time)
    if finite_mask.any():
        minimum = np.floor(travel_time[finite_mask].min())
    else:
        minimum = 0.0

    # 减去最小值并加上起始偏移量，得到“相对传播时间”
    travel_time = travel_time - minimum + settings.startOffset

    # --- 将传播时间转换为伪距 ---------------------------------------
    # settings.c 单位：m/s，这里 /1000 -> m/ms
    pseudoranges = travel_time * (settings.c / 1000.0)

    return pseudoranges
