# src/gnss/post_processing.py

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

from gnss.acquisition.acquisition_core import acquisition
from gnss.tracking.pre_run import pre_run
from gnss.tracking.show_channel_status import show_channel_status
# ✅ 使用内存版 tracking（一次性读入整个数据）
from gnss.tracking.tracking_core import tracking_from_array

from gnss.navigation.navigation import post_navigation

from gnss.utils.plot_acquisition import plot_acquisition
from gnss.utils.plot_tracking import plot_tracking
from gnss.utils.plot_navigation import plot_navigation


def _dtype_from_string(s: str):
    """
    将 MATLAB 的 settings.dataType 映射到 numpy 的 dtype。
    目前只处理最常见的几种。
    """
    s = s.lower()
    if s == "int8":
        return np.int8
    if s in ("int16", "short"):
        return np.int16
    if s in ("int32", "int"):
        return np.int32
    if s in ("uint8",):
        return np.uint8
    # 可以按需扩展
    raise ValueError(f"不支持的数据类型: {s}")


def _save_tracking_results(
    filename: str,
    track_results,
    settings,
    acq_results,
    channel,
):
    """
    简单版本：把关键结果保存为 .npz 文件。
    如果你以后想兼容 MATLAB 的 trackingResults.mat，
    可以改成用 scipy.io.savemat。
    """
    np.savez(
        filename,
        trackResults=np.array(track_results, dtype=object),
        settings=SimpleNamespace(**settings.__dict__),
        acqResults=SimpleNamespace(**acq_results),
        channel=np.array(channel, dtype=object),
    )


def post_processing(settings, acq_results: Optional[dict] = None):
    """
    Python 版 postProcessing.m

    参数
    ----
    settings : Settings
        来自 gnss.settings.init_settings() 的配置对象。
    acq_results : dict 或 None
        如果你想跳过捕获（settings.skipAcquisition = 1），
        可以把之前保存好的 acq_results 传进来。
        正常使用时可以不传，函数会自动执行捕获。
    """

    print("Starting processing...")
    print("开始处理...")

    # ---------- 打开数据文件 ----------
    filename = settings.fileName
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"无法找到数据文件: {filename}")

    # 统一在这里一次性把整段数据读入内存（仿真场景推荐）
    dtype = _dtype_from_string(settings.dataType)
    try:
        with open(filename, "rb") as f:
            raw_signal = np.fromfile(f, dtype=dtype)
    except OSError as e:
        raise RuntimeError(f"无法读取文件 {filename}: {e}") from e

    if raw_signal.size == 0:
        raise RuntimeError("原始数据文件为空，无法处理。")

    # ---------- 捕获阶段 ----------
    if settings.skipAcquisition == 0 or acq_results is None:
        # 计算每个 C/A 码周期的采样点数 (1 ms)
        samples_per_code = int(
            round(
                settings.samplingFreq
                / (settings.codeFreqBasis / settings.codeLength)
            )
        )

        # 读取 11 ms 数据用于粗捕获 + 精细频率估计
        n_samples = 11 * samples_per_code

        # 根据 skipNumberOfBytes 计算应跳过的样本数
        bytes_per_sample = np.dtype(dtype).itemsize
        skip_samples = settings.skipNumberOfBytes // bytes_per_sample

        start_idx = skip_samples
        end_idx = start_idx + n_samples

        if end_idx > raw_signal.size:
            raise RuntimeError(
                f"原始数据长度不足以进行 11 ms 捕获："
                f"需要 {end_idx} 个样本，实际只有 {raw_signal.size} 个样本。"
            )

        # 转成 float 方便后续处理
        data = raw_signal[start_idx:end_idx].astype(np.float64, copy=False)

        print("   Acquiring satellites...")
        print("   正在捕获卫星...")

        acq_results = acquisition(data, settings)

        # 绘制捕获结果
        plot_acquisition(acq_results)

    # ---------- 检查是否捕获到任何卫星 ----------
    # carrFreq 全为 0 -> 没有成功捕获
    carr_freq = (
        acq_results["carr_freq"]
        if isinstance(acq_results, dict)
        else acq_results.carrFreq
    )
    if not np.any(carr_freq):
        print("No GNSS signals detected, signal processing finished.")
        print("未检测到 GNSS 信号，信号处理结束。")
        return

    # ---------- 初始化通道 ----------
    channel = pre_run(acq_results, settings)
    show_channel_status(channel, settings)

    # ---------- 跟踪阶段（使用内存版 tracking_from_array） ----------
    start_time = datetime.now()
    print(f"   Tracking started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   跟踪开始于 {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 这里不再传文件句柄，而是整段 raw_signal
    track_results, channel = tracking_from_array(raw_signal, channel, settings)

    elapsed = datetime.now() - start_time
    # 格式化为 HH:MM:SS
    hhmmss = str(elapsed).split(".")[0]
    print(f"   Tracking is over (elapsed time {hhmmss})")
    print(f"   跟踪结束 (耗时 {hhmmss})")

    # ---------- 保存结果 ----------
    print('   Saving Acq & Tracking results to "trackingResults.npz"')
    print('   正在将捕获和跟踪结果保存到 "trackingResults.npz" 文件中')

    _save_tracking_results(
        "trackingResults.npz",
        track_results,
        settings,
        acq_results
        if isinstance(acq_results, dict)
        else {
            "carr_freq": acq_results.carrFreq,
            "code_phase": acq_results.codePhase,
            "peak_metric": acq_results.peakMetric,
        },
        channel,
    )

    # ---------- 导航解算 ----------
    print("   Calculating navigation solutions...")
    print("   正在计算导航解...")

    nav_solutions, eph = post_navigation(track_results, settings)

    print("   Processing is complete for this data block")
    print("   此数据块处理完毕")

    # ---------- 绘图 ----------
    print("   Ploting results...")
    print("   正在绘制结果...")

    if settings.plotTracking:
        plot_tracking(range(1, settings.numberOfChannels + 1), track_results, settings)

    plot_navigation(nav_solutions, settings)

    print("Post processing of the signal is over.")
    print("信号后处理全部结束。")

    return nav_solutions, eph, track_results, acq_results, channel
