# src/gnss/post_processing.py

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Tuple, List

import numpy as np

from gnss.acquisition.acquisition_core import acquisition
from gnss.tracking.pre_run import pre_run
from gnss.tracking.show_channel_status import show_channel_status
# ✅ 使用内存版 tracking（一次性读入需要的这段数据）
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


def _parse_channel_selection(user_input: str, max_ch: int) -> List[int]:
    """
    将用户输入的通道选择字符串解析为整数列表。

    允许的形式：
        "" 或 "all" / "a" / "*"   -> 1..max_ch
        "1"                       -> [1]
        "1,3,5"                   -> [1,3,5]
        "1-4"                     -> [1,2,3,4]
        "1-3,6"                   -> [1,2,3,6]
    """
    text = (user_input or "").strip()
    if text == "" or text.lower() in ("all", "a", "*"):
        return list(range(1, max_ch + 1))

    # 兼容中文逗号
    text = text.replace("，", ",")

    channels: List[int] = []

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        # 范围：如 "2-5"
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for ch in range(start, end + 1):
                if 1 <= ch <= max_ch:
                    channels.append(ch)
        else:
            # 单个通道
            try:
                ch = int(part)
            except ValueError:
                continue
            if 1 <= ch <= max_ch:
                channels.append(ch)

    # 去重并排序
    channels = sorted(set(channels))
    return channels


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

    dtype = _dtype_from_string(settings.dataType)
    try:
        with open(filename, "rb") as f:
            raw_signal = np.fromfile(f, dtype=dtype)
    except OSError as e:
        raise RuntimeError(f"无法读取文件 {filename}: {e}") from e

    if raw_signal.size == 0:
        raise RuntimeError("原始数据文件为空，无法处理。")

    # 每个样本字节数 + 需要跳过的样本
    bytes_per_sample = np.dtype(dtype).itemsize
    skip_samples = settings.skipNumberOfBytes // bytes_per_sample

    if skip_samples >= raw_signal.size:
        raise RuntimeError(
            f"skipNumberOfBytes 太大，跳过后的样本数为 0："
            f"skip_samples = {skip_samples}, 文件总样本数 = {raw_signal.size}"
        )

    # ====== 统一计算：每 ms 的采样点数（= 每个 C/A 码周期的采样点数）======
    samples_per_ms = int(
        round(
            settings.samplingFreq
            / (settings.codeFreqBasis / settings.codeLength)
        )
    )

    # 可用总毫秒数（从 skip 之后算起）
    total_ms_available = (raw_signal.size - skip_samples) // samples_per_ms

    print(f"[DEBUG] 文件总样本数 = {raw_signal.size}")
    print(f"[DEBUG] skip_samples = {skip_samples}")
    print(f"[DEBUG] samples_per_ms = {samples_per_ms}")
    print(f"[DEBUG] total_ms_available = {total_ms_available}")
    print(f"[DEBUG] settings.msToProcess(原始) = {settings.msToProcess}")

    if total_ms_available <= 0:
        raise RuntimeError("skip 之后没有足够样本用于处理。")

    # 如果设置的 msToProcess 超过文件可提供的长度，就自动裁剪
    if settings.msToProcess > total_ms_available:
        print(
            f"warning: settings.msToProcess = {settings.msToProcess} ms "
            f"大于文件可用长度 {total_ms_available} ms，自动裁剪为可用长度。"
        )
        settings.msToProcess = int(total_ms_available)

    print(f"[DEBUG] settings.msToProcess(实际使用) = {settings.msToProcess}")

    # ---------- 捕获阶段 ----------
    if settings.skipAcquisition == 0 or acq_results is None:
        # 读取 11 ms 数据用于粗捕获 + 精细频率估计
        n_samples_acq = 11 * samples_per_ms

        start_idx = skip_samples
        end_idx = start_idx + n_samples_acq

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

    # ---------- 跟踪阶段（使用 skip 之后的所有数据） ----------
    start_time = datetime.now()
    print(f"   Tracking started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   跟踪开始于 {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 这里不再手动裁剪到 msToProcess*samples_per_ms，
    # 而是把 skip 之后的所有样本都交给 tracking。
    track_signal = raw_signal[skip_samples:].astype(np.float64, copy=False)

    track_results, channel = tracking_from_array(track_signal, channel, settings)

    elapsed = datetime.now() - start_time
    # 格式化为 HH:MM:SS
    hhmmss = str(elapsed).split(".")[0]
    print(f"   Tracking is over (elapsed time {hhmmss})")
    print(f"   跟踪结束 (耗时 {hhmmss})")

    # ---------- 保存结果（由用户控制，带时间戳） ----------
    save_results: bool = getattr(settings, "saveTrackingResults", False)
    results_dir: str = getattr(settings, "resultsDir", "Results_Data")

    if save_results:
        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"trackingResults_{timestamp}.npz")

        print(f'   Saving Acq & Tracking results to "{out_path}"')
        print(f'   正在将捕获和跟踪结果保存到 "{out_path}" 文件中')

        # 将 acq_results 统一为 dict 形式再保存
        if isinstance(acq_results, dict):
            acq_to_save = acq_results
        else:
            acq_to_save = {
                "carr_freq": acq_results.carrFreq,
                "code_phase": acq_results.codePhase,
                "peak_metric": acq_results.peakMetric,
            }

        _save_tracking_results(
            out_path,
            track_results,
            settings,
            acq_to_save,
            channel,
        )
    else:
        print("   Skip saving trackingResults (settings.saveTrackingResults = False)")
        print("   跳过保存 trackingResults（如需保存，请将 settings.saveTrackingResults 设为 True）")

    # ---------- 导航解算 ----------
    print("   Calculating navigation solutions...")
    print("   正在计算导航解...")

    nav_solutions, eph = post_navigation(track_results, settings)

    print("   Processing is complete for this data block")
    print("   此数据块处理完毕")

    # ---------- 绘图（tracking 是否画图由用户控制） ----------
    print("   Ploting results...")
    print("   正在绘制结果...")

    # 1) 跟踪结果：由 settings.plotTracking + 交互式输入共同控制
    if getattr(settings, "plotTracking", False):
        try:
            ans = input("是否绘制跟踪结果图？(y/n，回车默认为 n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans.startswith("y"):
            prompt = (
                f"请输入要绘制的通道号，例如 1 或 1,3,5 或 1-4 或 all "
                f"(默认 all，当前最多 {settings.numberOfChannels} 路): "
            )
            try:
                ch_text = input(prompt)
            except (EOFError, KeyboardInterrupt):
                ch_text = ""

            channels = _parse_channel_selection(ch_text, settings.numberOfChannels)
            if channels:
                print(f"   正在绘制通道 {channels} 的跟踪结果图...")
                plot_tracking(channels, track_results, settings)
            else:
                print("   未选择有效通道，跳过绘制跟踪结果。")
        else:
            print("   用户选择不绘制跟踪结果图。")
    else:
        print("   settings.plotTracking = False，跳过跟踪结果绘图。")

    # 2) 导航结果
    plot_navigation(nav_solutions, settings)

    print("Post processing of the signal is over.")
    print("信号后处理全部结束。")

    return nav_solutions, eph, track_results, acq_results, channel
