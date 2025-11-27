# src/gnss/utils/probe_data.py

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _dtype_from_string(s: str):
    """将 MATLAB 风格的数据类型字符串映射到 numpy.dtype。"""
    s = s.lower()
    if s == "int8":
        return np.int8
    if s in ("int16", "short"):
        return np.int16
    if s in ("int32", "int"):
        return np.int32
    if s == "uint8":
        return np.uint8
    if s == "uint16":
        return np.uint16
    # 如需更多类型可以在这里扩展
    raise ValueError(f"不支持的数据类型: {s}")


def _welch_psd(x: np.ndarray,
               fs: float,
               nperseg: int = 16384,
               noverlap: int = 1024,
               nfft: int = 2048):
    """
    简单实现一个 Welch 功率谱估计（只用于画图，不追求绝对标定）。
    返回:
        freqs_Hz, psd (线性刻度)
    """
    x = np.asarray(x, dtype=float)
    if x.size < nperseg:
        nperseg = x.size
    if nperseg <= 0:
        return np.array([]), np.array([])

    step = nperseg - noverlap
    if step <= 0:
        step = nperseg

    segments = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg]
        segments.append(seg)
    if not segments:
        segments = [x[:nperseg]]

    window = np.hanning(nperseg)
    psd_acc = None
    for seg in segments:
        seg_win = seg * window
        X = np.fft.rfft(seg_win, n=nfft)
        power = (np.abs(X) ** 2)
        if psd_acc is None:
            psd_acc = power
        else:
            psd_acc += power

    psd = psd_acc / len(segments)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, psd


def probe_data(settings, file_name: Optional[str] = None):
    """
    Python 版 probeData.m

    用法:
        probe_data(settings)
        probe_data(settings, file_name="xxx.bin")

    参数
    ----
    settings : Settings
        需要字段:
            - fileName
            - dataType
            - skipNumberOfBytes
            - samplingFreq
            - codeFreqBasis
            - codeLength
    file_name : str or None
        如果为 None，则使用 settings.fileName。
    """

    # 处理 file_name 参数
    if file_name is None:
        file_name_str = settings.fileName
    else:
        file_name_str = file_name

    if not isinstance(file_name_str, str):
        raise TypeError("文件名必须是字符串 (File name must be a string)")

    if not os.path.isfile(file_name_str):
        raise FileNotFoundError(f"无法找到数据文件: {file_name_str}")

    # 以二进制只读方式打开文件
    try:
        f = open(file_name_str, "rb")
    except OSError as e:
        raise RuntimeError(f"无法读取文件 {file_name_str}: {e}") from e

    # 跳过指定字节
    f.seek(settings.skipNumberOfBytes, os.SEEK_SET)

    # 每个 C/A 码周期（1 ms）的采样点数
    samples_per_code = int(
        round(
            settings.samplingFreq
            / (settings.codeFreqBasis / settings.codeLength)
        )
    )

    # 读取 10 ms 数据
    n_samples = 10 * samples_per_code
    dtype = _dtype_from_string(settings.dataType)

    data = np.fromfile(f, dtype=dtype, count=n_samples)
    f.close()

    if data.size < n_samples:
        raise RuntimeError(
            "无法从数据文件中读取足够的数据 "
            "(Could not read enough data from the data file.)"
        )

    data = data.astype(float)

    # ---------- 开始画图 ----------
    plt.figure(100, figsize=(10, 8))
    plt.clf()

    # 时间轴：0 ~ 5 ms（比原数据短，用来画一个小片段）
    time_scale = np.arange(0, int(5e-3 * settings.samplingFreq) + 1) / settings.samplingFreq

    # 1) 时域图（只画一个码周期的 1/50）
    plt.subplot(2, 2, 1)

    n_plot = samples_per_code // 50
    n_plot = min(n_plot, data.size, time_scale.size)
    t_ms = time_scale[:n_plot] * 1000.0

    plt.plot(t_ms, data[:n_plot])
    plt.grid(True)
    plt.title("时域图 (Time domain plot)")
    plt.xlabel("时间 (ms)")
    plt.ylabel("幅度 (Amplitude)")
    plt.tight_layout()

    # 2) 频域图（Welch 功率谱估计）
    plt.subplot(2, 2, 2)

    x = data - np.mean(data)
    freqs, psd = _welch_psd(
        x,
        fs=settings.samplingFreq,
        nperseg=16384,
        noverlap=1024,
        nfft=2048,
    )

    if freqs.size > 0:
        freqs_MHz = freqs / 1e6
        # 为了和 MATLAB pwelch 视觉接近，这里用对数刻度
        plt.semilogy(freqs_MHz, psd + 1e-12)  # 防止 log(0)
        plt.grid(True)
        plt.title("频域图 (Frequency domain plot)")
        plt.xlabel("频率 (MHz)")
        plt.ylabel("功率谱 (arb. units)")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "数据长度不足以计算 Welch PSD",
                 ha="center", va="center", transform=plt.gca().transAxes)

    # 3) 直方图
    plt.subplot(2, 2, (3, 4))
    plt.hist(data, bins=range(-128, 129))
    dmax = np.max(np.abs(data)) + 1
    plt.xlim(-dmax, dmax)
    plt.grid(True)
    plt.title("直方图 (Histogram)")
    plt.xlabel("采样值 (Bin)")
    plt.ylabel("数量 (Number in bin)")

    plt.tight_layout()
    plt.show()