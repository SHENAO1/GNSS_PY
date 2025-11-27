# placeholder: debugging plots

# src/gnss/utils/plot_utils.py

import numpy as np
import matplotlib.pyplot as plt


def plot_acquisition(acq_results):
    """
    绘制 GNSS 卫星捕获结果，与 MATLAB 的 plotAcquisition 完全一致。

    参数
    ----
    acq_results : dict 或对象
        必须包含：
            - peakMetric : 长度 32 的数组（或更长，但只绘 1~32）
            - carrFreq   : 同样长度的数组
    """

    # 提取字段（兼容 dict 和对象）
    if isinstance(acq_results, dict):
        peak_metric = np.asarray(acq_results["peakMetric"])
        carr_freq = np.asarray(acq_results["carrFreq"])
    else:
        peak_metric = np.asarray(acq_results.peakMetric)
        carr_freq = np.asarray(acq_results.carrFreq)

    # 创建图形窗口，相当于 MATLAB 的 figure(101)
    plt.figure(101)
    plt.clf()  # 清空上一轮的内容（如果有）
    ax = plt.gca()

    # 绘制蓝色条形图（所有 PRN）
    ax.bar(np.arange(len(peak_metric)), peak_metric, color="C0", label="未捕获的信号")

    # 标题与标签
    ax.set_title("信号捕获结果 (Acquisition results)", fontsize=14)
    ax.set_xlabel("卫星 PRN 号 (无条形图表示该卫星未在搜索列表中)", fontsize=12)
    ax.set_ylabel("捕获度量值 (Acquisition Metric)", fontsize=12)

    # 坐标范围：X: 0~33
    ymax = peak_metric.max() if len(peak_metric) > 0 else 1
    ax.set_xlim(0, 33)
    ax.set_ylim(0, ymax * 1.05)

    # 设置次刻度
    ax.minorticks_on()

    # Y 方向网格
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.6)

    # 已成功捕获的信号（carrFreq > 0）
    acquired = peak_metric * (carr_freq > 0)

    # 覆盖绿色条形图
    ax.bar(
        np.arange(len(acquired)),
        acquired,
        color=(0, 0.8, 0),
        label="已捕获的信号",
    )

    ax.legend()

    plt.tight_layout()
    plt.show()

__all__ = ["plot_acquisition"]
