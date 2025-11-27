# placeholder: debugging plots

# src/gnss/utils/plot_utils.py
# src/gnss/utils/plot_acquisition.py

import numpy as np
import matplotlib.pyplot as plt


def _get_first_existing_key(d, candidates):
    """
    在字典 d 里，从 candidates 这些候选键名中找出第一个存在的；
    如果都不存在，返回 None。
    """
    for k in candidates:
        if k in d:
            return k
    return None


def plot_acquisition(acq_results):
    """
    绘制 GNSS 卫星捕获结果，与 MATLAB 的 plotAcquisition 功能类似。

    参数
    ----
    acq_results : dict 或对象
        需要至少包含：
            - 峰值度量：
                'peakMetric' 或 'peak_metric' 或 'peakRatio' 等
            - 载波频率：
                'carrFreq' 或 'carr_freq' 或 'carrier_freq'
        （如果有 PRN 信息：'PRN' / 'prn'，会优先用作横轴）
    """

    # -------- 提取字段（兼容多种写法） --------
    if isinstance(acq_results, dict):
        peak_key = _get_first_existing_key(
            acq_results,
            ["peakMetric", "peak_metric", "peakRatio", "peak_ratio"],
        )
        carr_key = _get_first_existing_key(
            acq_results,
            ["carrFreq", "carr_freq", "carrier_freq"],
        )
        prn_key = _get_first_existing_key(
            acq_results,
            ["PRN", "prn", "sat_prn"],
        )

        if peak_key is None:
            raise KeyError(
                "plot_acquisition: acq_results 中找不到峰值度量字段 "
                "（期待 'peakMetric' 或 'peak_metric' 等）。\n"
                f"  当前 keys = {list(acq_results.keys())}"
            )

        if carr_key is None:
            raise KeyError(
                "plot_acquisition: acq_results 中找不到载波频率字段 "
                "（期待 'carrFreq' 或 'carr_freq' / 'carrier_freq' 等）。\n"
                f"  当前 keys = {list(acq_results.keys())}"
            )

        peak_metric = np.asarray(acq_results[peak_key], dtype=float)
        carr_freq = np.asarray(acq_results[carr_key], dtype=float)

        if prn_key is not None:
            prn = np.asarray(acq_results[prn_key], dtype=int)
        else:
            # 没有 PRN，就用 1,2,3,... 的下标来代替
            prn = np.arange(1, len(peak_metric) + 1, dtype=int)

    else:
        # 兼容“对象形式”的结果：acq_results.peakMetric / acq_results.peak_metric 等
        # 注意：这里会一个个尝试属性名
        def _get_attr_any(obj, candidates, desc):
            for name in candidates:
                if hasattr(obj, name):
                    return getattr(obj, name)
            raise AttributeError(
                f"plot_acquisition: acq_results 对象中找不到 {desc} 属性，"
                f"尝试过: {candidates}"
            )

        peak_metric = np.asarray(
            _get_attr_any(
                acq_results,
                ["peakMetric", "peak_metric", "peakRatio", "peak_ratio"],
                "峰值度量",
            ),
            dtype=float,
        )
        carr_freq = np.asarray(
            _get_attr_any(
                acq_results,
                ["carrFreq", "carr_freq", "carrier_freq"],
                "载波频率",
            ),
            dtype=float,
        )

        # PRN 是可选
        try:
            prn = np.asarray(
                _get_attr_any(acq_results, ["PRN", "prn", "sat_prn"], "PRN 号"),
                dtype=int,
            )
        except AttributeError:
            prn = np.arange(1, len(peak_metric) + 1, dtype=int)

    # -------- 防御性检查 --------
    if peak_metric.size == 0:
        print("[plot_acquisition] peak_metric 为空，没有可绘制的捕获结果。")
        return

    if carr_freq.size != peak_metric.size:
        raise ValueError(
            f"plot_acquisition: peak_metric 与 carr_freq 长度不一致："
            f"{peak_metric.size} vs {carr_freq.size}"
        )

    if prn.size != peak_metric.size:
        # 如果 PRN 数量不匹配，就退回用索引代替，防止乱图
        print(
            "[plot_acquisition] 警告：PRN 数量与 peak_metric 不一致，将使用索引代替 PRN。"
        )
        prn = np.arange(1, len(peak_metric) + 1, dtype=int)

    # -------- 开始绘图 --------
    plt.figure(101)
    plt.clf()
    ax = plt.gca()

    # 蓝色条形图：所有卫星的 peak_metric
    # 横轴用 PRN
    ax.bar(prn, peak_metric, color="C0", label="未捕获的信号")

    # 标题与标签
    ax.set_title("信号捕获结果 (Acquisition results)", fontsize=14)
    ax.set_xlabel("卫星 PRN 号 (无条形图表示该卫星未在搜索列表中)", fontsize=12)
    ax.set_ylabel("捕获度量值 (Acquisition Metric)", fontsize=12)

    # 坐标范围
    ymax = peak_metric.max() if peak_metric.size > 0 else 1.0
    # PRN 通常在 1~32，这里稍微加一点边界
    ax.set_xlim(0.5, max(33, prn.max() + 0.5))
    ax.set_ylim(0, ymax * 1.05)

    # 设置 X 轴刻度为整数 PRN
    ax.set_xticks(prn)

    # 网格
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.6)

    # 已成功捕获的信号：载波频率 > 0
    acquired_mask = carr_freq > 0
    acquired = peak_metric * acquired_mask

    # 绿色覆盖条形图：只在成功捕获的 PRN 上有值
    ax.bar(prn, acquired, color=(0, 0.8, 0), label="已捕获的信号")

    ax.legend()
    plt.tight_layout()
    plt.show()


__all__ = ["plot_acquisition"]
