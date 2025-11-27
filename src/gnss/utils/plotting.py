"""绘图工具，对应 Matlab skyPlot.m、showChannelStatus.m (占位)。"""
"""
全局绘图工具：
- 设置 matplotlib 中文字体（解决中文显示方框问题）
- 设置全局字号、线宽、图例字号
- 可供整个 GNSS 工程调用
"""

import matplotlib.pyplot as plt

def set_global_plot_style(
    font="Microsoft YaHei",  # Windows 推荐字体（100%支持中文）
    font_size=14,
    title_size=18,
    label_size=14,
    tick_size=12,
):
    """
    设置全局绘图样式，解决中文显示问题。
    你可以在主程序入口（main.py 或 plot 文件顶部）调用一次。
    """

    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负号乱码

    # 全局字号
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = title_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size

    # 线宽、图例等
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["legend.fontsize"] = font_size

    print(f"[Plotting] 全局绘图样式已应用：字体 = {font}")


# 可选：工程启动时自动调用
set_global_plot_style()
