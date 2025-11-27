# placeholder for sky plot

"""
sky_plot(AZ, EL, PRN, line_style='auto', ax=None)

功能：
    从接收机视角绘制“天空图”（Sky Plot）。
    显示在一段时间内，可见卫星在天空中的方位/仰角轨迹。

输入参数：
    AZ          - 卫星方位角矩阵 (单位: 度)，形状 (Nsat, Ntime)
                  每一行代表一颗卫星，每一列代表一个历元。
    EL          - 卫星仰角矩阵 (单位: 度)，形状必须与 AZ 相同。
    PRN         - 卫星 PRN 号向量，长度为 Nsat。
    line_style  - (可选) 线型字符串，所有卫星轨迹使用相同样式。
                  例如 'r-.'，如果为 'auto' 则使用 matplotlib 默认配色。
    ax          - (可选) matplotlib 的 Axes 对象。如果为 None，则新建一个图和坐标轴。

返回：
    hpol        - 由 ax.plot 返回的 Line2D 对象列表（每颗卫星一条折线轨迹）。
"""

from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


def sky_plot(
    az: np.ndarray,
    el: np.ndarray,
    prn: Sequence[int],
    line_style: str = "auto",
    ax: Optional[plt.Axes] = None,
):
    # ===================== 1. 检查与解析输入参数 ============================
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    prn = np.asarray(prn)

    if az.shape != el.shape:
        raise ValueError("AZ 和 EL 矩阵的维度必须相同。")

    if az.shape[0] != prn.shape[0]:
        raise ValueError("PRN 数组长度必须与 AZ/EL 的行数一致。")

    # 如果没有传入坐标轴，则新建一个
    if ax is None:
        fig, ax = plt.subplots()

    # 取一个默认颜色，用于网格线（这里简单用灰色代替 MATLAB 的 xcolor）
    tc = "0.5"

    # 保持当前坐标轴，用于叠加绘图
    ax.set_aspect("equal", adjustable="box")

    # ===================== 2. 绘制白色圆形背景 ============================
    # 以 (0,0) 为中心，半径 90 的圆
    circle = plt.Circle(
        (0, 0),
        90,
        facecolor="white",
        edgecolor=tc,
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(circle)

    # ===================== 3. 绘制方位角“辐条”网格 =======================
    # 需要 6 条线，将圆分成 12 个 30° 扇区
    th = (np.arange(1, 7) * 2 * np.pi / 12.0)  # 弧度制
    cst = np.cos(th)
    snt = np.sin(th)

    # 一条穿过圆心的线有两个端点：(+,-) 成对出现
    cs = np.vstack((cst, -cst))
    sn = np.vstack((snt, -snt))

    # 绘制 6 条辐条线
    ax.plot(
        90 * sn,
        90 * cs,
        linestyle=":",
        color=tc,
        linewidth=0.5,
        zorder=1,
    )

    # ===================== 4. 标注方位角刻度（0~330，每 30°） =============
    rt = 1.1 * 90  # 标签稍微画在主圆外面

    for i in range(len(th)):
        # 上半部分（或右半部分）标签：30, 60, ... , 180
        ax.text(
            rt * snt[i],
            rt * cst[i],
            f"{(i+1)*30}",
            ha="center",
            va="center",
        )

        # 对应的另一半：210, 240, ...，以及 0/360
        if i == len(th) - 1:
            loc = "0"
        else:
            loc = str(180 + (i + 1) * 30)

        ax.text(
            -rt * snt[i],
            -rt * cst[i],
            loc,
            ha="center",
            va="center",
        )

    # ===================== 5. 绘制仰角网格线（同心圆） ====================
    # 使用单位圆模板
    th_c = np.linspace(0, 2 * np.pi, 200)
    xunit = np.cos(th_c)
    yunit = np.sin(th_c)

    for elevation in range(0, 91, 15):
        # 仰角转为绘图半径
        # 仰角 90°(天顶) → cos(90)=0 → 半径=0（中心）
        # 仰角 0°(地平线) → cos(0)=1 → 半径=90（最外圈）
        elevation_spherical = 90 * np.cos(np.deg2rad(elevation))

        ax.plot(
            yunit * elevation_spherical,
            xunit * elevation_spherical,
            linestyle=":",
            color=tc,
            linewidth=0.5,
            zorder=1,
        )

        # 在该圆顶部添加仰角刻度
        ax.text(
            0,
            elevation_spherical,
            f"{elevation}",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )

    # 设置坐标轴范围，留出标题和标签位置
    ax.set_xlim(-95, 95)
    ax.set_ylim(-90, 101)

    # ===================== 6. 转换卫星数据坐标 ============================
    # 仰角转绘图半径
    el_spherical = 90 * np.cos(np.deg2rad(el))

    # 极坐标 (az, r) → 笛卡尔坐标 (x, y)
    # 注意：方位角 0°(北) 在 Y 轴正方向，90°(东) 在 X 轴正方向：
    #   x = r * sin(az), y = r * cos(az)
    yy = el_spherical * np.cos(np.deg2rad(az))
    xx = el_spherical * np.sin(np.deg2rad(az))

    # ===================== 7. 绘制卫星轨迹 ===============================
    if line_style == "auto":
        # 使用默认颜色/线型
        # 注意：plot 默认按列为一条曲线，这里需要转置：
        hpol = ax.plot(xx.T, yy.T, ".-")
    else:
        hpol = ax.plot(xx.T, yy.T, line_style)

    # 在每条轨迹的最后一个点画圆圈，表示当前卫星位置
    ax.plot(
        xx[:, -1],
        yy[:, -1],
        "o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="k",
    )

    # 在当前卫星位置旁边标出 PRN 号
    for i in range(len(prn)):
        if prn[i] != 0:
            ax.text(
                xx[i, -1],
                yy[i, -1],
                f"  {int(prn[i])}",
                color="b",
                ha="left",
                va="center",
            )

    # ===================== 8. 最后格式化调整 ==============================
    # 确保圆形不被压扁
    ax.set_aspect("equal", adjustable="box")

    # 关闭坐标轴刻度和框线
    ax.axis("off")

    return hpol
