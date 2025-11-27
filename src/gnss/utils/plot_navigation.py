# src/gnss/utils/plot_navigation.py

import numpy as np
import matplotlib.pyplot as plt

# 你自己的天空图函数（已有）
from gnss.navigation.skyplot import sky_plot


# -----------------------
# 辅助函数：deg → [deg, min, sec]
# -----------------------
def deg_to_dms(deg):
    """将十进制度 → (度, 分, 秒)"""
    d = int(deg)
    m_float = abs(deg - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return d, m, s


# ======================
# 主函数：plot_navigation
# ======================
def plot_navigation(nav_solutions, settings):
    """
    绘制导航解算结果，与 MATLAB plotNavigation 完全一致。

    参数
    ----
    nav_solutions : 对象或 dict
        必须包含字段：
            E, N, U （UTM坐标）
            longitude, latitude, height
            DOP
            channel.az / channel.el / channel.PRN
    settings : Settings
        settings.truePosition.E/N/U 用于参考点选择
    """

    if nav_solutions is None:
        print("plotNavigation: No navigation data to plot.")
        return

    # -----------------------
    # 1. 选择参考坐标
    # -----------------------
    E = np.asarray(nav_solutions.E)
    N = np.asarray(nav_solutions.N)
    U = np.asarray(nav_solutions.U)

    # 若无真实参考点，则用平均位置
    if (
        np.isnan(settings.truePosition.E)
        or np.isnan(settings.truePosition.N)
        or np.isnan(settings.truePosition.U)
    ):
        ref_E = np.nanmean(E)
        ref_N = np.nanmean(N)
        ref_U = np.nanmean(U)

        # 经纬度平均值
        lon_mean = np.nanmean(nav_solutions.longitude)
        lat_mean = np.nanmean(nav_solutions.latitude)
        hgt_mean = np.nanmean(nav_solutions.height)

        lon_dms = deg_to_dms(lon_mean)
        lat_dms = deg_to_dms(lat_mean)

        ref_point_text = (
            "Mean Position\n"
            f"Lat: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]:.2f}''\n"
            f"Lng: {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]:.2f}''\n"
            f"Hgt: {hgt_mean:+6.1f}"
        )
    else:
        ref_E = settings.truePosition.E
        ref_N = settings.truePosition.N
        ref_U = settings.truePosition.U
        ref_point_text = "Reference Position"

    # -----------------------
    # 2. 设置图形窗口
    # -----------------------
    fig_num = 300
    plt.figure(fig_num, figsize=(10, 12))
    plt.clf()
    plt.suptitle("Navigation solutions", fontsize=14)

    # -----------------------
    # 子图布局：4×2
    # -----------------------
    ax1 = plt.subplot(4, 2, (1, 2, 3, 4))   # 顶部大图
    ax2 = plt.subplot(4, 2, (5, 7))         # 左下角
    ax3 = plt.subplot(4, 2, (6, 8))         # 右下角（天空图）

    # -----------------------
    # 3. 绘制 ENU 差值随时间变化
    # -----------------------
    diff_E = E - ref_E
    diff_N = N - ref_N
    diff_U = U - ref_U

    t = np.arange(len(E)) * settings.navSolPeriod  # 时间轴（毫秒）

    ax1.plot(t, diff_E, label="E")
    ax1.plot(t, diff_N, label="N")
    ax1.plot(t, diff_U, label="U")

    ax1.set_title("Coordinates variations in UTM system")
    ax1.set_xlabel(f"Measurement period: {settings.navSolPeriod} ms")
    ax1.set_ylabel("Variations (m)")
    ax1.grid(True)
    ax1.legend()
    ax1.autoscale()

    # -----------------------
    # 4. 3D 位置图（俯视）
    # -----------------------
    ax2.plot(diff_E, diff_N, "+", label="Measurements")
    ax2.plot(0, 0, "r+", markersize=10, label=ref_point_text)

    ax2.set_title("Positions in UTM system (3D plot)")
    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("North (m)")
    ax2.grid(True, which="both")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend()

    # MATLAB uses view(0, 90) → top-down view  
    # 2D 投影即可，无需 3D axes。

    # -----------------------
    # 5. 天空图（SkyPlot）
    # -----------------------
    sky_plot(
        ax3,
        nav_solutions.channel.az,
        nav_solutions.channel.el,
        nav_solutions.channel.PRN[:, 0],  # MATLAB PRN(:,1)
    )
    mean_pdop = np.nanmean(nav_solutions.DOP[1, :])  # MATLAB DOP(2,:)
    ax3.set_title(f"Sky plot (mean PDOP: {mean_pdop:.2f})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


__all__ = ["plot_navigation"]
