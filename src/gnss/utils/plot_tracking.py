# src/gnss/utils/plot_tracking.py

import numpy as np
import matplotlib.pyplot as plt


def _get_field(tr, name):
    """兼容 struct 对象 / dict 的字段访问。"""
    if hasattr(tr, name):
        return getattr(tr, name)
    if isinstance(tr, dict) and name in tr:
        return tr[name]
    raise AttributeError(f"trackResults 中缺少字段 {name!r}")


def plot_tracking(channel_list, track_results, settings):
    """
    绘制指定通道的跟踪结果，对应 MATLAB 的 plotTracking.m

    参数
    ----
    channel_list : 序列
        要绘制的通道号列表（1-based）。
    track_results : 序列
        跟踪结果，每个元素是一个通道。
        需要字段：
            PRN, I_P, Q_P, pllDiscr, pllDiscrFilt,
            dllDiscr, dllDiscrFilt, I_E, Q_E, I_L, Q_L
    settings : Settings
        需要字段：
            numberOfChannels, msToProcess
    """

    # 安全过滤通道号：只保留 1..numberOfChannels
    valid_channels = [
        ch for ch in channel_list
        if 1 <= int(ch) <= settings.numberOfChannels
    ]

    for ch in valid_channels:
        idx = int(ch) - 1  # Python 0-based
        tr = track_results[idx]

        # --- 取出各轨迹数据 ----------------------------------------
        I_P = np.asarray(_get_field(tr, "I_P"), dtype=float)
        Q_P = np.asarray(_get_field(tr, "Q_P"), dtype=float)
        pll_discr = np.asarray(_get_field(tr, "pllDiscr"), dtype=float)
        pll_discr_filt = np.asarray(_get_field(tr, "pllDiscrFilt"), dtype=float)
        dll_discr = np.asarray(_get_field(tr, "dllDiscr"), dtype=float)
        dll_discr_filt = np.asarray(_get_field(tr, "dllDiscrFilt"), dtype=float)

        I_E = np.asarray(_get_field(tr, "I_E"), dtype=float)
        Q_E = np.asarray(_get_field(tr, "Q_E"), dtype=float)
        I_L = np.asarray(_get_field(tr, "I_L"), dtype=float)
        Q_L = np.asarray(_get_field(tr, "Q_L"), dtype=float)

        # 为了避免长度不一致，按最短长度截断
        n_samples = min(
            len(I_P),
            len(Q_P),
            len(pll_discr),
            len(pll_discr_filt),
            len(dll_discr),
            len(dll_discr_filt),
            len(I_E),
            len(Q_E),
            len(I_L),
            len(Q_L),
            settings.msToProcess,
        )

        if n_samples <= 0:
            print(f"plot_tracking: 通道 {ch} 没有有效数据，跳过。")
            continue

        I_P = I_P[:n_samples]
        Q_P = Q_P[:n_samples]
        pll_discr = pll_discr[:n_samples]
        pll_discr_filt = pll_discr_filt[:n_samples]
        dll_discr = dll_discr[:n_samples]
        dll_discr_filt = dll_discr_filt[:n_samples]
        I_E = I_E[:n_samples]
        Q_E = Q_E[:n_samples]
        I_L = I_L[:n_samples]
        Q_L = Q_L[:n_samples]

        # 时间轴（秒），MATLAB: (1:msToProcess)/1000
        time_axis = np.arange(1, n_samples + 1) / 1000.0

        PRN = _get_field(tr, "PRN")

        # --- 图窗设置 ------------------------------------------------
        fig_num = ch + 200
        plt.figure(fig_num, figsize=(10, 8))
        plt.clf()
        plt.suptitle(f"通道 {ch} (PRN {PRN}) 跟踪结果", fontsize=14)

        # 子图网格 3x3
        ax11 = plt.subplot(3, 3, 1)         # I/Q 星座图
        ax12 = plt.subplot(3, 3, (2, 3))    # 导航电文
        ax21 = plt.subplot(3, 3, 4)         # 原始 PLL 鉴别器
        ax22 = plt.subplot(3, 3, (5, 6))    # 相关峰幅度
        ax31 = plt.subplot(3, 3, 7)         # 滤波后 PLL
        ax32 = plt.subplot(3, 3, 8)         # 原始 DLL
        ax33 = plt.subplot(3, 3, 9)         # 滤波后 DLL

        # ----- 1. I/Q 星座图 ----------------------------------------
        ax11.plot(I_P, Q_P, ".", markersize=2)
        ax11.grid(True)
        ax11.set_aspect("equal", adjustable="box")
        ax11.set_title("I/Q 星座图 (即时支路)")
        ax11.set_xlabel("I_P (同相分量)")
        ax11.set_ylabel("Q_P (正交分量)")

        # ----- 2. 导航电文比特流 (I_P) ------------------------------
        ax12.plot(time_axis, I_P)
        ax12.grid(True)
        ax12.set_title("导航电文比特流 (I_P 分量)")
        ax12.set_xlabel("时间 (s)")
        ax12.set_xlim(time_axis[0], time_axis[-1])

        # ----- 3. 原始 PLL 鉴别器输出 -------------------------------
        ax21.plot(time_axis, pll_discr, "r")
        ax21.grid(True)
        ax21.set_xlim(time_axis[0], time_axis[-1])
        ax21.set_xlabel("时间 (s)")
        ax21.set_ylabel("幅度")
        ax21.set_title("原始 PLL 鉴别器输出")

        # ----- 4. 相关峰幅度 (E/P/L) --------------------------------
        amp_E = np.sqrt(I_E**2 + Q_E**2)
        amp_P = np.sqrt(I_P**2 + Q_P**2)
        amp_L = np.sqrt(I_L**2 + Q_L**2)

        ax22.plot(time_axis, amp_E, "-*", label=r"$\sqrt{I_E^2 + Q_E^2}$ (Early)")
        ax22.plot(time_axis, amp_P, "-*", label=r"$\sqrt{I_P^2 + Q_P^2}$ (Prompt)")
        ax22.plot(time_axis, amp_L, "-*", label=r"$\sqrt{I_L^2 + Q_L^2}$ (Late)")

        ax22.grid(True)
        ax22.set_title("相关峰幅度")
        ax22.set_xlabel("时间 (s)")
        ax22.set_xlim(time_axis[0], time_axis[-1])
        ax22.legend()

        # ----- 5. 滤波后的 PLL 鉴别器输出 ---------------------------
        ax31.plot(time_axis, pll_discr_filt, "b")
        ax31.grid(True)
        ax31.set_xlim(time_axis[0], time_axis[-1])
        ax31.set_xlabel("时间 (s)")
        ax31.set_ylabel("幅度")
        ax31.set_title("滤波后的 PLL 鉴别器输出")

        # ----- 6. 原始 DLL 鉴别器输出 -------------------------------
        ax32.plot(time_axis, dll_discr, "r")
        ax32.grid(True)
        ax32.set_xlim(time_axis[0], time_axis[-1])
        ax32.set_xlabel("时间 (s)")
        ax32.set_ylabel("幅度")
        ax32.set_title("原始 DLL 鉴别器输出")

        # ----- 7. 滤波后的 DLL 鉴别器输出 ---------------------------
        ax33.plot(time_axis, dll_discr_filt, "b")
        ax33.grid(True)
        ax33.set_xlim(time_axis[0], time_axis[-1])
        ax33.set_xlabel("时间 (s)")
        ax33.set_ylabel("幅度")
        ax33.set_title("滤波后的 DLL 鉴别器输出")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


__all__ = ["plot_tracking"]
