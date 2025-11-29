# src/gnss/tracking/tracking_core.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np

# 尝试导入 CuPy，用于 GPU 加速；没有也没关系，会自动回退到 NumPy
try:
    import cupy as cp
except ImportError:
    cp = None

# Windows 控制台按键检测（用于手动停止）
try:
    import msvcrt  # 仅 Windows 有
except ImportError:
    msvcrt = None

from gnss.tracking.loop.calc_loop_coef import calc_loop_coef
from gnss.acquisition.ca_code import generate_ca_code


def _dtype_from_string(s: str):
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
    raise ValueError(f"不支持的数据类型: {s}")


def _as_attr(obj, name):
    """兼容 dict / SimpleNamespace / 自定义对象."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _init_track_results(code_periods: int, n_ch: int) -> list[SimpleNamespace]:
    """
    创建与 SoftGNSS 结构兼容的 trackResults 列表。
    【重要修复】必须在循环内部为每个通道单独创建 numpy 数组，
    防止所有通道共享同一个数组的内存引用。
    """
    results = []
    for _ in range(n_ch):
        # 必须在这里创建新的数组对象！
        res = SimpleNamespace(
            status="-",
            PRN=0,
            # 记录绝对采样位置（用于调试）
            absoluteSample=np.zeros(code_periods, dtype=np.int64),
            
            # 频率记录
            codeFreq=np.full(code_periods, np.inf, dtype=float),
            carrFreq=np.full(code_periods, np.inf, dtype=float),
            
            # 相关器输出
            I_P=np.zeros(code_periods, dtype=float),
            I_E=np.zeros(code_periods, dtype=float),
            I_L=np.zeros(code_periods, dtype=float),
            Q_E=np.zeros(code_periods, dtype=float),
            Q_P=np.zeros(code_periods, dtype=float),
            Q_L=np.zeros(code_periods, dtype=float),
            
            # 鉴别器输出
            dllDiscr=np.full(code_periods, np.inf, dtype=float),
            dllDiscrFilt=np.full(code_periods, np.inf, dtype=float),
            pllDiscr=np.full(code_periods, np.inf, dtype=float),
            pllDiscrFilt=np.full(code_periods, np.inf, dtype=float),
        )
        results.append(res)
    return results


def _get_xp(settings):
    """
    根据 settings 决定使用哪种数组库：
    - 如果 settings.use_gpu_tracking=True 且 CuPy + CUDA 都可用，则用 cupy
    - 否则回退到 numpy
    """
    use_gpu = getattr(settings, "use_gpu_tracking", False)
    if use_gpu and (cp is not None):
        try:
            # 这里会访问 CUDA Runtime，如果缺 DLL 会抛异常
            _ = cp.cuda.runtime.getVersion()
            return cp
        except Exception as e:
            print("[WARN] CuPy/CUDA 不可用，自动回退到 NumPy (CPU)。")
            print(f"       原因: {e}")
            return np
    return np


# ======================================================================
# 1. 原始文件版 tracking：保持 NumPy 实现，不做 GPU（方便对比/回退）
# ======================================================================
def tracking(
    f,  # Python 的二进制文件对象
    channel: Sequence,
    settings,
) -> Tuple[Sequence[SimpleNamespace], Sequence]:
    """
    Python 版 tracking.m（基于“文件流”逐毫秒读取数据）
    """

    n_ch = settings.numberOfChannels
    code_periods = settings.msToProcess

    # 打印控制（带默认值）
    verbose = getattr(settings, "verboseTracking", True)
    print_interval = getattr(settings, "trackingPrintInterval", 1000)

    # 手动停止开关（可选）
    enable_manual_stop = getattr(settings, "enableManualStopTracking", False)

    # ================= 初始化 trackResults 结构 =================
    # 【修复后】每个通道拥有独立的内存空间
    track_results = _init_track_results(code_periods, n_ch)

    # ================= 跟踪环参数 =================

    # DLL
    early_late_spc = settings.dllCorrelatorSpacing  # [chips]
    PDI_code = 0.001  # [s]
    tau1_code, tau2_code = calc_loop_coef(
        settings.dllNoiseBandwidth,
        settings.dllDampingRatio,
        1.0,
    )

    # PLL
    PDI_carr = 0.001  # [s]
    tau1_carr, tau2_carr = calc_loop_coef(
        settings.pllNoiseBandwidth,
        settings.pllDampingRatio,
        0.25,
    )

    dtype = _dtype_from_string(settings.dataType)
    bytes_per_sample = np.dtype(dtype).itemsize

    fs = float(settings.samplingFreq)
    code_len = float(settings.codeLength)
    code_freq_basis = float(settings.codeFreqBasis)

    stop_all = False  # 用于在任意通道/任意毫秒请求停止时，提前退出所有通道

    # ================== 通道循环 ==================
    for ch_idx in range(n_ch):
        ch = channel[ch_idx]
        prn = _as_attr(ch, "PRN")

        if prn == 0:
            # 该通道未分配卫星
            continue

        tr = track_results[ch_idx]
        tr.PRN = prn

        # === 将文件指针移动到该通道的起始码相位对应的位置 ===
        code_phase_samples = int(_as_attr(ch, "codePhase"))
        offset_bytes = settings.skipNumberOfBytes + (code_phase_samples - 1) * bytes_per_sample
        f.seek(offset_bytes, 0)

        # === 生成 C/A 码，并前后各扩展一个 chip ===
        ca = generate_ca_code(prn)  # 长度 1023，元素 +/-1
        ca_ext = np.concatenate([[ca[-1]], ca, [ca[0]]])  # 长度 1025

        # === 初始化本地 NCO / 相位等变量 ===
        code_freq = code_freq_basis
        rem_code_phase = 0.0

        carr_freq = float(_as_attr(ch, "acquiredFreq"))
        carr_freq_basis = carr_freq
        rem_carr_phase = 0.0

        old_code_nco = 0.0
        old_code_err = 0.0

        old_carr_nco = 0.0
        old_carr_err = 0.0

        try:
            # ================== 毫秒循环 ==================
            for loop_cnt in range(code_periods):
                if verbose and ((loop_cnt + 1) % print_interval == 0):
                    msg = (
                        f"Tracking: Ch {ch_idx+1}/{n_ch} "
                        f"PRN {prn}; {loop_cnt+1}/{code_periods} ms"
                    )
                    if enable_manual_stop:
                        msg += "   [按 Q 手动停止 / Ctrl+C 强制退出]"
                    else:
                        msg += "   [Ctrl+C 强制退出]"
                    print(msg)

                # === 手动停止逻辑 ===
                if enable_manual_stop and (msvcrt is not None) and msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"q", b"Q"):
                        print(
                            f"\n[Tracking] 用户按下 Q，提前终止跟踪 "
                            f"(Ch {ch_idx+1}, PRN {prn}, {loop_cnt+1}/{code_periods} ms)"
                        )
                        tr.status = "K"  # Killed / 手动终止
                        stop_all = True
                        break

                # ---------- 读取本 ms 数据 ----------
                code_phase_step = code_freq / fs
                blksize = int(np.ceil((code_len - rem_code_phase) / code_phase_step))

                raw = np.fromfile(f, dtype=dtype, count=blksize)
                if raw.size != blksize:
                    print("Not able to read the specified number of samples for tracking, exiting!")
                    stop_all = True
                    break

                raw = raw.astype(float)

                # ---------- 生成 E/P/L 码序列 ----------
                idx_array = np.arange(blksize, dtype=float)

                # Early
                t_early = rem_code_phase - early_late_spc + idx_array * code_phase_step
                idx_e = np.ceil(t_early).astype(int)
                early_code = ca_ext[idx_e]

                # Late
                t_late = rem_code_phase + early_late_spc + idx_array * code_phase_step
                idx_l = np.ceil(t_late).astype(int)
                late_code = ca_ext[idx_l]

                # Prompt
                t_prompt = rem_code_phase + idx_array * code_phase_step
                idx_p = np.ceil(t_prompt).astype(int)
                prompt_code = ca_ext[idx_p]

                # 更新余码相位（下一毫秒的起点）
                rem_code_phase = (t_prompt[-1] + code_phase_step) - (code_len - 0.0)

                # ---------- 生成本地载波 ----------
                time = np.arange(blksize + 1, dtype=float) / fs
                trigarg = 2.0 * np.pi * carr_freq * time + rem_carr_phase
                rem_carr_phase = np.mod(trigarg[-1], 2.0 * np.pi)

                carr_cos = np.cos(trigarg[:-1])
                carr_sin = np.sin(trigarg[:-1])

                q_baseband = carr_cos * raw
                i_baseband = carr_sin * raw

                # ---------- 相关积分 ----------
                I_E = float(np.sum(early_code * i_baseband))
                Q_E = float(np.sum(early_code * q_baseband))
                I_P = float(np.sum(prompt_code * i_baseband))
                Q_P = float(np.sum(prompt_code * q_baseband))
                I_L = float(np.sum(late_code * i_baseband))
                Q_L = float(np.sum(late_code * q_baseband))

                # ================= PLL：载波环 =================
                if I_P != 0.0:
                    carr_err = np.arctan(Q_P / I_P) / (2.0 * np.pi)
                else:
                    carr_err = 0.0

                carr_nco = (
                    old_carr_nco
                    + (tau2_carr / tau1_carr) * (carr_err - old_carr_err)
                    + carr_err * (PDI_carr / tau1_carr)
                )
                old_carr_nco = carr_nco
                old_carr_err = carr_err

                carr_freq = carr_freq_basis + carr_nco
                tr.carrFreq[loop_cnt] = carr_freq

                # ================= DLL：码环 =================
                mag_E = np.hypot(I_E, Q_E)
                mag_L = np.hypot(I_L, Q_L)
                denom = mag_E + mag_L

                if denom != 0.0:
                    code_err = (mag_E - mag_L) / denom
                else:
                    code_err = 0.0

                code_nco = (
                    old_code_nco
                    + (tau2_code / tau1_code) * (code_err - old_code_err)
                    + code_err * (PDI_code / tau1_code)
                )
                old_code_nco = code_nco
                old_code_err = code_err

                code_freq = code_freq_basis - code_nco
                tr.codeFreq[loop_cnt] = code_freq

                # ================= 记录测量值 =================
                tr.absoluteSample[loop_cnt] = f.tell()

                tr.dllDiscr[loop_cnt] = code_err
                tr.dllDiscrFilt[loop_cnt] = code_nco
                tr.pllDiscr[loop_cnt] = carr_err
                tr.pllDiscrFilt[loop_cnt] = carr_nco

                tr.I_E[loop_cnt] = I_E
                tr.I_P[loop_cnt] = I_P
                tr.I_L[loop_cnt] = I_L
                tr.Q_E[loop_cnt] = Q_E
                tr.Q_P[loop_cnt] = Q_P
                tr.Q_L[loop_cnt] = Q_L

        except KeyboardInterrupt:
            print(
                f"\n[Tracking] 检测到 Ctrl+C，中止跟踪 "
                f"(Ch {ch_idx+1}, PRN {prn})"
            )
            tr.status = "K"
            stop_all = True

        # 当前通道结束后，如果收到全局停止信号，则不再继续后续通道
        if stop_all:
            break

        tr.status = _as_attr(ch, "status")

    return track_results, channel


# ======================================================================
# 2. 【推荐用于仿真】内存数组版 tracking：支持 CPU / GPU 可切换
# ======================================================================
def tracking_from_array(
    raw_signal: np.ndarray,
    channel: Sequence,
    settings,
) -> Tuple[Sequence[SimpleNamespace], Sequence]:
    """
    内存数组版 tracking，用于仿真
    """

    n_ch = settings.numberOfChannels
    code_periods = settings.msToProcess

    # 选择数组后端：np 或 cp
    xp = _get_xp(settings)

    # 打印控制（带默认值）
    verbose = getattr(settings, "verboseTracking", True)
    print_interval = getattr(settings, "trackingPrintInterval", 1000)

    # 手动停止开关（可选）
    enable_manual_stop = getattr(settings, "enableManualStopTracking", False)

    # ================= 初始化 trackResults 结构 =================
    # 【修复后】每个通道拥有独立的内存空间
    track_results = _init_track_results(code_periods, n_ch)

    # ================= 跟踪环参数 =================

    # DLL
    early_late_spc = settings.dllCorrelatorSpacing  # [chips]
    PDI_code = 0.001  # [s]
    tau1_code, tau2_code = calc_loop_coef(
        settings.dllNoiseBandwidth,
        settings.dllDampingRatio,
        1.0,
    )

    # PLL
    PDI_carr = 0.001  # [s]
    tau1_carr, tau2_carr = calc_loop_coef(
        settings.pllNoiseBandwidth,
        settings.pllDampingRatio,
        0.25,
    )

    dtype = _dtype_from_string(settings.dataType)
    bytes_per_sample = np.dtype(dtype).itemsize

    # 确保输入为 numpy 数组，类型匹配
    if not isinstance(raw_signal, np.ndarray):
        raw_signal = np.asarray(raw_signal)
    if raw_signal.dtype != dtype:
        raw_signal = raw_signal.astype(dtype, copy=False)

    # 整个文件头部被跳过的样本数（以样本为单位）
    skip_samples = settings.skipNumberOfBytes // bytes_per_sample

    total_samples = raw_signal.size
    fs = float(settings.samplingFreq)
    code_len = float(settings.codeLength)
    code_freq_basis = float(settings.codeFreqBasis)

    stop_all = False  # 全局停止标志

    # ================== 通道循环 ==================
    for ch_idx in range(n_ch):
        ch = channel[ch_idx]
        prn = _as_attr(ch, "PRN")

        if prn == 0:
            continue

        tr = track_results[ch_idx]
        tr.PRN = prn

        # === 该通道对应的起始样本位置（整体数组上的索引） ===
        code_phase_samples = int(_as_attr(ch, "codePhase"))
        pos = skip_samples + (code_phase_samples - 1)

        # === 生成 C/A 码，并前后各扩展一个 chip（先 numpy，后转 xp） ===
        ca = generate_ca_code(prn)
        ca_ext = np.concatenate([[ca[-1]], ca, [ca[0]]])
        ca_ext_xp = xp.asarray(ca_ext)

        # === 初始化本地 NCO / 相位等变量 ===
        code_freq = code_freq_basis
        rem_code_phase = 0.0

        carr_freq = float(_as_attr(ch, "acquiredFreq"))
        carr_freq_basis = carr_freq
        rem_carr_phase = 0.0

        old_code_nco = 0.0
        old_code_err = 0.0

        old_carr_nco = 0.0
        old_carr_err = 0.0

        try:
            # ================== 毫秒循环 ==================
            for loop_cnt in range(code_periods):
                if verbose and ((loop_cnt + 1) % print_interval == 0):
                    backend = "GPU" if (xp is cp) else "CPU"
                    msg = (
                        f"[{backend} ARRAY] Tracking: Ch {ch_idx+1}/{n_ch} "
                        f"PRN {prn}; {loop_cnt+1}/{code_periods} ms"
                    )
                    if enable_manual_stop:
                        msg += "   [按 Q 手动停止 / Ctrl+C 强制退出]"
                    else:
                        msg += "   [Ctrl+C 强制退出]"
                    print(msg)

                # === 手动停止逻辑 ===
                if enable_manual_stop and (msvcrt is not None) and msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"q", b"Q"):
                        print(
                            f"\n[ARRAY Tracking] 用户按下 Q，提前终止跟踪 "
                            f"(Ch {ch_idx+1}, PRN {prn}, {loop_cnt+1}/{code_periods} ms)"
                        )
                        tr.status = "K"
                        stop_all = True
                        break

                # ---------- 读取本 ms 数据 ----------
                code_phase_step = code_freq / fs
                blksize = int(np.ceil((code_len - rem_code_phase) / code_phase_step))

                if pos + blksize > total_samples:
                    print(
                        f"[ARRAY] Not enough samples for tracking "
                        f"(ch={ch_idx+1}, loop={loop_cnt+1}), exiting!"
                    )
                    stop_all = True
                    break

                # numpy 切片 → xp.asarray
                raw_np = raw_signal[pos : pos + blksize]
                raw = xp.asarray(raw_np, dtype=float)
                pos += blksize

                # ---------- 生成 E/P/L 码序列 ----------
                idx_array = xp.arange(blksize, dtype=float)

                # Early
                t_early = rem_code_phase - early_late_spc + idx_array * code_phase_step
                idx_e = xp.ceil(t_early).astype(int)
                early_code = ca_ext_xp[idx_e]

                # Late
                t_late = rem_code_phase + early_late_spc + idx_array * code_phase_step
                idx_l = xp.ceil(t_late).astype(int)
                late_code = ca_ext_xp[idx_l]

                # Prompt
                t_prompt = rem_code_phase + idx_array * code_phase_step
                idx_p = xp.ceil(t_prompt).astype(int)
                prompt_code = ca_ext_xp[idx_p]

                # 更新余码相位
                t_prompt_last = float(t_prompt[-1])
                rem_code_phase = (t_prompt_last + code_phase_step) - (code_len - 0.0)

                # ---------- 生成本地载波 ----------
                time = xp.arange(blksize + 1, dtype=float) / fs
                trigarg = 2.0 * np.pi * carr_freq * time + rem_carr_phase
                rem_carr_phase = float(trigarg[-1] % (2.0 * np.pi))

                carr_cos = xp.cos(trigarg[:-1])
                carr_sin = xp.sin(trigarg[:-1])

                q_baseband = carr_cos * raw
                i_baseband = carr_sin * raw

                # ---------- 相关积分 ----------
                I_E = float(xp.sum(early_code * i_baseband))
                Q_E = float(xp.sum(early_code * q_baseband))
                I_P = float(xp.sum(prompt_code * i_baseband))
                Q_P = float(xp.sum(prompt_code * q_baseband))
                I_L = float(xp.sum(late_code * i_baseband))
                Q_L = float(xp.sum(late_code * q_baseband))

                # ================= PLL：载波环 =================
                if I_P != 0.0:
                    carr_err = np.arctan(Q_P / I_P) / (2.0 * np.pi)
                else:
                    carr_err = 0.0

                carr_nco = (
                    old_carr_nco
                    + (tau2_carr / tau1_carr) * (carr_err - old_carr_err)
                    + carr_err * (PDI_carr / tau1_carr)
                )
                old_carr_nco = carr_nco
                old_carr_err = carr_err

                carr_freq = carr_freq_basis + carr_nco
                tr.carrFreq[loop_cnt] = carr_freq

                # ================= DLL：码环 =================
                mag_E = np.hypot(I_E, Q_E)
                mag_L = np.hypot(I_L, Q_L)
                denom = mag_E + mag_L

                if denom != 0.0:
                    code_err = (mag_E - mag_L) / denom
                else:
                    code_err = 0.0

                code_nco = (
                    old_code_nco
                    + (tau2_code / tau1_code) * (code_err - old_code_err)
                    + code_err * (PDI_code / tau1_code)
                )
                old_code_nco = code_nco
                old_code_err = code_err

                code_freq = code_freq_basis - code_nco
                tr.codeFreq[loop_cnt] = code_freq

                # ================= 记录测量值 =================
                tr.absoluteSample[loop_cnt] = pos * bytes_per_sample

                tr.dllDiscr[loop_cnt] = code_err
                tr.dllDiscrFilt[loop_cnt] = code_nco
                tr.pllDiscr[loop_cnt] = carr_err
                tr.pllDiscrFilt[loop_cnt] = carr_nco

                tr.I_E[loop_cnt] = I_E
                tr.I_P[loop_cnt] = I_P
                tr.I_L[loop_cnt] = I_L
                tr.Q_E[loop_cnt] = Q_E
                tr.Q_P[loop_cnt] = Q_P
                tr.Q_L[loop_cnt] = Q_L

        except KeyboardInterrupt:
            print(
                f"\n[ARRAY Tracking] 检测到 Ctrl+C，中止跟踪 "
                f"(Ch {ch_idx+1}, PRN {prn})"
            )
            tr.status = "K"
            stop_all = True

        if stop_all:
            break

        tr.status = _as_attr(ch, "status")

    return track_results, channel