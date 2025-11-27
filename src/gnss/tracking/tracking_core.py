# placeholder for tracking core

# src/gnss/tracking/tracking_core.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np

from gnss.tracking.loop.calc_loop_coef import calc_loop_coef
from gnss.tracking.pre_run import Channel  # 如果你有类型定义的话，可选
from gnss.tracking.generate_ca_code import generate_ca_code  # 需要你提供对应 Python 版


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


def tracking(
    f,  # Python 的二进制文件对象
    channel: Sequence,
    settings,
) -> Tuple[Sequence[SimpleNamespace], Sequence]:
    """
    Python 版 tracking.m

    参数
    ----
    f : file-like
        已用 'rb' 打开的原始数据文件对象。
    channel : 序列
        每个元素包含：
            - PRN
            - codePhase
            - acquiredFreq
            - status
    settings : Settings
        需要字段：
            - numberOfChannels
            - msToProcess
            - dllCorrelatorSpacing
            - dllNoiseBandwidth
            - dllDampingRatio
            - pllNoiseBandwidth
            - pllDampingRatio
            - dataType
            - samplingFreq
            - codeLength
            - codeFreqBasis
            - skipNumberOfBytes
    """

    n_ch = settings.numberOfChannels
    code_periods = settings.msToProcess

    # ================= 初始化 trackResults 结构 =================
    template = SimpleNamespace(
        status="-",
        PRN=0,
        absoluteSample=np.zeros(code_periods, dtype=np.int64),
        codeFreq=np.full(code_periods, np.inf, dtype=float),
        carrFreq=np.full(code_periods, np.inf, dtype=float),
        I_P=np.zeros(code_periods, dtype=float),
        I_E=np.zeros(code_periods, dtype=float),
        I_L=np.zeros(code_periods, dtype=float),
        Q_E=np.zeros(code_periods, dtype=float),
        Q_P=np.zeros(code_periods, dtype=float),
        Q_L=np.zeros(code_periods, dtype=float),
        dllDiscr=np.full(code_periods, np.inf, dtype=float),
        dllDiscrFilt=np.full(code_periods, np.inf, dtype=float),
        pllDiscr=np.full(code_periods, np.inf, dtype=float),
        pllDiscrFilt=np.full(code_periods, np.inf, dtype=float),
    )

    track_results = [SimpleNamespace(**template.__dict__) for _ in range(n_ch)]

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
        # MATLAB: fseek(fid, skipBytes + codePhase-1, 'bof')
        # 这里假定 1 个样本 = 1 个 dataType 的字长
        code_phase_samples = int(_as_attr(ch, "codePhase"))
        offset_bytes = settings.skipNumberOfBytes + (code_phase_samples - 1) * bytes_per_sample
        f.seek(offset_bytes, 0)

        # === 生成 C/A 码，并前后各扩展一个 chip ===
        ca = generate_ca_code(prn)  # 长度 1023，元素 +/-1
        ca_ext = np.concatenate([[ca[-1]], ca, [ca[0]]])  # 长度 1025

        # === 初始化本地 NCO / 相位等变量 ===
        code_freq = settings.codeFreqBasis
        rem_code_phase = 0.0

        carr_freq = float(_as_attr(ch, "acquiredFreq"))
        carr_freq_basis = carr_freq
        rem_carr_phase = 0.0

        old_code_nco = 0.0
        old_code_err = 0.0

        old_carr_nco = 0.0
        old_carr_err = 0.0

        # ================== 毫秒循环 ==================
        for loop_cnt in range(code_periods):
            # 简单进度输出（可选）
            if (loop_cnt + 1) % 1000 == 0:
                print(
                    f"Tracking: Ch {ch_idx+1}/{n_ch} "
                    f"PRN {prn}; {loop_cnt+1}/{code_periods} ms"
                )

            # ---------- 读取本 ms 数据 ----------
            code_phase_step = code_freq / settings.samplingFreq
            # blksize 对应 MATLAB: ceil((codeLength - remCodePhase) / codePhaseStep)
            blksize = int(
                np.ceil((settings.codeLength - rem_code_phase) / code_phase_step)
            )

            raw = np.fromfile(f, dtype=dtype, count=blksize)
            if raw.size != blksize:
                print("Not able to read the specified number of samples for tracking, exiting!")
                return track_results, channel

            raw = raw.astype(float)

            # ---------- 生成 E/P/L 码序列 ----------
            idx_array = np.arange(blksize)

            # Early
            t_early = rem_code_phase - early_late_spc + idx_array * code_phase_step
            idx_e = np.ceil(t_early).astype(int)  # 0-based index into ca_ext
            early_code = ca_ext[idx_e]

            # Late
            t_late = rem_code_phase + early_late_spc + idx_array * code_phase_step
            idx_l = np.ceil(t_late).astype(int)
            late_code = ca_ext[idx_l]

            # Prompt
            t_prompt = rem_code_phase + idx_array * code_phase_step
            idx_p = np.ceil(t_prompt).astype(int)
            prompt_code = ca_ext[idx_p]

            # 更新剩余码相位（基于 Prompt 轨迹）
            rem_code_phase = (t_prompt[-1] + code_phase_step) - 1023.0

            # ---------- 生成本地载波 ----------
            time = np.arange(blksize + 1) / settings.samplingFreq
            trigarg = 2.0 * np.pi * carr_freq * time + rem_carr_phase
            rem_carr_phase = np.mod(trigarg[-1], 2.0 * np.pi)

            carr_cos = np.cos(trigarg[:-1])
            carr_sin = np.sin(trigarg[:-1])

            # ---------- 混频到基带 & 相关 ----------
            q_baseband = carr_cos * raw  # Q 分支
            i_baseband = carr_sin * raw  # I 分支

            I_E = float(np.sum(early_code * i_baseband))
            Q_E = float(np.sum(early_code * q_baseband))
            I_P = float(np.sum(prompt_code * i_baseband))
            Q_P = float(np.sum(prompt_code * q_baseband))
            I_L = float(np.sum(late_code * i_baseband))
            Q_L = float(np.sum(late_code * q_baseband))

            # ================= PLL：载波环 =================
            # Costas atan 鉴相器
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

            code_freq = settings.codeFreqBasis - code_nco
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

        # 通道状态：目前简单复制 acquisition/pre_run 中的状态
        tr.status = _as_attr(ch, "status")

    return track_results, channel
