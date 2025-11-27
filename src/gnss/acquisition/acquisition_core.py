# placeholder for acquisition core

import numpy as np

from .ca_code import make_ca_table, generate_ca_code


def _nextpow2(n: int) -> int:
    """返回 >= n 的下一个 2 的整数次幂。"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def acquisition(long_signal: np.ndarray, settings):
    """
    功能：对采集到的“数据”执行冷启动捕获，搜索 settings.acqSatelliteList 中所有 PRN。
    对每颗卫星返回：
        - 载波频率 carr_freq
        - 码相位   code_phase
        - 峰值比   peak_metric

    参数
    ----
    long_signal : np.ndarray
        11 ms 原始前端信号，实/复数均可（本代码按实数处理，与原 MATLAB 一致）。
    settings : object
        拥有以下属性的对象（或你可改成 dict 索引方式）：
            samplingFreq    : 采样频率 [Hz]
            codeFreqBasis   : C/A 码速率 [Hz] (1.023e6)
            codeLength      : C/A 码长度 (1023)
            IF              : 中频频率 [Hz]
            acqSearchBand   : 捕获搜索带宽 [kHz]（总带宽）
            acqSatelliteList: 要捕获的 PRN 列表 (例如 [1, 2, ..., 32])
            acqThreshold    : 峰值比阈值
    """

    # ========== 1. 初始化 ==========

    # 每个 C/A 码周期的采样点数 (1 ms)
    samples_per_code = int(
        round(
            settings.samplingFreq /
            (settings.codeFreqBasis / settings.codeLength)
        )
    )

    # 取前 2 ms 用于粗捕获
    signal1 = long_signal[0:samples_per_code]
    signal2 = long_signal[samples_per_code:2 * samples_per_code]

    # 去直流，用于精细频率搜索
    signal0_dc = long_signal - np.mean(long_signal)

    # 采样周期
    ts = 1.0 / settings.samplingFreq

    # 本地载波相位点
    phase_points = np.arange(samples_per_code) * 2.0 * np.pi * ts

    # 频率搜索步进 0.5 kHz，acqSearchBand 是总带宽（单位 kHz）
    # 例如 acqSearchBand = 4 -> 搜索 -2kHz ~ +2kHz，步长 0.5kHz
    number_of_frq_bins = int(round(settings.acqSearchBand * 2)) + 1

    # 生成所有 PRN 的采样后 C/A 码表
    ca_codes_table = make_ca_table(settings)  # shape ~ (max_prn, samples_per_code)

    # 预分配
    results = np.zeros((number_of_frq_bins, samples_per_code), dtype=float)
    frq_bins = np.zeros(number_of_frq_bins, dtype=float)

    # acqResults 结构体（用 dict 存）
    max_prn = 32  # softGNSS 默认 1~32
    acq_results = {
        "carr_freq": np.zeros(max_prn, dtype=float),
        "code_phase": np.zeros(max_prn, dtype=int),
        "peak_metric": np.zeros(max_prn, dtype=float),
    }

    print("(", end="", flush=True)

    # ========== 2. 对列表中所有 PRN 进行搜索 ==========

    for prn in settings.acqSatelliteList:
        prn_idx = prn - 1  # Python 下标，从 0 开始

        # --- C/A 码 FFT（取共轭用于频域相关） ---
        ca_code_time = ca_codes_table[prn_idx, :]  # shape: (samples_per_code,)
        ca_code_freq_dom = np.conj(np.fft.fft(ca_code_time))

        # --- 在所有频点上做频域相关 ---
        for frq_bin_index in range(number_of_frq_bins):
            # 当前频点对应的绝对载波频率
            frq_bins[frq_bin_index] = (
                settings.IF
                - (settings.acqSearchBand / 2.0) * 1000.0
                + 0.5e3 * frq_bin_index
            )

            carr = frq_bins[frq_bin_index] * phase_points
            sin_carr = np.sin(carr)
            cos_carr = np.cos(carr)

            # 下变频到基带：I/Q
            i1 = sin_carr * signal1
            q1 = cos_carr * signal1
            i2 = sin_carr * signal2
            q2 = cos_carr * signal2

            iq_freq_dom1 = np.fft.fft(i1 + 1j * q1)
            iq_freq_dom2 = np.fft.fft(i2 + 1j * q2)

            conv_code_iq1 = iq_freq_dom1 * ca_code_freq_dom
            conv_code_iq2 = iq_freq_dom2 * ca_code_freq_dom

            acq_res1 = np.abs(np.fft.ifft(conv_code_iq1)) ** 2
            acq_res2 = np.abs(np.fft.ifft(conv_code_iq2)) ** 2

            # 选功率更大的那一毫秒，兼顾数据位翻转问题
            if acq_res1.max() > acq_res2.max():
                results[frq_bin_index, :] = acq_res1
            else:
                results[frq_bin_index, :] = acq_res2

        # ========== 3. 在 2D 结果中寻找相关峰值 ==========

        # 全局最大峰值 — 得到峰值大小及其频点 / 码相位索引
        peak_size = results.max()
        max_flat_index = results.argmax()
        freq_bin_index, code_phase = np.unravel_index(
            max_flat_index, results.shape
        )

        # 每个码片的采样点数
        samples_per_chip = int(
            round(settings.samplingFreq / settings.codeFreqBasis)
        )

        # 排除最高峰附近 +-1 码片范围，寻找次高峰
        valid_indices = np.ones(samples_per_code, dtype=bool)
        for i in range(code_phase - samples_per_chip,
                       code_phase + samples_per_chip + 1):
            idx = i % samples_per_code  # 环绕索引
            valid_indices[idx] = False

        code_phase_range = np.nonzero(valid_indices)[0]
        second_peak_size = results[freq_bin_index, code_phase_range].max()

        peak_metric = peak_size / second_peak_size
        acq_results["peak_metric"][prn_idx] = peak_metric

        # ========== 4. 阈值判断 & 精细频率搜索 ==========

        if peak_metric > settings.acqThreshold:
            # 检测到信号
            print(f"{prn:02d} ", end="", flush=True)

            # --- 生成 10 ms 长的 C/A 码 ---
            ca_code_1ms = generate_ca_code(prn)  # 长度应为 1023
            # 时间序列（从 1 到 10*samples_per_code，与 MATLAB 对应）
            t = ts * np.arange(1, 10 * samples_per_code + 1)
            chip_period = 1.0 / settings.codeFreqBasis
            code_value_index = np.floor(t / chip_period).astype(int)
            # 环绕到 0..1022
            idx = np.mod(code_value_index, 1023)
            long_ca_code = ca_code_1ms[idx]

            # --- 利用粗码相位，从原始信号中截取 10 ms 信号并去除 C/A 码调制 ---
            start = code_phase  # code_phase 已是 0-based
            stop = start + 10 * samples_per_code
            x_carrier = signal0_dc[start:stop] * long_ca_code

            # --- 精细频率 FFT 搜索 ---
            fft_num_pts = 8 * _nextpow2(len(x_carrier))
            fft_xc = np.abs(np.fft.fft(x_carrier, fft_num_pts))

            uniq_fft_pts = (fft_num_pts + 1) // 2
            # 对应 MATLAB: fftxc(5 : uniqFftPts-5)
            search_slice = fft_xc[4:uniq_fft_pts - 5]
            fft_max_index_rel = np.argmax(search_slice)
            fft_max_index = fft_max_index_rel + 4  # 补回偏移

            fft_freq_bins = (
                np.arange(uniq_fft_pts) * settings.samplingFreq / fft_num_pts
            )

            acq_results["carr_freq"][prn_idx] = fft_freq_bins[fft_max_index]
            acq_results["code_phase"][prn_idx] = code_phase

        else:
            # 未检测到此 PRN
            print(". ", end="", flush=True)

    print(")")

    return acq_results
