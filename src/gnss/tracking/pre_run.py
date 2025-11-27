"""
preRun(acqResults, settings)

功能：
    初始化所有跟踪通道（Tracking Channels）。
    根据捕获(acquisition)结果，将最强的卫星信号分配到各跟踪通道中，
    为后续 DLL/PLL 跟踪环路做好准备。

输入：
    acqResults:
        可以是 dict 或对象，字段/属性名支持多种写法：
        - 峰值度量：
            peakMetric / peak_metric / peakRatio / peak_ratio
        - 载波频率：
            carrFreq / carr_freq / carrier_freq
        - 码相位：
            codePhase / code_phase
        - PRN（可选）：
            PRN / prn / sat_prn

    settings:
        - settings.numberOfChannels: 跟踪通道数量（如 12 或 32）

输出：
    channels:
        一个列表，每个元素为一个 dict，表示一个通道的状态：
        {
            "PRN": int,          # 跟踪的卫星号
            "acquiredFreq": float,
            "codePhase": int,
            "status": "T" 或 "-"
        }
"""

from typing import List, Dict, Any
import numpy as np


def _get_field(acq: Any, candidates, desc: str):
    """
    从 acq 中取字段/属性：
    - 如果 acq 是 dict，就在 candidates 里找第一个存在的 key
    - 如果 acq 是对象，就在 candidates 里找第一个存在的属性
    """
    if isinstance(acq, dict):
        for k in candidates:
            if k in acq:
                return acq[k]
        raise KeyError(
            f"pre_run: 在 acqResults 中找不到 {desc} 字段，"
            f"尝试过: {candidates}；当前 keys = {list(acq.keys())}"
        )
    else:
        for k in candidates:
            if hasattr(acq, k):
                return getattr(acq, k)
        raise AttributeError(
            f"pre_run: 在 acqResults 对象中找不到 {desc} 属性，"
            f"尝试过: {candidates}"
        )


def pre_run(acqResults: Any, settings: Any) -> List[Dict[str, Any]]:
    """
    Python 版 preRun() —— 初始化跟踪通道
    """

    # ===============================================================
    # 0. 从 acqResults 中读出各字段（自动兼容命名风格）
    # ===============================================================
    peak_metric = np.asarray(
        _get_field(
            acqResults,
            ["peakMetric", "peak_metric", "peakRatio", "peak_ratio"],
            "峰值度量",
        ),
        dtype=float,
    )

    carr_freq = np.asarray(
        _get_field(
            acqResults,
            ["carrFreq", "carr_freq", "carrier_freq"],
            "载波频率",
        ),
        dtype=float,
    )

    code_phase = np.asarray(
        _get_field(
            acqResults,
            ["codePhase", "code_phase"],
            "码相位",
        ),
        dtype=int,
    )

    # PRN 信息可选，如果没有，就默认 1,2,3...
    try:
        prn_array = np.asarray(
            _get_field(
                acqResults,
                ["PRN", "prn", "sat_prn"],
                "PRN 号",
            ),
            dtype=int,
        )
    except (KeyError, AttributeError):
        prn_array = np.arange(1, len(peak_metric) + 1, dtype=int)

    # 基本长度一致性检查
    if not (
        len(peak_metric) == len(carr_freq) == len(code_phase) == len(prn_array)
    ):
        raise ValueError(
            "pre_run: acqResults 中各字段长度不一致：\n"
            f"  peak_metric: {len(peak_metric)}\n"
            f"  carr_freq  : {len(carr_freq)}\n"
            f"  code_phase : {len(code_phase)}\n"
            f"  PRN        : {len(prn_array)}"
        )

    # ===============================================================
    # 1. 创建所有通道的数据结构（初始为空闲状态）
    # ===============================================================
    channels: List[Dict[str, Any]] = []

    empty_channel: Dict[str, Any] = {
        "PRN": 0,             # 分配到的卫星 PRN 号
        "acquiredFreq": 0.0,  # 捕获载波频率（作为 NCO 起始值）
        "codePhase": 0,       # 捕获码相位（Tracking 起点）
        "status": "-",        # "-" 表示空闲；"T" 表示 Tracking 状态
    }

    for _ in range(settings.numberOfChannels):
        channels.append(empty_channel.copy())

    # ===============================================================
    # 2. 找出“捕获成功”的卫星，并按 peakMetric 从大到小排序
    #    （载波频率 > 0 视为捕获成功）
    # ===============================================================
    acquired_mask = carr_freq > 0
    acquired_indices = np.where(acquired_mask)[0]

    if acquired_indices.size == 0:
        # 一个都没捕到，直接返回空闲通道
        print("[pre_run] 警告：没有任何卫星捕获成功，所有通道保持空闲。")
        return channels

    # 仅在“已捕获”的索引集合内，按 peak_metric 降序排序
    sort_order = np.argsort(-peak_metric[acquired_indices])
    sorted_acquired_indices = acquired_indices[sort_order]

    acquired_sat_count = int(sorted_acquired_indices.size)

    # 实际可分配通道数量 = min(可用通道数, 捕获到的卫星数)
    usable_channels = min(settings.numberOfChannels, acquired_sat_count)

    # ===============================================================
    # 3. 将最强的卫星分配给前几个 Tracking 通道
    # ===============================================================
    for ch_idx in range(usable_channels):
        idx = int(sorted_acquired_indices[ch_idx])  # 0-based 索引
        prn = int(prn_array[idx])

        channels[ch_idx]["PRN"] = prn
        channels[ch_idx]["acquiredFreq"] = float(carr_freq[idx])
        channels[ch_idx]["codePhase"] = int(code_phase[idx])
        channels[ch_idx]["status"] = "T"  # 激活通道，进入 Tracking 状态

    # ===============================================================
    # 4. 未分配的通道仍为 "-" 状态
    # ===============================================================
    return channels
