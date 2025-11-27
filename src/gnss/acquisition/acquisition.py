import numpy as np
from .settings import GnssSettings

def acquisition_parallel_code_phase_search(long_signal: np.ndarray,
                                           settings: GnssSettings):
    """
    Python 版并行码相位搜索骨架：
    - 输入 11 ms 信号
    - 返回 acq_results, debug_results（先用简单结构代替）
    """
    # 计算每毫秒采样点数，与 Matlab 一致
    samples_per_code = int(round(
        settings.sampling_freq /
        (settings.code_freq_basis / settings.code_length)
    ))

    # 这里先打印一下，验证参数是否正确
    print(f"samples_per_code = {samples_per_code}")
    print(f"signal length    = {len(long_signal)}")

    # TODO: 后面我们一步步实现真正的并行码相位搜索

    acq_results = {}   # 以后可以做成 dict: {prn: {freq, code_phase, peak_metric}}
    debug_results = {} # 以后放 2D 相关矩阵等

    return acq_results, debug_results