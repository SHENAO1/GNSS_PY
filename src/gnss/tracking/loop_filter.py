"""环路滤波器系数计算，对应 Matlab calcLoopCoef.m。"""

from dataclasses import dataclass

@dataclass
class LoopFilterParams:
    noise_bandwidth: float  # 噪声带宽 [Hz]
    damping_ratio: float    # 阻尼比
    loop_gain: float        # 环路增益
    sample_interval: float  # 采样间隔 [s]


def calc_loop_coef(params: LoopFilterParams):
    """根据带宽、阻尼比等参数计算二阶环路滤波器系数。

    这里先占位，具体公式以后从 Matlab 迁移。
    """
    raise NotImplementedError("TODO: 从 calcLoopCoef.m 迁移公式")
