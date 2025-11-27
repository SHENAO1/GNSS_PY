# placeholder for calcLoopCoef conversion
"""
Function: calc_loop_coef

功能：计算锁相环(PLL)或锁数环(DLL)的环路滤波器系数。

描述：
    这个函数根据给定的环路性能指标（噪声带宽、阻尼比）和环路增益，
    计算出二阶环路中比例积分(PI)滤波器的两个关键时间常数 tau1 和 tau2。
    这些系数用于实际的电路或算法设计中，以实现期望的环路动态响应和噪声性能。

函数调用格式：
    tau1, tau2 = calc_loop_coef(LBW, zeta, k)

输入参数 (Inputs):
    LBW : 环路噪声带宽 (Loop Noise Bandwidth)，单位通常是 rad/s (弧度/秒)。
          它衡量了环路的带宽，决定了环路对相位变化的跟踪速度和对噪声的抑制能力。

    zeta : 阻尼比 (Damping Ratio)，无量纲。
           它决定了环路阶跃响应的瞬态特性（如超调和振荡）。
           通常取值在 0.707 附近。

    k : 环路总增益 (Loop Gain)，由鉴相器增益、电荷泵增益、VCO增益等串联而成。

输出参数 (Outputs):
    tau1, tau2 : 环路滤波器的系数 (时间常数)。

        对于一个传递函数为 F(s) = (1 + s*tau2) / (s*tau1) 的无源PI滤波器，
        或者相关的有源PI滤波器结构：

        - tau2 决定了零点位置 (-1/tau2)，与比例路径相关。
        - tau1 决定了积分器增益 (1/tau1)，与积分路径相关。
"""

import numpy as np


def calc_loop_coef(LBW: float, zeta: float, k: float):
    """
    计算 PLL/DLL 二阶环路的 tau1 和 tau2 系数
    """

    # 步骤1: 求解自然角频率 Wn
    # 公式来源：
    #   B_L = (Wn/2) * (zeta + 1/(4*zeta))
    # 这里反推求 Wn：
    #
    # 注意：LBW 输入应为 rad/s，如果原始单位为 Hz，请先转换：
    # LBW_rad = LBW_Hz * 2*pi
    Wn = LBW * 8.0 * zeta / (4.0 * zeta * zeta + 1.0)

    # 步骤2: 求解 tau1 和 tau2
    # 根据标准二阶 PLL 关系式：
    # Wn^2 = k / tau1
    tau1 = k / (Wn * Wn)

    # 根据：
    # 2*zeta*Wn = k * tau2 / tau1
    # 结合上式消元可得：
    # tau2 = 2*zeta / Wn
    tau2 = 2.0 * zeta / Wn

    return tau1, tau2

# 示例用法
""" from gnss.tracking.loop.calc_loop_coef import calc_loop_coef

def run_tracking(signal, settings):
    tau1, tau2 = calc_loop_coef(
        LBW=settings.dll_noise_bw,
        zeta=settings.dll_damping,
        k=settings.dll_gain
    )
    print("DLL Loop Tau1:", tau1)
    print("DLL Loop Tau2:", tau2) """
