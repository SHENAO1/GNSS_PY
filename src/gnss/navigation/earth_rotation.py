import numpy as np
import math


def e_r_corr(traveltime: float, X_sat: np.ndarray) -> np.ndarray:
    """
    地球自转改正（Earth Rotation Correction / Sagnac Effect）

    对应 MATLAB:
        X_sat_rot = e_r_corr(traveltime, X_sat)

    功能：
    ----
    GNSS 信号从卫星到接收机的传播过程中，地球在不断自转，
    因此卫星“发射时刻”的位置与“接收时刻计算得到的位置”并不一致。
    本函数通过绕 Z 轴旋转一个小角度来修正卫星位置。

    参数
    ----
    traveltime : float
        信号传播时间（秒），通常约 0.07 - 0.09 s（视卫星高度而定）

    X_sat : ndarray, shape (3,)
        卫星在接收时刻的 ECEF 坐标 [X, Y, Z]（米）

    返回
    ----
    X_sat_rot : ndarray, shape (3,)
        修正地球自转后的卫星在发射时刻的 ECEF 坐标
    """

    # ====================== 常数：地球自转角速度 ====================== #
    Omega_e_dot = 7.292115147e-5  # rad/s

    # 旋转角度 = Ω_e * signal_travel_time
    omegatau = Omega_e_dot * traveltime

    cos_ot = math.cos(omegatau)
    sin_ot = math.sin(omegatau)

    # 绕 Z 轴的旋转矩阵 R3(ω*t)
    R3 = np.array([
        [cos_ot,  sin_ot, 0.0],
        [-sin_ot, cos_ot, 0.0],
        [0.0,     0.0,    1.0],
    ])

    # 执行旋转
    X_sat_rot = R3 @ X_sat

    return X_sat_rot
