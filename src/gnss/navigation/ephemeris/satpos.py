"""
satpos.py

根据广播星历计算 GPS 卫星在给定“传输时刻”的：
- ECEF 坐标 (X, Y, Z)，单位：米
- 卫星钟差校正（包含相对论效应），单位：秒

对应 MATLAB:
    [satPositions, satClkCorr] = satpos(transmitTime, prnList, eph, settings)
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import math
import numpy as np

# 使用你项目里已经写好的 check_t（对应 MATLAB 的 check_t.m）
from gnss.navigation.ephemeris.nav_party_chk import check_t


# ======================= GPS 常数（与 MATLAB 一致） ======================= #
GPS_PI = 3.1415926535898       # GPS 坐标系中的 π
OMEGA_E_DOT = 7.2921151467e-5  # 地球自转角速度 [rad/s]
GM = 3.986005e14               # μ = GM, 地心引力常数 [m^3/s^2]
F = -4.442807633e-10           # 相对论钟差常数 [s / sqrt(m)]


def satpos(
    transmit_time: float,
    prn_list: Iterable[int],
    eph_all: Dict[int, Dict[str, float]],
    settings=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据星历计算多个卫星在“信号传输时刻”的 ECEF 坐标和卫星钟差。

    对应 MATLAB:
        [satPositions, satClkCorr] = satpos(transmitTime, prnList, eph, settings);

    参数
    ----
    transmit_time : float
        信号传输时刻（GPS 时间，秒），通常是接收时刻减去粗略传播时间。
    prn_list : Iterable[int]
        需要计算的 PRN 列表，例如 [1, 3, 8, 11]。
    eph_all : Dict[int, Dict[str, float]]
        所有卫星的星历字典：
            key   = PRN 号 (int)
            value = 对应 PRN 的星历参数字典，字段名与 ephemeris.py 中一致，
                    如 "sqrtA", "e", "t_oe", "M_0", "omega" 等。
    settings : Any
        保留参数（为了接口跟 MATLAB 一致），当前函数内部没有使用，可为 None。

    返回
    ----
    sat_positions : np.ndarray
        形状为 (3, N) 的卫星坐标矩阵，单位：米。
        第 0 行: X 坐标
        第 1 行: Y 坐标
        第 2 行: Z 坐标

    sat_clk_corr : np.ndarray
        长度为 N 的卫星钟差校正（秒）。
        使用时应从观测量对应的时间中减去该值：
            t_corrected = t_measured - sat_clk_corr[k]
    """

    prns = list(prn_list)
    num_sats = len(prns)

    # 预分配结果
    sat_positions = np.zeros((3, num_sats), dtype=float)
    sat_clk_corr = np.zeros(num_sats, dtype=float)

    # =================== 逐颗卫星处理 =================== #
    for idx, prn in enumerate(prns):
        # 取出该 PRN 的星历字典
        eph = eph_all[prn]

        # ================== 1. 初始卫星钟差（不含相对论项） ==================

        # 时间差：当前时刻相对于星历中 t_oc 的偏移（注意要做周界规约）
        dt = check_t(transmit_time - eph["t_oc"])

        # 星历中提供的钟差多项式 + 群延迟差 T_GD
        # satClkCorr = (a_f2 * dt + a_f1) * dt + a_f0 - T_GD
        clk_corr = (eph["a_f2"] * dt + eph["a_f1"]) * dt + eph["a_f0"] - eph["T_GD"]

        # 传播信号的实际发送时刻（考虑钟差）
        time_tx = transmit_time - clk_corr

        # ================== 2. 根据轨道参数计算卫星在该时刻的空间位置 ==========

        # 2.1 恢复轨道半长轴 a
        a = eph["sqrtA"] * eph["sqrtA"]  # sqrtA 的单位是 m^0.5

        # 2.2 与 t_oe 的时间差 tk（同样要做周界规约）
        tk = check_t(time_tx - eph["t_oe"])

        # 2.3 平均角速度 n0 和改正后的平均角速度 n
        n0 = math.sqrt(GM / (a ** 3))
        n = n0 + eph["deltan"]

        # 2.4 平近点角 M
        M = eph["M_0"] + n * tk
        # 规约到 [0, 2π)
        M = math.remainder(M + 2 * GPS_PI, 2 * GPS_PI)

        # 2.5 用迭代求解偏近点角 E
        E = M
        for _ in range(10):
            E_old = E
            E = M + eph["e"] * math.sin(E)
            dE = math.remainder(E - E_old, 2 * GPS_PI)
            if abs(dE) < 1.0e-12:
                break

        # 再次规约 E 到 [0, 2π)
        E = math.remainder(E + 2 * GPS_PI, 2 * GPS_PI)

        # 2.6 相对论钟差改正项 dtr
        dtr = F * eph["e"] * eph["sqrtA"] * math.sin(E)

        # 2.7 真近点角 ν
        sin_E = math.sin(E)
        cos_E = math.cos(E)
        sqrt1_e2 = math.sqrt(1.0 - eph["e"] ** 2)
        nu = math.atan2(sqrt1_e2 * sin_E, cos_E - eph["e"])

        # 2.8 近地点角距 φ = ν + ω
        phi = nu + eph["omega"]
        phi = math.remainder(phi, 2 * GPS_PI)

        # 2.9 轨道摄动改正：轨道幅角 u、轨道半径 r、轨道倾角 i
        u = (
            phi
            + eph["C_uc"] * math.cos(2 * phi)
            + eph["C_us"] * math.sin(2 * phi)
        )

        r = (
            a * (1.0 - eph["e"] * cos_E)
            + eph["C_rc"] * math.cos(2 * phi)
            + eph["C_rs"] * math.sin(2 * phi)
        )

        i = (
            eph["i_0"]
            + eph["iDot"] * tk
            + eph["C_ic"] * math.cos(2 * phi)
            + eph["C_is"] * math.sin(2 * phi)
        )

        # 2.10 升交点经度 Ω(t)
        Omega = (
            eph["omega_0"]
            + (eph["omegaDot"] - OMEGA_E_DOT) * tk
            - OMEGA_E_DOT * eph["t_oe"]
        )
        Omega = math.remainder(Omega + 2 * GPS_PI, 2 * GPS_PI)

        # 2.11 计算 ECEF 坐标
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        cos_i = math.cos(i)
        sin_i = math.sin(i)

        # 对应 MATLAB:
        # X = cos(u)*r * cos(Omega) - sin(u)*r * cos(i)*sin(Omega);
        # Y = cos(u)*r * sin(Omega) + sin(u)*r * cos(i)*cos(Omega);
        # Z = sin(u)*r * sin(i);
        x = cos_u * r * cos_Omega - sin_u * r * cos_i * sin_Omega
        y = cos_u * r * sin_Omega + sin_u * r * cos_i * cos_Omega
        z = sin_u * r * sin_i

        sat_positions[0, idx] = x
        sat_positions[1, idx] = y
        sat_positions[2, idx] = z

        # ================== 3. 把相对论钟差加到总钟差里 ==================

        # 最终钟差 = 多项式钟差 - T_GD + dtr
        sat_clk_corr[idx] = clk_corr + dtr

    return sat_positions, sat_clk_corr
