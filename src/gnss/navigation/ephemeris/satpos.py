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
    """

    prns = list(prn_list)
    num_sats = len(prns)

    # 预分配结果
    sat_positions = np.zeros((3, num_sats), dtype=float)
    sat_clk_corr = np.zeros(num_sats, dtype=float)

    # =================== 逐颗卫星处理 =================== #
    for idx, prn in enumerate(prns):
        # 取出该 PRN 的星历（可能是 dict，也可能是 SimpleNamespace）
        eph_obj = eph_all[prn]
        if hasattr(eph_obj, "__dict__"):
            eph = eph_obj.__dict__
        else:
            eph = eph_obj

        # ---- 检查是否有计算所需的全部字段，防止 KeyError ----
        required_fields = [
            "sqrtA", "t_oe", "deltan", "M_0", "e",
            "omega", "C_uc", "C_us", "C_rc", "C_rs",
            "i_0", "iDot", "C_ic", "C_is",
            "omega_0", "omegaDot",
            "t_oc", "a_f0", "a_f1", "a_f2", "T_GD",
        ]
        missing = [k for k in required_fields if k not in eph]
        if missing:
            print(f"[satpos] PRN {prn} 星历缺少字段: {missing}，跳过该星。")
            sat_positions[:, idx] = np.nan
            sat_clk_corr[idx] = np.nan
            continue

        # ================== 1. 初始卫星钟差（不含相对论项） ==================
        dt = check_t(transmit_time - eph["t_oc"])

        # 星历中提供的钟差多项式 + 群延迟差 T_GD
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

        x = cos_u * r * cos_Omega - sin_u * r * cos_i * sin_Omega
        y = cos_u * r * sin_Omega + sin_u * r * cos_i * cos_Omega
        z = sin_u * r * sin_i

        sat_positions[0, idx] = x
        sat_positions[1, idx] = y
        sat_positions[2, idx] = z

        # ================== 3. 把相对论钟差加到总钟差里 ==================
        sat_clk_corr[idx] = clk_corr + dtr

    return sat_positions, sat_clk_corr

