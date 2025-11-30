"""
satpos.py

根据广播星历计算 GPS 卫星在给定“传输时刻”的：
- ECEF 坐标 (X, Y, Z)，单位：米
- 卫星钟差校正（包含相对论效应），单位：秒
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import math
import numpy as np

# 使用项目中的 check_t
from gnss.navigation.ephemeris.ephemeris import check_t

# ======================= GPS 常数 ======================= #
GPS_PI = 3.1415926535898
OMEGA_E_DOT = 7.2921151467e-5
GM = 3.986005e14
F = -4.442807633e-10


# ---------- MATLAB rem(x,2*pi) → Python [0,2π) 包角 ----------
def wrap_0_2pi(x: float) -> float:
    return x % (2.0 * GPS_PI)


def satpos(
    transmit_time: float,
    prn_list: Iterable[int],
    eph_all: Dict[int, Dict[str, float]],
    settings=None,
) -> Tuple[np.ndarray, np.ndarray]:

    prns = list(prn_list)
    num_sats = len(prns)

    sat_positions = np.zeros((3, num_sats), dtype=float)
    sat_clk_corr = np.zeros(num_sats, dtype=float)

    # ===================================================== #
    for idx, prn in enumerate(prns):

        eph_obj = eph_all[prn]
        eph = eph_obj.__dict__ if hasattr(eph_obj, "__dict__") else eph_obj

        # 需要的字段检查
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

        # ================== 1. 初始钟差 ================== #
        dt = check_t(transmit_time - eph["t_oc"])

        clk_corr = (
            (eph["a_f2"] * dt + eph["a_f1"]) * dt
            + eph["a_f0"]
            - eph["T_GD"]
        )

        time_tx = transmit_time - clk_corr

        # ================== 2. 轨道位置 ================== #

        # 2.1 半长轴
        a = eph["sqrtA"] * eph["sqrtA"]

        # 2.2 tk
        tk = check_t(time_tx - eph["t_oe"])

        # 2.3 平均角速度
        n0 = math.sqrt(GM / (a ** 3))
        n = n0 + eph["deltan"]

        # 2.4 平近点角 M
        M = eph["M_0"] + n * tk
        M = wrap_0_2pi(M + 2 * GPS_PI)

        # 2.5 偏近点角 E（10 次迭代）
        E = M
        for _ in range(10):
            E_old = E
            E = M + eph["e"] * math.sin(E)
            if abs(E - E_old) < 1e-12:
                break
        E = wrap_0_2pi(E + 2 * GPS_PI)

        # 2.6 相对论钟差项
        dtr = F * eph["e"] * eph["sqrtA"] * math.sin(E)

        # 2.7 真近点角 ν
        sin_E = math.sin(E)
        cos_E = math.cos(E)
        sqrt1_e2 = math.sqrt(1.0 - eph["e"] ** 2)
        nu = math.atan2(sqrt1_e2 * sin_E, cos_E - eph["e"])

        # 2.8 近地点角距 φ
        phi = wrap_0_2pi(nu + eph["omega"])

        # 2.9 轨道摄动改正
        u = phi + eph["C_uc"] * math.cos(2 * phi) + eph["C_us"] * math.sin(2 * phi)
        r = (
            a * (1 - eph["e"] * cos_E)
            + eph["C_rc"] * math.cos(2 * phi)
            + eph["C_rs"] * math.sin(2 * phi)
        )
        i = (
            eph["i_0"]
            + eph["iDot"] * tk
            + eph["C_ic"] * math.cos(2 * phi)
            + eph["C_is"] * math.sin(2 * phi)
        )

        # 2.10 升交点经度 Ω
        Omega = (
            eph["omega_0"]
            + (eph["omegaDot"] - OMEGA_E_DOT) * tk
            - OMEGA_E_DOT * eph["t_oe"]
        )
        Omega = wrap_0_2pi(Omega + 2 * GPS_PI)

        # 2.11 ECEF 坐标
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        cos_i = math.cos(i)
        sin_i = math.sin(i)

        x = cos_u * r * cos_Omega - sin_u * r * cos_i * sin_Omega
        y = cos_u * r * sin_Omega + sin_u * r * cos_i * cos_Omega
        z = sin_u * r * sin_i

        sat_positions[:, idx] = [x, y, z]

        # ================== 3. 最终钟差 ================== #
        sat_clk_corr[idx] = clk_corr + dtr

    return sat_positions, sat_clk_corr
