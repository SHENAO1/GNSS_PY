"""
positioning.py

GNSS 位置解算相关函数：

- least_square_pos: 使用伪距观测进行最小二乘定位 + DOP 计算
"""

from __future__ import annotations

from typing import Tuple, Sequence, Union

import numpy as np
import math

# 依赖的其他模块（你前面已经/正在转换的函数）：
# 地球自转改正：e_r_corr   —— 需要你从对应的 MATLAB e_r_corr.m 转成 Python
# 地平坐标变换：topocent    —— 建议放在 gnss.utils.geo_functions 里
# 对流层改正：tropo        —— 我们之前已经放在 gnss.navigation.troposphere 里
from gnss.navigation.troposphere import tropo
from gnss.utils.geo_functions import topocent
from gnss.navigation.earth_rotation import e_r_corr  # ← 你稍后会转换这个 e_r_corr.m


Number = Union[float, int]


def _get_setting(settings, name: str, default=None):
    """
    小工具函数：同时兼容
    - settings.name 属性形式
    - settings['name'] 字典形式

    如果找不到且没有默认值，则抛出异常。
    """
    if settings is None:
        if default is not None:
            return default
        raise ValueError(f"settings 为 None，且未提供 {name} 默认值。")

    # 1) 对象属性
    if hasattr(settings, name):
        return getattr(settings, name)

    # 2) dict 类型
    if isinstance(settings, dict) and name in settings:
        return settings[name]

    # 3) 大小写/风格稍微兼容一下
    alt_names = {
        "useTropCorr": ["use_trop_corr", "use_tropo_corr"],
    }
    if name in alt_names:
        for alt in alt_names[name]:
            if hasattr(settings, alt):
                return getattr(settings, alt)
            if isinstance(settings, dict) and alt in settings:
                return settings[alt]

    if default is not None:
        return default

    raise AttributeError(f"settings 中未找到字段 {name} 。")


def least_square_pos(
    satpos: np.ndarray,
    obs: Sequence[Number],
    settings,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用最小二乘法，根据伪距观测解算接收机位置、接收机钟差以及 DOP。

    对应 MATLAB:
        [pos, el, az, dop] = leastSquarePos(satpos, obs, settings);

    参数
    ----
    satpos : np.ndarray
        卫星位置矩阵，形状为 (3, N)
        每一列为一颗卫星的 ECEF 坐标 [X; Y; Z]，单位：米。

    obs : Sequence[float]
        对应各卫星的伪距观测值（单位：米），长度为 N。
        例如：[20000000, 21000000, ...]。

    settings : object 或 dict
        接收机设置，至少需要包含：
            - c           : 光速 [m/s]
            - useTropCorr : 是否启用对流层改正 (1 或 0)

        你可以：
        - 在 settings.py 里定义一个 dataclass / 类，字段名与 MATLAB 一致；
        - 或者使用 dict，只要有上述 key 即可。

    返回
    ----
    pos : np.ndarray
        接收机位置和钟差，形状为 (4,)：
            pos[0] = X (ECEF, m)
            pos[1] = Y (ECEF, m)
            pos[2] = Z (ECEF, m)
            pos[3] = dt (接收机钟差，单位：米对应的等效距离)
        注意：这里 dt 是「以距离形式表示的钟差」，如果需要转成秒，可用 dt / c。

    el : np.ndarray
        各卫星的高度角（单位：度），长度 N。

    az : np.ndarray
        各卫星的方位角（单位：度），长度 N。

    dop : np.ndarray
        DOP 向量，长度 5，对应：
            dop[0] = GDOP
            dop[1] = PDOP
            dop[2] = HDOP
            dop[3] = VDOP
            dop[4] = TDOP
    """

    # ------------------------ 初始化 ------------------------ #
    nmb_of_iterations = 7            # 外层迭代次数
    dtr = math.pi / 180.0           # 度 → 弧度

    # 卫星数
    satpos = np.asarray(satpos, dtype=float)
    if satpos.shape[0] != 3:
        raise ValueError("satpos 的形状应为 (3, N)，即每列为一颗卫星的 [X; Y; Z].")

    n_sats = satpos.shape[1]

    # 伪距观测
    obs = np.asarray(obs, dtype=float)
    if obs.size != n_sats:
        raise ValueError("obs 的长度必须与卫星数量一致。")

    # 接收机位置 [X, Y, Z, dt] 初始值（全 0）
    pos = np.zeros(4, dtype=float)

    # 结果数组
    A = np.zeros((n_sats, 4), dtype=float)
    omc = np.zeros(n_sats, dtype=float)      # observed - computed
    az = np.zeros(n_sats, dtype=float)       # 方位角 [deg]
    el = np.zeros(n_sats, dtype=float)       # 高度角 [deg]

    # 从 settings 中取光速 c 和对流层开关 useTropCorr
    c = float(_get_setting(settings, "c"))
    use_trop = int(_get_setting(settings, "useTropCorr", default=0))

    # ------------------------ 迭代求解位置 ------------------------ #
    for iter_idx in range(nmb_of_iterations):

        for i in range(n_sats):
            sat_xyz = satpos[:, i]

            if iter_idx == 0:
                # 第一次迭代：先不考虑地球自转、对流层等
                Rot_X = sat_xyz.copy()
                trop = 2.0  # SoftGNSS 里用 2m 作为一个粗略的对流层初始值
            else:
                # 计算几何距离平方（使用上一轮 pos）
                dx = sat_xyz[0] - pos[0]
                dy = sat_xyz[1] - pos[1]
                dz = sat_xyz[2] - pos[2]
                rho2 = dx * dx + dy * dy + dz * dz

                # 信号传播时间
                traveltime = math.sqrt(rho2) / c

                # 地球自转改正（Sagnac Effect）
                # Rot_X 为“发射时刻”在 ECEF 中的卫星位置
                Rot_X = e_r_corr(traveltime, sat_xyz)

                # 由接收机位置 → 该卫星的地平坐标，得到方位角和高度角
                # topocent(pos_xyz, sat_xyz_minus_rec_xyz)
                rec_xyz = pos[:3]
                az_i, el_i, dist = topocent(rec_xyz, Rot_X - rec_xyz)

                az[i] = az_i   # 单位：度
                el[i] = el_i   # 单位：度

                # 对流层改正
                if use_trop == 1:
                    trop = tropo(
                        math.sin(el[i] * dtr),  # sin(高度角)
                        0.0,   # hsta (km)
                        1013.0,  # p (mbar)
                        293.0,   # tkel (K)
                        50.0,    # hum (%)
                        0.0, 0.0, 0.0,  # hp, htkel, hhum
                    )
                else:
                    trop = 0.0

            # 观测减计算：obs - ρ - 钟差 - 对流层
            geometric_range = np.linalg.norm(Rot_X - pos[:3], ord=2)
            omc[i] = obs[i] - geometric_range - pos[3] - trop

            # 构造 A 矩阵的第 i 行
            # 注意这里用 obs(i) 作为分母（与 SoftGNSS 原版保持一致）
            A[i, 0] = -(Rot_X[0] - pos[0]) / obs[i]
            A[i, 1] = -(Rot_X[1] - pos[1]) / obs[i]
            A[i, 2] = -(Rot_X[2] - pos[2]) / obs[i]
            A[i, 3] = 1.0

        # 如果 A 矩阵秩不足 4，则定位不可解
        if np.linalg.matrix_rank(A) != 4:
            # 与 MATLAB 一致：返回 pos = [0 0 0 0]
            return np.zeros(4, dtype=float), el, az, np.zeros(5, dtype=float)

        # 解最小二乘：x = (A\omc)
        # numpy 等价：x = np.linalg.lstsq(A, omc, rcond=None)[0]
        x, *_ = np.linalg.lstsq(A, omc, rcond=None)

        # 更新位置估计 pos = pos + x
        pos += x

    # 循环结束后 pos 为最终估计，保持为 shape=(4,)
    # 与 MATLAB 最后 pos = pos' 形成 1x4 行向量是一致的（只不过 Python 中是 1D 向量）

    # ------------------------ 计算 DOP ------------------------ #
    # Q = (A^T * A)^(-1)
    Q = np.linalg.inv(A.T @ A)

    dop = np.zeros(5, dtype=float)
    dop[0] = math.sqrt(np.trace(Q))                  # GDOP
    dop[1] = math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])  # PDOP
    dop[2] = math.sqrt(Q[0, 0] + Q[1, 1])           # HDOP
    dop[3] = math.sqrt(Q[2, 2])                     # VDOP
    dop[4] = math.sqrt(Q[3, 3])                     # TDOP

    return pos, el, az, dop
