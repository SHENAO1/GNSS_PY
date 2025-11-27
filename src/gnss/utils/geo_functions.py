"""
geo_functions.py

地理/角度/坐标变换相关的小工具函数集合。

包含：
- dms2mat   : 打包 DMS 实数 → [度, 分, 秒] 向量
- deg2dms   : 十进制度 → 打包 DMS 实数
- clsin     : Clenshaw 正弦级数（实参数）
- clksin    : Clenshaw 正弦级数（复参数）
- geo2cart  : 大地坐标 → ECEF 笛卡尔坐标
- cart2geo  : ECEF 笛卡尔坐标 → 大地坐标（按椭球类型）
- togeod    : 通用 ECEF → 大地坐标（按 a, 1/f）
- find_utm_zone : 根据经纬度计算 UTM 分区编号
- cart2utm  : ECEF → ED50 → UTM (E,N,U)
- topocent  : ECEF → 局部 ENU，并计算方位角/高度角/距离
"""

import math
from typing import Sequence, Tuple

import numpy as np


# ============================================================================
# 1. 角度格式转换
# ============================================================================

def dms2mat(dms_input: float, n: int) -> Tuple[float, float, float]:
    """
    将实数 a = dd*100 + mm + s/100 形式的角度，拆分为 [dd, mm, ss] 形式。

    对应 MATLAB: matOutput = dms2mat(dmsInput, n)

    参数
    ----
    dms_input : float
        形如 dd*100 + mm + s/100 的实数。
        例如： 12120.1234 表示 121度20分12.34秒（具体按你实际数据）。
        函数内部会自动处理正负号。
    n : int
        小数秒保留精度，表示 10 的幂（需要为负数）。
        比如：
        - n = -3 → 保留到 10^-3 秒（毫秒），即 ss.xxx
        - n = -2 → 保留到 10^-2 秒（厘秒）

        注意：本函数按原 MATLAB 逻辑实现，实际上是“截断”到
        指定小数位，而不是严格数学意义上的“四舍五入”。

    返回
    ----
    (deg, minute, second) : Tuple[float, float, float]
        deg    : 整数度（如果输入为负，会体现在 deg 上）
        minute : 整数分（0~59）
        second : 秒，保留到指定精度
    """

    # 记录是否为负号，原 MATLAB 中只对正数进行度分秒拆分
    neg_arg = False
    if dms_input < 0:
        dms_input = -dms_input
        neg_arg = True

    # --- 拆分度、分、秒（与 MATLAB 对齐） -------------------------
    int_deg = int(dms_input // 100)        # floor(dmsInput/100)
    mm = int(dms_input - 100 * int_deg)    # floor(dmsInput - 100*int_deg)

    # 秒的浮点数，然后格式化为 10 位小数的字符串
    seconds_float = (dms_input - 100 * int_deg - mm) * 100.0
    ssdec_str = f"{seconds_float:.10f}"    # 对应 MATLAB 的 '%2.10f'
    ssdec_val = float(ssdec_str)

    # --- 秒溢出检查：秒到了 60 的情况 -----------------------------
    if abs(ssdec_val - 60.0) < 1e-8:
        mm += 1
        ssdec_str = "0.0000000000"
        ssdec_val = 0.0

    # 分溢出：60 分进 1 度
    if mm == 60:
        int_deg += 1
        mm = 0

    # 负号只加在“度”上
    if neg_arg:
        int_deg = -int_deg

    # --- 根据 n 控制秒的小数位数（保持与 MATLAB 相同的截断逻辑） ---
    if n >= 0:
        raise ValueError("dms2mat: 参数 n 应为负数（例如 n=-3 表示保留毫秒）。")

    # MATLAB: matOutput(3) = str2double(ssdec(1:-n+3));
    end_idx = -n + 3
    if end_idx > len(ssdec_str):
        end_idx = len(ssdec_str)

    ss_truncated_str = ssdec_str[:end_idx]
    second = float(ss_truncated_str)

    return float(int_deg), float(mm), second


def deg2dms(deg: float) -> float:
    """
    将角度（十进制度）转换为 DMS 打包格式：
        dms = degrees*100 + minutes + seconds/100

    对应 MATLAB: dmsOutput = deg2dms(deg)

    参数
    ----
    deg : float
        十进制度（可以为负）

    返回
    ----
    dmsOutput : float
        形如 dd*100 + mm + ss/100 的“打包 DMS 格式”

    例如：
        12.345° → 12度20分42秒
        输出为：12*100 + 20 + 42/100 = 1220.42
    """

    neg_arg = False
    if deg < 0:
        deg = -deg
        neg_arg = True

    int_deg = int(deg)
    decimal = deg - int_deg

    min_part = decimal * 60.0
    minute = int(min_part)

    sec_part = min_part - minute
    second = sec_part * 60.0

    # 秒溢出
    if abs(second - 60.0) < 1e-12:
        minute += 1
        second = 0.0

    # 分溢出
    if minute == 60:
        int_deg += 1
        minute = 0

    dms_output = int_deg * 100 + minute + second / 100.0

    if neg_arg:
        dms_output = -dms_output

    return dms_output


# ============================================================================
# 2. Clenshaw 级数：clsin / clksin
# ============================================================================

def clsin(ar: Sequence[float], degree: int, argument: float) -> float:
    """
    Clenshaw 求和（Clenshaw Summation），用于计算三角级数：
        result = Σ ar[k] * sin(k * argument)

    对应 MATLAB:
        result = clsin(ar, degree, argument)

    参数
    ----
    ar : Sequence[float]
        系数数组（MATLAB 的 ar(t)）
    degree : int
        最高阶数（例如 4）
    argument : float
        自变量（弧度）

    返回
    ----
    result : float
        级数求和结果
    """

    cos_arg = 2.0 * math.cos(argument)
    hr1 = 0.0
    hr = 0.0

    # Clenshaw 反向递推
    for t in range(degree, 0, -1):
        hr2 = hr1
        hr1 = hr
        hr = ar[t - 1] + cos_arg * hr1 - hr2

    return hr * math.sin(argument)


def clksin(ar: Sequence[float], degree: int,
           arg_real: float, arg_imag: float) -> Tuple[float, float]:
    """
    Clenshaw 求和算法（Clenshaw Summation），
    用于计算“复参数正弦级数”，SoftGNSS 中用于 Krüger 投影的 dN/dE 修正。

    对应 MATLAB：
        [re, im] = clksin(ar, degree, arg_real, arg_imag);

    参数：
    ----
    ar : 系数数组（长度至少为 degree）
         对应 MATLAB 的 ar(t)，即 a_1, a_2, ..., a_degree
    degree : int
         递推阶数（通常为 4）
    arg_real : float
         实部（例如 Np）
    arg_imag : float
         虚部（例如 Ep）

    返回：
    ----
    re, im : float, float
        级数求和结果的实部和虚部
    """

    sin_arg_r = math.sin(arg_real)
    cos_arg_r = math.cos(arg_real)
    sinh_arg_i = math.sinh(arg_imag)
    cosh_arg_i = math.cosh(arg_imag)

    # 复参数 Clenshaw 系数
    r = 2.0 * cos_arg_r * cosh_arg_i
    i = -2.0 * sin_arg_r * sinh_arg_i

    hr1 = 0.0
    hr = 0.0
    hi1 = 0.0
    hi = 0.0

    # Clenshaw 递推
    for t in range(degree, 0, -1):
        hr2 = hr1
        hr1 = hr
        hi2 = hi1
        hi1 = hi

        z = ar[t - 1] + r * hr1 - i * hi - hr2
        hi = i * hr1 + r * hi1 - hi2
        hr = z

    r2 = sin_arg_r * cosh_arg_i
    i2 = cos_arg_r * sinh_arg_i

    re = r2 * hr - i2 * hi
    im = r2 * hi + i2 * hr

    return re, im


# ============================================================================
# 3. 椭球 / 大地坐标与 ECEF 转换
# ============================================================================

def geo2cart(
    phi_dms: Sequence[float],
    lambda_dms: Sequence[float],
    h: float,
    i: int,
) -> Tuple[float, float, float]:
    """
    geo2cart: 将大地坐标 (φ, λ, h) 转换为笛卡尔坐标 (X, Y, Z).

    对应 MATLAB:
        [X, Y, Z] = geo2cart(phi, lambda, h, i);

    其中：
    - phi      : 纬度，格式为 [度 分 秒]（北纬为正，南纬为负）
    - lambda   : 经度，格式为 [度 分 秒]（东经为正，西经为负）
    - h        : 椭球高，单位：米
    - i        : 参考椭球类型（1~5）
                  1. International Ellipsoid 1924
                  2. International Ellipsoid 1967
                  3. World Geodetic System 1972
                  4. Geodetic Reference System 1980
                  5. World Geodetic System 1984 (WGS84, GNSS 常用)

    返回
    ----
    X, Y, Z : float
        笛卡尔坐标（ECEF），单位：米
    """

    if len(phi_dms) != 3 or len(lambda_dms) != 3:
        raise ValueError("phi_dms 和 lambda_dms 必须是长度为 3 的 [度, 分, 秒] 序列。")

    if i < 1 or i > 5:
        raise ValueError("参数 i 椭球类型必须在 1~5 之间。")

    # DMS → 十进制度 → 弧度
    phi_deg = phi_dms[0] + phi_dms[1] / 60.0 + phi_dms[2] / 3600.0
    lam_deg = lambda_dms[0] + lambda_dms[1] / 60.0 + lambda_dms[2] / 3600.0

    b = phi_deg * math.pi / 180.0
    l = lam_deg * math.pi / 180.0

    # 椭球参数
    a_list = [
        6378388.0,
        6378160.0,
        6378135.0,
        6378137.0,
        6378137.0,
    ]
    f_list = [
        1.0 / 297.0,
        1.0 / 298.247,
        1.0 / 298.26,
        1.0 / 298.257222101,
        1.0 / 298.257223563,
    ]

    a = a_list[i - 1]
    f = f_list[i - 1]

    ex2 = (2.0 - f) * f / ((1.0 - f) ** 2)
    c = a * math.sqrt(1.0 + ex2)
    cosb = math.cos(b)
    N = c / math.sqrt(1.0 + ex2 * cosb * cosb)

    cosl = math.cos(l)
    sinl = math.sin(l)
    sinb = math.sin(b)

    X = (N + h) * cosb * cosl
    Y = (N + h) * cosb * sinl
    Z = (((1.0 - f) ** 2) * N + h) * sinb

    return X, Y, Z


def cart2geo(X: float, Y: float, Z: float, i: int) -> Tuple[float, float, float]:
    """
    将笛卡尔坐标 (X, Y, Z) 转换为给定参考椭球上的大地坐标 (phi, lambda, h)。

    对应 MATLAB:
        [phi, lambda, h] = cart2geo(X, Y, Z, i);

    椭球类型 i：
        1. International Ellipsoid 1924
        2. International Ellipsoid 1967
        3. World Geodetic System 1972
        4. Geodetic Reference System 1980
        5. World Geodetic System 1984 (WGS84)

    返回
    ----
    phi : float
        大地纬度，单位：度
    lambda_ : float
        大地经度，单位：度
    h : float
        椭球高，单位：米
    """

    if i < 1 or i > 5:
        raise ValueError("椭球类型 i 必须在 1~5 之间。")

    a_list = [6378388.0, 6378160.0, 6378135.0, 6378137.0, 6378137.0]
    f_list = [
        1.0 / 297.0,
        1.0 / 298.247,
        1.0 / 298.26,
        1.0 / 298.257222101,
        1.0 / 298.257223563,
    ]

    a = a_list[i - 1]
    f = f_list[i - 1]

    lambda_rad = math.atan2(Y, X)

    ex2 = (2.0 - f) * f / ((1.0 - f) ** 2)
    c = a * math.sqrt(1.0 + ex2)

    r_xy = math.hypot(X, Y)
    phi = math.atan(Z / (r_xy * (1.0 - (2.0 - f) * f)))

    h = 0.1
    old_h = 0.0
    iterations = 0

    while abs(h - old_h) > 1.0e-12:
        old_h = h

        cos_phi = math.cos(phi)
        N = c / math.sqrt(1.0 + ex2 * cos_phi * cos_phi)

        denom = r_xy * (1.0 - (2.0 - f) * f * N / (N + h))
        phi = math.atan(Z / denom)

        h = r_xy / math.cos(phi) - N

        iterations += 1
        if iterations > 100:
            print(
                f"Failed to approximate h with desired precision. h-oldh: {h - old_h:.6e}"
            )
            break

    phi_deg = phi * 180.0 / math.pi
    lambda_deg = lambda_rad * 180.0 / math.pi

    return phi_deg, lambda_deg, h


def togeod(a: float, finv: float,
           X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """
    TOGEOD: ECEF 笛卡尔坐标 (X, Y, Z) → 大地坐标 (纬度, 经度, 高程)

    对应 MATLAB:
        [dphi, dlambda, h] = togeod(a, finv, X, Y, Z);

    参数
    ----
    a : float
        参考椭球的长半轴 (semi-major axis)
    finv : float
        椭球扁率的倒数 (inverse flattening)，例如 WGS84 为 298.257223563
    X, Y, Z : float
        ECEF 笛卡尔坐标

    返回
    ----
    dphi : float
        大地纬度（十进制度）
    dlambda : float
        大地经度（十进制度）
    h : float
        椭球高，单位与 X/Y/Z/a 一致
    """

    h = 0.0
    tolsq = 1.0e-10
    maxit = 10

    rtd = 180.0 / math.pi

    if finv < 1.0e-20:
        esq = 0.0
    else:
        esq = (2.0 - 1.0 / finv) / finv

    oneesq = 1.0 - esq

    P = math.hypot(X, Y)

    if P > 1.0e-20:
        dlambda = math.atan2(Y, X) * rtd
    else:
        dlambda = 0.0

    if dlambda < 0.0:
        dlambda += 360.0

    r = math.hypot(P, Z)

    if r > 1.0e-20:
        sinphi = Z / r
    else:
        sinphi = 0.0

    dphi = math.asin(sinphi)

    if r < 1.0e-20:
        h = 0.0
        dphi_deg = dphi * rtd
        return dphi_deg, dlambda, h

    h = r - a * (1.0 - sinphi * sinphi / finv)

    for i in range(maxit):
        sinphi = math.sin(dphi)
        cosphi = math.cos(dphi)

        N_phi = a / math.sqrt(1.0 - esq * sinphi * sinphi)

        dP = P - (N_phi + h) * cosphi
        dZ = Z - (N_phi * oneesq + h) * sinphi

        h = h + (sinphi * dZ + cosphi * dP)
        dphi = dphi + (cosphi * dZ - sinphi * dP) / (N_phi + h)

        if dP * dP + dZ * dZ < tolsq:
            break

        if i == maxit - 1:
            print(
                f"Warning: TOGEOD did not fully converge in {maxit} iterations. "
                f"Residual dP^2 + dZ^2 = {dP * dP + dZ * dZ:.3e}"
            )

    dphi_deg = dphi * rtd

    return dphi_deg, dlambda, h


# ============================================================================
# 4. UTM 区带与 UTM 投影
# ============================================================================

def find_utm_zone(latitude: float, longitude: float) -> int:
    """
    根据给定的纬度(latitude)和经度(longitude)计算所在的 UTM 分区（Zone 编号）。

    对应 MATLAB:
        utmZone = findUtmZone(latitude, longitude)

    参数
    ----
    latitude : float
        纬度（十进制度），范围必须为 [-80, 84]
    longitude : float
        经度（十进制度），范围必须为 [-180, 180]

    返回
    ----
    utmZone : int
        UTM 分区编号（1~60）
    """

    if longitude < -180 or longitude > 180:
        raise ValueError("Longitude must be within [-180, 180] degrees.")

    if latitude < -80 or latitude > 84:
        raise ValueError("Latitude must be within [-80, 84] degrees.")

    utm_zone = int((longitude + 180) // 6) + 1

    if latitude > 72:
        if 0 <= longitude < 9:
            utm_zone = 31
        elif 9 <= longitude < 21:
            utm_zone = 33
        elif 21 <= longitude < 33:
            utm_zone = 35
        elif 33 <= longitude < 42:
            utm_zone = 37
    elif 56 <= latitude < 64:
        if 3 <= longitude < 12:
            utm_zone = 32

    return utm_zone


def cart2utm(X: float, Y: float, Z: float, zone: int) -> Tuple[float, float, float]:
    """
    将三维笛卡尔坐标 (X, Y, Z)（ITRF96 / WGS84 近似）转换为
    指定 UTM 带号的 (E, N, U)：

    - E: Easting (米)
    - N: Northing (米)
    - U: Uping（大致相当于高程，单位米）

    完全对应 MATLAB 版 cart2utm.m，包含：
    1. ITRF96 → ED50 的七参数近似转换（缩放 + 平移 + 小角度旋转）
    2. 基于 Krüger 展开式的 UTM 投影（f = 1/297, a = 6378388）
    3. 南北半球的处理（B<0 时做 20000000 的偏移）
    """

    # ----- 1. ITRF96 → ED50 近似转换 -----
    a = 6378388.0
    f = 1.0 / 297.0

    ex2 = (2 - f) * f / ((1 - f) ** 2)
    c = a * math.sqrt(1 + ex2)

    vec_x = X
    vec_y = Y
    vec_z = Z - 4.5

    alpha = 0.756e-6
    R11, R12 = 1.0, -alpha
    R21, R22 = alpha, 1.0
    tx, ty, tz = 89.5, 93.8, 127.6
    scale = 0.9999988

    vx = scale * (R11 * vec_x + R12 * vec_y) + tx
    vy = scale * (R21 * vec_x + R22 * vec_y) + ty
    vz = scale * vec_z + tz

    L = math.atan2(vy, vx)

    N1 = 6395000.0
    B = math.atan2(vz / ((1 - f) ** 2 * N1), math.hypot(vx, vy) / N1)

    U = 0.1
    oldU = 0.0
    iterations = 0

    while abs(U - oldU) > 1e-4:
        oldU = U
        N1 = c / math.sqrt(1 + ex2 * (math.cos(B) ** 2))

        B = math.atan2(
            vz / ((1 - f) ** 2 * N1 + U),
            math.hypot(vx, vy) / (N1 + U),
        )

        U = math.hypot(vx, vy) / math.cos(B) - N1

        iterations += 1
        if iterations > 100:
            print(
                f"Failed to approximate U with desired precision. "
                f"U-oldU: {U - oldU:.6e}"
            )
            break

    # ----- 2. Krüger UTM 投影参数 -----
    m0 = 0.0004
    n = f / (2 - f)
    m = n ** 2 * (1 / 4 + n ** 2 / 64)
    w = (a * (-n - m0 + m * (1 - m0))) / (1 + n)
    Q_n = a + w

    E0 = 500000.0
    L0_deg = (zone - 30) * 6 - 3
    L0 = L0_deg * math.pi / 180.0

    # Krüger 系数（f = 1/297 时预计算）
    bg = [
        -3.37077907e-3,
        4.73444769e-6,
        -8.29914570e-9,
        1.58785330e-11,
    ]

    gb = [
        3.37077588e-3,
        6.62769080e-6,
        1.78718601e-8,
        5.49266312e-11,
    ]

    gtu = [
        8.41275991e-4,
        7.67306686e-7,
        1.21291230e-9,
        2.48508228e-12,
    ]

    utg = [
        -8.41276339e-4,
        -5.95619298e-8,
        -1.69485209e-10,
        -2.20473896e-13,
    ]

    # ----- 3. 椭球地理坐标 → 球面地理坐标 -----
    neg_geo = B < 0
    Bg_r = abs(B)

    # 椭球纬度 → 球面纬度：Bg_r = Bg_r + clsin(bg, 4, 2*Bg_r)
    Bg_r = Bg_r + clsin(bg, 4, 2 * Bg_r)

    Lg_r = L - L0
    cos_BN = math.cos(Bg_r)

    Np = math.atan2(math.sin(Bg_r), math.cos(Lg_r) * cos_BN)
    Ep = math.atanh(math.sin(Lg_r) * cos_BN)

    # ----- 4. 球面 N,E → 椭球 N,E 修正（Clenshaw + 2*Np,2*Ep） -----
    Np2 = 2.0 * Np
    Ep2 = 2.0 * Ep

    dN, dE = clksin(gtu, 4, Np2, Ep2)

    # 标准 Krüger 实现中，dN,dE 本身就是修正量，直接相加
    Np = Np + dN
    Ep = Ep + dE

    N = Q_n * Np
    E = Q_n * Ep + E0

    if neg_geo:
        N = -N + 20000000.0

    return E, N, U


# ============================================================================
# 5. ECEF → 局部 ENU（Topocentric）
# ============================================================================

def topocent(X: np.ndarray, dx: np.ndarray) -> Tuple[float, float, float]:
    """
    将矢量 dx 转换到接收机局部坐标系（Topocentric ENU）下，
    并计算方位角 Az、高度角 El 以及距离 D。

    对应 MATLAB：
        [Az, El, D] = topocent(X, dx)

    参数
    ----
    X : ndarray, shape (3,)
        接收机在 ECEF 下的坐标 [X, Y, Z] (m)
    dx : ndarray, shape (3,)
        从接收机指向卫星的矢量 Δ[X, Y, Z] (m)

    返回
    ----
    Az : float
        方位角（度），从北方向顺时针 0~360°
    El : float
        高度角（度）
    D : float
        距离（m）
    """

    dtr = math.pi / 180.0

    # WGS84 椭球参数
    a = 6378137.0
    f = 1.0 / 298.257223563

    phi, lam, h = togeod(a, f, X[0], X[1], X[2])

    lam_rad = lam * dtr
    phi_rad = phi * dtr

    cl = math.cos(lam_rad)
    sl = math.sin(lam_rad)
    cb = math.cos(phi_rad)
    sb = math.sin(phi_rad)

    F = np.array([
        [-sl,       -sb * cl,   cb * cl],
        [cl,        -sb * sl,   cb * sl],
        [0.0,           cb,         sb],
    ])

    local_vector = F.T @ dx

    E = float(local_vector[0])
    N = float(local_vector[1])
    U = float(local_vector[2])

    hor_dis = math.hypot(E, N)

    if hor_dis < 1e-20:
        Az = 0.0
        El = 90.0
    else:
        Az = math.degrees(math.atan2(E, N))
        El = math.degrees(math.atan2(U, hor_dis))

    if Az < 0.0:
        Az += 360.0

    D = float(math.sqrt(dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2))

    return Az, El, D
