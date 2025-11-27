"""
troposphere.py

对流层延迟改正模型函数。

本文件中的函数用于计算伪距/载波相位的对流层路径延迟，
结果应当从观测量中**减去**：
    ρ_corrected = ρ_measured - ddr
"""

from typing import Union


Number = Union[float, int]


def tropo(
    sinel: Number,
    hsta: Number,
    p: Number,
    tkel: Number,
    hum: Number,
    hp: Number,
    htkel: Number,
    hhum: Number,
) -> float:
    """
    计算对流层改正量 ddr（单位：米）。

    对应 MATLAB:
        ddr = tropo(sinel, hsta, p, tkel, hum, hp, htkel, hhum);

    参数说明
    ----------
    sinel : float
        卫星高度角的正弦值，即 sin(E)。如果传入 < 0，会被截断为 0。
    hsta : float
        接收机高度，单位 **km**。
    p : float
        高度 hp 处的大气压强，单位 mbar（mb）。
    tkel : float
        高度 htkel 处的温度，单位 Kelvin。
    hum : float
        高度 hhum 处的相对湿度，单位 %。
    hp : float
        气压观测高度，单位 km。
    htkel : float
        温度观测高度，单位 km。
    hhum : float
        湿度观测高度，单位 km。

    返回
    ----------
    ddr : float
        对流层改正量（单位：米）。
        使用时应从伪距/载波观测中减去该值：
            ρ_corrected = ρ_measured - ddr
    """

    # 地球椭球长半轴（km）
    a_e = 6378.137
    # 与大地曲率相关的常数（原模型中的经验参数）
    b0 = 7.839257e-5
    # 温度直减率 (K/km)，注意这里是负值
    tlapse = -6.5

    # ---- 1. 根据给定高度的温度和湿度，换算到海平面等效参数 -----------------

    # hhum 处温度，使用线性直减率修正（单位：K）
    tkhum = tkel + tlapse * (hhum - htkel)

    # 饱和水汽压计算中的指数项（Magnus 公式形式）
    atkel = 7.5 * (tkhum - 273.15) / (237.3 + tkhum - 273.15)

    # 地面（hhum 高度处）的水汽压 e0（单位：kPa 左右量级）
    # MATLAB: e0 = 0.0611 * hum * 10^atkel;
    e0 = 0.0611 * hum * (10 ** atkel)

    # 把温度外推到海平面温度 tksea（K）
    tksea = tkel - tlapse * htkel

    # 幂指数，用于气压随高度变化的经验模型
    # MATLAB: em = -978.77 / (2.8704e6*tlapse*1.0e-5);
    em = -978.77 / (2.8704e6 * tlapse * 1.0e-5)

    # hhum 处的温度 tkelh（K）
    tkelh = tksea + tlapse * hhum

    # 海平面等效水汽压 e0sea
    # MATLAB: e0sea = e0 * (tksea/tkelh)^(4*em);
    e0sea = e0 * (tksea / tkelh) ** (4 * em)

    # 气压的海平面等效值 psea
    # MATLAB: tkelp = tksea + tlapse*hp; psea = p * (tksea/tkelp)^em;
    tkelp = tksea + tlapse * hp
    psea = p * (tksea / tkelp) ** em

    # ---- 2. 修正 sin(el)，避免负值导致几何发散 ---------------------------

    sinel_val = float(sinel)
    if sinel_val < 0.0:
        sinel_val = 0.0

    # 对流层总改正量（干段 + 湿段），单位：m
    tropo_corr = 0.0

    # ---- 3. 第一段：干分量 (hydrostatic) ---------------------------------

    # 折射率在海平面的基础值
    # MATLAB: refsea = 77.624e-6 / tksea;
    refsea = 77.624e-6 / tksea
    # 顶层高度 htop（km），对流层上界高度的经验公式
    # MATLAB: htop = 1.1385e-5 / refsea;
    htop = 1.1385e-5 / refsea
    # 折射率在海平面的实际值
    refsea = refsea * psea
    # 高度 hsta 处的折射率 ref
    # MATLAB: ref = refsea * ((htop-hsta)/htop)^4;
    ref = refsea * ((htop - hsta) / htop) ** 4

    # 我们用一个两次循环来替代 MATLAB 中的 while + done 标志：
    # 第 0 次迭代：干分量；第 1 次迭代：湿分量
    for step in range(2):
        # rtop = (a_e+htop)^2 - (a_e+hsta)^2*(1-sinel^2);
        rtop = (a_e + htop) ** 2 - (a_e + hsta) ** 2 * (1.0 - sinel_val ** 2)

        # 几何检查：如果出现负值，强行截断为 0
        if rtop < 0.0:
            rtop = 0.0

        # rtop = sqrt(rtop) - (a_e+hsta)*sinel;
        from math import sqrt

        rtop = sqrt(rtop) - (a_e + hsta) * sinel_val

        # a 和 b 是用于近似积分的系数
        a = -sinel_val / (htop - hsta)
        b = -b0 * (1.0 - sinel_val ** 2) / (htop - hsta)

        # rn(i) = rtop^(i+1)，MATLAB i=1:8 → rtop^2 ... rtop^9
        rn = [rtop ** (i + 2) for i in range(8)]

        # alpha 系数向量（前 6 项在所有情况下都有）
        alpha = [
            2 * a,
            2 * a ** 2 + 4 * b / 3.0,
            a * (a ** 2 + 3 * b),
            a ** 4 / 5.0 + 2.4 * a ** 2 * b + 1.2 * b ** 2,
            2 * a * b * (a ** 2 + 3 * b) / 3.0,
            b ** 2 * (6 * a ** 2 + 4 * b) * 1.428571e-1,
            0.0,
            0.0,
        ]

        # 如果 b^2 足够大，则再增加第 7、8 项
        if b * b > 1.0e-35:
            alpha[6] = a * b ** 3 / 2.0
            alpha[7] = b ** 4 / 9.0

        # dr = rtop + alpha * rn （向量点乘）
        dr = rtop
        dr += sum(alpha[i] * rn[i] for i in range(8))

        # tropo 累加该层贡献，ref 单位为“折射率”，乘以 1000 把 km 转换为 m
        tropo_corr += dr * ref * 1000.0

        # 第一次循环结束后，切换到湿分量 (wet term)
        if step == 0:
            # 湿延迟的海平面折射率
            # MATLAB:
            # refsea = (371900.0e-6/tksea-12.92e-6)/tksea;
            refsea = (371900.0e-6 / tksea - 12.92e-6) / tksea
            # 对应的顶层高度
            # MATLAB:
            # htop = 1.1385e-5 * (1255/tksea+0.05)/refsea;
            htop = 1.1385e-5 * (1255.0 / tksea + 0.05) / refsea
            # 对应高度处折射率
            # MATLAB:
            # ref = refsea * e0sea * ((htop-hsta)/htop)^4;
            ref = refsea * e0sea * ((htop - hsta) / htop) ** 4

    # 最终返回对流层改正量（米）
    return tropo_corr
