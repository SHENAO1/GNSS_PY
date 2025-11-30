# src/gnss/navigation/ephemeris/nav_party_chk.py

import numpy as np


def nav_party_chk(ndat) -> int:
    """
    Python 版 navPartyChk，对应 MATLAB:

        status = navPartyChk(ndat)

    输入:
        ndat : 长度为 32 的数组，可为 0/1 或 ±1。
               代表一个 GPS 导航字:
               ndat[0]  -> D29*
               ndat[1]  -> D30*
               ndat[2:26]  -> d1..d24
               ndat[26:32] -> D25..D30 (接收的奇偶校验位)

    输出:
        status :  +1 或 -1 表示奇偶校验通过
                  0 表示奇偶校验失败

        注意: +1/-1 的符号用于指示当前字前 24bit 是否需要反相，
             完全沿用 MATLAB 语义。
    """

    ndat = np.asarray(ndat, dtype=int).flatten()
    if ndat.size != 32:
        raise ValueError(f"nav_party_chk 期望 32 比特，实际 {ndat.size}")

    # ---- 将输入统一映射到 ±1 表示 ----
    # MATLAB 里 0 -> -1, 1 -> +1
    ndat_pm = np.where(ndat == 0, -1, ndat)     # 若之前已是 ±1 不受影响
    ndat_pm = np.where(ndat_pm == -1, -1, 1)    # 防止出现除了 0/1/-1 以外的值

    # --- Check if the data bits must be inverted --------------------------
    # MATLAB: if (ndat(2) ~= 1) ndat(3:26) = -1 .* ndat(3:26);
    # Python 索引对应: ndat_pm[1] != 1 时，反转 2:26
    if ndat_pm[1] != 1:
        ndat_pm[2:26] = -ndat_pm[2:26]

    # --- Calculate 6 parity bits -----------------------------------------
    # 下面完全按 MATLAB 索引 (减 1) 转成 Python 写法

    parity = np.zeros(6, dtype=int)

    # parity(1)
    parity[0] = (
        ndat_pm[0] * ndat_pm[2] * ndat_pm[3] * ndat_pm[4] * ndat_pm[6] *
        ndat_pm[7] * ndat_pm[11] * ndat_pm[12] * ndat_pm[13] * ndat_pm[14] *
        ndat_pm[15] * ndat_pm[18] * ndat_pm[19] * ndat_pm[21] * ndat_pm[24]
    )

    # parity(2)
    parity[1] = (
        ndat_pm[1] * ndat_pm[3] * ndat_pm[4] * ndat_pm[5] * ndat_pm[7] *
        ndat_pm[8] * ndat_pm[12] * ndat_pm[13] * ndat_pm[14] * ndat_pm[15] *
        ndat_pm[16] * ndat_pm[19] * ndat_pm[20] * ndat_pm[22] * ndat_pm[25]
    )

    # parity(3)
    parity[2] = (
        ndat_pm[0] * ndat_pm[2] * ndat_pm[4] * ndat_pm[5] * ndat_pm[6] *
        ndat_pm[8] * ndat_pm[9] * ndat_pm[13] * ndat_pm[14] * ndat_pm[15] *
        ndat_pm[16] * ndat_pm[17] * ndat_pm[20] * ndat_pm[21] * ndat_pm[23]
    )

    # parity(4)
    parity[3] = (
        ndat_pm[1] * ndat_pm[3] * ndat_pm[5] * ndat_pm[6] * ndat_pm[7] *
        ndat_pm[9] * ndat_pm[10] * ndat_pm[14] * ndat_pm[15] * ndat_pm[16] *
        ndat_pm[17] * ndat_pm[18] * ndat_pm[21] * ndat_pm[22] * ndat_pm[24]
    )

    # parity(5)
    parity[4] = (
        ndat_pm[1] * ndat_pm[2] * ndat_pm[4] * ndat_pm[6] * ndat_pm[7] *
        ndat_pm[8] * ndat_pm[10] * ndat_pm[11] * ndat_pm[15] * ndat_pm[16] *
        ndat_pm[17] * ndat_pm[18] * ndat_pm[19] * ndat_pm[22] * ndat_pm[23] *
        ndat_pm[25]
    )

    # parity(6)
    parity[5] = (
        ndat_pm[0] * ndat_pm[4] * ndat_pm[6] * ndat_pm[7] * ndat_pm[9] *
        ndat_pm[10] * ndat_pm[11] * ndat_pm[12] * ndat_pm[14] * ndat_pm[16] *
        ndat_pm[20] * ndat_pm[23] * ndat_pm[24] * ndat_pm[25]
    )

    # --- Compare received parity with calculated parity ------------------
    # MATLAB: if ((sum(parity == ndat(27:32))) == 6)
    # 其中 ndat(27:32) 是接收的 D25..D30，同样是 ±1 表示
    recv_parity = ndat_pm[26:32]

    if np.all(parity == recv_parity):
        # Parity OK: 输出 -1 * ndat(2) (D30*)
        status = int(-ndat_pm[1])
    else:
        status = 0

    return status
