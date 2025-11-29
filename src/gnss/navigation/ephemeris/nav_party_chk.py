# src/gnss/navigation/ephemeris/nav_party_chk.py

import numpy as np

def nav_party_chk(ndat: np.ndarray) -> int:
    """
    GPS 导航电文奇偶校验 (Parity Check)。
    
    参数:
    ndat: 长度为 32 的数组 (0/1)，包含：
          - [0:2] : 前一个字的最后两比特 (D29*, D30*)
          - [2:32]: 当前字的 30 比特 (其中最后 6 比特是接收到的奇偶校验位)
    
    返回:
    status: 0 表示校验通过，-1 表示失败。
    """
    ndat = np.array(ndat, dtype=int).flatten()
    
    if len(ndat) != 32:
        return -1

    # 提取 D29*, D30* (来自上一个字)
    d29_star = ndat[0]
    d30_star = ndat[1]
    
    # 当前字的 24 数据位 (d1 ... d24) 对应索引 ndat[2] ... ndat[25]
    # 接收到的 6 校验位 (PB1 ... PB6) 对应索引 ndat[26] ... ndat[31]
    d = ndat[2:26]  # 数据位
    pb_received = ndat[26:32] # 接收到的校验位

    # 根据 IS-GPS-200 表 20-XIV 计算校验位
    # 为了简化计算，定义 parity 位的计算掩码 (源自 IS-GPS-200)
    
    pb_calc = np.zeros(6, dtype=int)
    
    # 这里的异或运算 (.) 是模2加法
    
    # PB1 (D25)
    idx_1 = [0,1,2,4,5,9,10,11,12,13,16,17,19,22] # 对应 D1..D23
    sum_1 = d29_star + np.sum(d[idx_1])
    pb_calc[0] = sum_1 % 2

    # PB2 (D26)
    idx_2 = [1,2,3,5,6,10,11,12,13,14,17,18,20,23]
    sum_2 = d30_star + np.sum(d[idx_2])
    pb_calc[1] = sum_2 % 2

    # PB3 (D27)
    idx_3 = [0,2,3,4,6,7,11,12,13,14,15,18,19,21]
    sum_3 = d29_star + np.sum(d[idx_3])
    pb_calc[2] = sum_3 % 2

    # PB4 (D28)
    idx_4 = [1,3,4,5,7,8,12,13,14,15,16,19,20,22]
    sum_4 = d30_star + np.sum(d[idx_4])
    pb_calc[3] = sum_4 % 2

    # PB5 (D29)
    idx_5 = [0,2,4,5,6,8,9,13,14,15,16,17,20,21,23]
    sum_5 = d30_star + np.sum(d[idx_5])
    pb_calc[4] = sum_5 % 2

    # PB6 (D30)
    idx_6 = [2,4,6,7,8,9,10,12,14,18,21,22,23]
    sum_6 = d29_star + np.sum(d[idx_6])
    pb_calc[5] = sum_6 % 2

    # 比较计算出的校验位和接收到的校验位
    if np.array_equal(pb_calc, pb_received):
        return 0
    else:
        return -1


def check_t(time: float) -> float:
    """
    处理 GPS 时间的周内跳变（week crossover）。
    防止计算卫星位置时，传输时间与星历参考时间(toe)跨越了周界（+/- 302400秒）。
    """
    half_week = 302400.0
    corr_time = time

    if time > half_week:
        corr_time = time - 2.0 * half_week
    elif time < -half_week:
        corr_time = time + 2.0 * half_week

    return corr_time