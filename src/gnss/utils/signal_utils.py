# placeholder: signal helper functions

"""
Function: check_phase

checkPhase: 检查GPS导航电文数据字的相位/极性

作用：
    根据前一个数据字的最后一个比特(D30*)来检查并校正当前30比特数据字的极性。
    这个过程是 GPS 标准定位服务信号规范 (GPS SPS Signal Specification)
    中定义的一个步骤。

函数用法:
    word = check_phase(word, D30Star)

输入参数:
    word        - 从导航电文中解调出的一个 30 比特长的数据字。
                  (它是一个字符串，必须只包含 '0' 和 '1')。

    D30Star     - 前一个数据字的第 30 个比特 (D30*)。
                  (注意：这是未经奇偶校验的原始比特，所以用星号 * 表示)。
                  它也是一个字符类型 ('0' 或 '1')。

输出参数:
    word        - 数据比特 (前 24 位) 极性被校正后的数据字。
                  (字符串)。
"""

def invert(bits: str) -> str:
    """
    invert 函数：将 '0' → '1'，'1' → '0'
    （此函数对应 MATLAB 的 invert.m）
    """
    inverted = []
    for b in bits:
        if b == '0':
            inverted.append('1')
        elif b == '1':
            inverted.append('0')
        else:
            raise ValueError(f"非法 bit: {b}（必须为 '0' 或 '1'）")
    return ''.join(inverted)


def check_phase(word: str, D30Star: str) -> str:
    """
    检查并修正导航电文字的极性
    """

    # 判断前一个数据字的最后一个比特(D30*)是否为 '1'
    if D30Star == '1':
        # 如果 D30* 为 '1'，根据 GPS 规范，当前数据字的数据比特(前 24 位)必须被反转。
        # 注：'反转' 指的是 '0' 变成 '1'，'1' 变成 '0'。
        #
        # 调用一个名为 invert 的辅助函数来反转 word 的前 24 个比特。
        # 导航电文的后 6 位(25-30)是奇偶校验位，它们不参与反转。

        data_bits = invert(word[:24])
        parity_bits = word[24:]  # 25-30 位保持不变

        word = data_bits + parity_bits

    # 如果 D30Star 不等于 '1'（即为 '0'），则不执行任何操作，
    # 直接返回原始的 word。

    return word


# -----------------------------------------------------------
# 注意：此代码对应 MATLAB 的 helper 函数 invert(bits)
# MATLAB 中其功能大概如下：
#
# function inverted_bits = invert(bits)
#   inverted_bits = bits;
#   inverted_bits(bits == '0') = '1';
#   inverted_bits(bits == '1') = '0';
# end
#
# 本 Python 版本已经在本文件中实现 invert()
# -----------------------------------------------------------

# 如何使用？
""" from gnss.utils.signal_utils import check_phase

w = "010101010101010101010101010101"
D30Star = "1"

new_w = check_phase(w, D30Star)
print(new_w) """






"""
invert(data)
将二进制输入字符串按照逐位异或的方法进行反转：0->1，1->0

此函数对应 MATLAB 的 invert.m，并保留了全部原始注释逻辑。
"""


def invert(data: str) -> str:
    """
    Inverts the binary input-string so that '0' becomes '1' and '1' becomes '0'.

    参数:
        data : 输入的二进制字符串，例如 "10110"

    返回:
        result : 反转后的二进制字符串，例如 "01001"
    """

    # --- 开始分析 ---

    # 1. 获取输入二进制字符串的长度。
    # 例如，如果 data 是 '10110'，那么 dataLength 就是 5。
    dataLength = len(data)

    # 2. 创建一个长度与输入相同、全为 '1' 的字符串作为“掩码 (mask)”。
    # 例如，如果 dataLength 是 5，那么 temp 就是 '11111'。
    temp = "1" * dataLength

    # 3. 将这个全 '1' 的字符串转换为一个十进制整数，作为“反转掩码”。
    # 例如，'11111' → 31
    invertMask = int(temp, 2)

    # 4. 核心步骤：
    #    a. int(data,2) : 将二进制字符串转换为无符号整数
    #    b. ^ invertMask : 与掩码进行按位异或，达到翻转效果
    #    c. format(..., '0{}b'.format(dataLength)) : 转换回固定长度二进制字符
    #
    # 异或示例 (binary):
    #   10110  (22)
    # ^ 11111  (31)
    # --------
    #   01001  (9)
    #
    xor_value = int(data, 2) ^ invertMask

    # 将结果转换回 dataLength 位二进制字符串（不足补0）
    result = format(xor_value, f"0{dataLength}b")

    return result
