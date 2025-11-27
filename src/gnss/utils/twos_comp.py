"""
twosComp2dec(binaryNumber)
将一个二进制补码字符串转换为十进制整数。

输入的 binaryNumber 必须是一个只包含 '0' 和 '1' 的字符串，
表示一个二进制补码形式的整数。

语法:
    intNumber = twos_comp2dec(binaryNumber)

输出:
    intNumber: 转换后的十进制整数。
"""

def twos_comp2dec(binaryNumber: str) -> int:
    """
    TWOSCOMP2DEC: 将二进制补码字符串转换为有符号十进制整数。

    参数:
        binaryNumber : 字符串，例如 '1011'，应只包含 '0' 或 '1'

    返回:
        intNumber : 以二补码解释后的整数
    """

    # --- 检查输入是否为字符串 -------------------------------------------------
    # MATLAB 中使用 ischar/isstring；Python 中使用 isinstance(str)
    if not isinstance(binaryNumber, str):
        raise TypeError("输入必须是字符串 (Input must be a string).")

    # --- 将二进制字符串转换为无符号十进制数 ---------------------------------
    # int(binaryNumber, 2) 始终按无符号解释
    # 例如 int('1111', 2) == 15
    intNumber = int(binaryNumber, 2)

    # --- 如果这是一个负数（即最高位为 '1'），则进行修正 ----------------------
    # 在二进制补码表示法中，最高有效位（MSB）为 '1' 代表负数。
    if binaryNumber[0] == '1':
        # 获取二进制数的位数 (N)
        numBits = len(binaryNumber)

        # 修正公式:
        # 正确的负数值 = (无符号数值) - 2^N
        # 对应 MATLAB 公式: intNumber = intNumber - 2^numBits
        intNumber = intNumber - (1 << numBits)

    # 函数结束，intNumber 中已存储了最终结果。
    return intNumber
