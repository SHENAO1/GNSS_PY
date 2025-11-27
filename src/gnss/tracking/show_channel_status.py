"""
show_channel_status(channels, settings)

功能：
    在终端打印所有跟踪通道的状态，格式与 MATLAB 版本一致。

输入：
    channels : 一个列表，每个元素为 dict，格式参见 pre_run()。
    settings : 包含全局接收机设置，例如：
                - settings.numberOfChannels
                - settings.IF  （中频，IF，用于计算 Doppler）

输出：
    无（直接打印）
"""

def show_channel_status(channels, settings):
    # 表头与分隔线
    print("\n*=========*=====*===============*===========*=============*========*")
    print(  "| Channel | PRN |   Frequency   |  Doppler  | Code Offset | Status |")
    print(  "*=========*=====*===============*===========*=============*========*")

    num_ch = settings.numberOfChannels

    # 遍历每一个通道
    for ch_idx in range(num_ch):

        ch = channels[ch_idx]

        if ch["status"] != "-":
            # 活动通道（Tracking）
            # Doppler = 捕获频率 - IF
            doppler = ch["acquiredFreq"] - settings.IF

            print("|      {:2d} | {:3d} |  {:2.5e} |   {:5.0f}   |    {:6d}   |     {}  |".format(
                ch_idx + 1,               # 通道编号（从 1 开始）
                ch["PRN"],                # PRN
                ch["acquiredFreq"],       # 捕获频率
                doppler,                  # 多普勒频移
                ch["codePhase"],          # 码相位
                ch["status"]              # 状态
            ))
        else:
            # 非活动通道（Off）
            print("|      {:2d} | --- |  ------------ |   -----   |    ------   |   Off  |".format(
                ch_idx + 1
            ))

    print("*=========*=====*===============*===========*=============*========*\n")
