# src/gnss/main.py

from __future__ import annotations

import os
import sys

# === 手动把 src 加入 sys.path，保证能 import gnss ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/gnss
SRC_DIR = os.path.dirname(CURRENT_DIR)                    # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gnss.settings import init_settings
from gnss.utils.probe_data import probe_data
from gnss.post_processing import post_processing
from gnss.utils import plotting


def main():
    # ===== 初始化 settings，相当于 MATLAB 的 initSettings.m =====
    settings = init_settings()

    # ===== 跟踪阶段控制选项 =====================================
    # 是否尝试使用 GPU（CuPy），目前环境还在调试，可以先保持 False
    settings.use_gpu_tracking = False

    # 是否在 tracking 中打印进度
    #   True  -> 按一定间隔打印
    #   False -> 完全不打印（tracking_core 里就不会 print）
    settings.verboseTracking = True

    # 每隔多少毫秒打印一次进度（单位：ms）
    # 例如：
    #   1000 -> 每 1000 ms 打印一次（原来的行为）
    #   5000 -> 每 5000 ms 打印一次
    #   10000 -> 每 1 秒钟数据打印一次
    settings.trackingPrintInterval = 5000

    # 想改频率，只要改上面这一行就可以了 👆


    settings.saveTrackingResults = False           # 需要保存时打开
    # settings.resultsDir = r"E:\GNSS_py_results"  # 想改目录就改这里

    # 是否绘制跟踪结果的各种图形，默认 Flase
    settings.plotTracking = False


    # ========================================================

    # ===== 探测原始数据（时域/频域/直方图） =====
    try:
        print(f'Probing data ({settings.fileName})...')
        probe_data(settings)
    except Exception as e:
        # 和 MATLAB 里 try/catch 的逻辑一样：打印错误然后退出
        print()
        print("原始数据探测出错：")
        print(f"  {e}")
        print('  (请修改 "settings.py" 中的 init_settings() 或相应配置后重试)')
        return

    print("  Raw IF data plotted")
    print('  (如需重新配置，请修改 "settings.py" 中的 init_settings() )')
    print()

    # ===== 询问是否开始 GNSS 处理 =====
    try:
        user_input = input('Enter "1" to initiate GNSS processing or "0" to exit : ')
    except EOFError:
        # 非交互环境（比如脚本调用）就直接退出
        return

    user_input = user_input.strip()

    if user_input == "1":
        print()
        # 开始完整后处理流程，相当于 MATLAB 的 postProcessing.m
        post_processing(settings)
    else:
        print("Exit without processing.")
        return


if __name__ == "__main__":
    main()
