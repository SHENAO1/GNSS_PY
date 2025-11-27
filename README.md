# GNSS_PY

GNSS_PY 是一个基于 Python 的 GNSS 软件接收机项目，主要参考 SoftGNSS 思路，实现从数据采集、捕获、跟踪、导航电文解算到伪距计算和可视化的一整套流程。

本项目目前主要用于个人科研：  
- GNSS 捕获算法对比与优化  
- 跟踪环路（DLL/PLL）性能分析  
- 伪距与导航解算  
- 与 MATLAB 版本代码的对照与验证  

---

## 目录结构

项目整体结构（简化版）如下：

```text
GNSS_PY/
├── .venv/                 # Python 虚拟环境（不会提交到 Git）
├── data/                  # 数据（建议 data/raw 不提交）
│   ├── raw/               # 原始前端采样数据 *.bin / *.dat / *.mat 等
│   └── processed/         # 预处理后的中间结果
├── notebooks/             # Jupyter 笔记本，用于实验与可视化
├── src/
│   └── gnss/
│       ├── __init__.py
│       ├── main.py        # 入口脚本：整体流程 / demo
│       ├── settings.py    # 配置（文件路径 / 采样频率 / IF 等）
│       ├── settings_cli.py# 命令行版配置入口（可选）
│       │
│       ├── acquisition/   # 捕获模块
│       │   ├── __init__.py
│       │   ├── acquisition.py
│       │   ├── acquisition_core.py
│       │   └── ca_code.py
│       │
│       ├── tracking/      # 跟踪模块
│       │   ├── __init__.py
│       │   ├── pre_run.py
│       │   ├── dll.py
│       │   ├── pll.py
│       │   ├── loop_filter.py
│       │   ├── loop/
│       │   │   ├── __init__.py
│       │   │   └── calc_loop_coef.py
│       │   ├── show_channel_status.py
│       │   └── tracking_core.py
│       │
│       ├── navigation/    # 导航电文与伪距
│       │   ├── __init__.py
│       │   ├── ephemeris.py
│       │   ├── nav_msg.py
│       │   ├── pseudorange.py
│       │   ├── skyplot.py
│       │   └── ephemeris/
│       │       ├── __init__.py
│       │       ├── ephemeris.py
│       │       └── nav_party_chk.py
│       │
│       └── utils/         # 通用工具
│           ├── __init__.py
│           ├── bit_utils.py
│           ├── io.py
│           ├── math_utils.py
│           ├── plot_acquisition.py
│           ├── plot_navigation.py
│           ├── plot_tracking.py
│           ├── plotting.py
│           ├── signal_utils.py
│           └── twos_comp.py
│
├── tests/                 # 单元测试（后续补充）
├── requirements.txt       # Python 依赖
└── README.md
