"""捕获模块 (acquisition)。

这里主要实现:
  - C/A 码相关工具 (ca_code)
  - 并行码相位搜索等捕获算法
"""

from .ca_code import load_ca_table, get_ca_code

# src/gnss/acquisition/acquisition.py
from .acquisition_core import acquisition

__all__ = ["acquisition"]
