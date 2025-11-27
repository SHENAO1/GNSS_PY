"""捕获模块 (acquisition)。

这里主要实现:
  - C/A 码相关工具 (ca_code)
  - 并行码相位搜索等捕获算法
"""

"""
GNSS acquisition subpackage.

这里对外只暴露一个高层接口 acquisition()，
真正的实现放在 acquisition_core.py 里。
"""

from .acquisition_core import acquisition

__all__ = ["acquisition"]

