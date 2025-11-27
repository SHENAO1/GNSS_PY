"""
GNSS Python package.
"""

# 这里只导出现有的函数，不再导出不存在的 GnssSettings
from .settings import init_settings

__all__ = ["init_settings"]
