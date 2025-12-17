"""
YOLO Launcher GUI Package
现代化的YOLO统一启动器图形界面
采用模块化设计，符合软件工程最佳实践
"""

__version__ = "2.0.0"
__author__ = "YOLO Team"
__description__ = "YOLO统一启动器 - 模块化图形界面"

# 导出子模块
from . import modules
from . import utils
from . import config
from . import workers

__all__ = ['modules', 'utils', 'config', 'workers']

