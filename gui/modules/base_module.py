"""
基础模块类
Base Module Class
提供所有功能模块的通用基础设施
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QObject, pyqtSignal


class BaseModule(QObject):
    """
    所有功能模块的基类
    Base class for all functional modules
    """
    
    # 信号定义
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(bool, int, int, str)  # show, current, total, text
    
    def __init__(self, parent=None):
        """
        初始化基础模块
        
        Args:
            parent: 父窗口对象，用于信号连接
        """
        super().__init__(parent)
        self.parent = parent
        self.tab_widget = None
        
    def log(self, message):
        """
        发送日志消息
        
        Args:
            message: 日志内容
        """
        self.log_signal.emit(message)
        
    def update_status(self, message):
        """
        更新状态栏消息
        
        Args:
            message: 状态消息
        """
        self.status_signal.emit(message)
        
    def show_progress(self, show=True, current=0, total=0, text=""):
        """
        显示或隐藏进度条
        
        Args:
            show: 是否显示进度条
            current: 当前进度
            total: 总进度
            text: 进度文本
        """
        self.progress_signal.emit(show, current, total, text)
        
    def create_tab(self):
        """
        创建标签页
        子类必须实现此方法
        
        Returns:
            QWidget: 标签页控件
        """
        raise NotImplementedError("Subclass must implement create_tab()")
        
    def connect_signals(self, main_window):
        """
        连接信号到主窗口
        
        Args:
            main_window: 主窗口对象
        """
        if hasattr(main_window, 'log_message'):
            self.log_signal.connect(main_window.log_message)
        if hasattr(main_window, 'update_status'):
            self.status_signal.connect(main_window.update_status)
        if hasattr(main_window, 'show_progress'):
            self.progress_signal.connect(main_window.show_progress)

