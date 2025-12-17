"""
UI工具模块
UI Utility Module
提供UI相关的辅助函数
"""

from typing import Optional
from PyQt5.QtWidgets import QMessageBox, QWidget


def show_message(parent: Optional[QWidget],
                title: str,
                message: str,
                message_type: str = 'info') -> int:
    """
    显示消息对话框
    
    Args:
        parent: 父窗口
        title: 标题
        message: 消息内容
        message_type: 消息类型 ('info', 'warning', 'error', 'question')
        
    Returns:
        用户选择的按钮
    """
    message_types = {
        'info': QMessageBox.information,
        'warning': QMessageBox.warning,
        'error': QMessageBox.critical,
        'question': QMessageBox.question
    }
    
    show_func = message_types.get(message_type.lower(), QMessageBox.information)
    
    if message_type.lower() == 'question':
        return show_func(
            parent, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
    else:
        return show_func(parent, title, message)


def show_info(parent: Optional[QWidget],
             message: str,
             title: str = "信息 Info") -> None:
    """显示信息对话框"""
    QMessageBox.information(parent, title, message)


def show_warning(parent: Optional[QWidget],
                message: str,
                title: str = "警告 Warning") -> None:
    """显示警告对话框"""
    QMessageBox.warning(parent, title, message)


def show_error(parent: Optional[QWidget],
              message: str,
              title: str = "错误 Error") -> None:
    """显示错误对话框"""
    QMessageBox.critical(parent, title, message)


def show_question(parent: Optional[QWidget],
                 message: str,
                 title: str = "确认 Confirm") -> bool:
    """
    显示确认对话框
    
    Returns:
        True表示用户点击Yes，False表示No
    """
    reply = QMessageBox.question(
        parent, title, message,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return reply == QMessageBox.Yes


def get_style_sheet(theme: str = 'default') -> str:
    """
    获取样式表
    
    Args:
        theme: 主题名称 ('default', 'dark', 'light')
        
    Returns:
        样式表字符串
    """
    if theme == 'dark':
        return """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #4a4a4a;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """
    elif theme == 'light':
        return """
            QMainWindow {
                background-color: #ffffff;
            }
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
        """
    else:  # default
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_progress(current: int, total: int) -> str:
    """
    格式化进度显示
    
    Args:
        current: 当前进度
        total: 总进度
        
    Returns:
        格式化的进度字符串
    """
    if total == 0:
        return "0%"
    
    percentage = (current / total) * 100
    return f"{current}/{total} ({percentage:.1f}%)"

