"""
UI辅助函数
UI Helper Functions
处理UI相关的通用功能
"""

from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QTextEdit, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt


def create_log_tab(parent):
    """创建日志标签页"""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    # 日志显示区域
    log_group = QGroupBox("運行日誌")
    log_layout = QVBoxLayout(log_group)
    
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            border: 1px solid #3e3e3e;
            border-radius: 4px;
            padding: 5px;
        }
    """)
    log_layout.addWidget(log_text)
    
    # 日志操作按钮
    log_btn_layout = QHBoxLayout()
    
    clear_log_btn = QPushButton("🗑️ 清空日誌")
    clear_log_btn.clicked.connect(lambda: clear_log(log_text))
    log_btn_layout.addWidget(clear_log_btn)
    
    save_log_btn = QPushButton("💾 保存日誌")
    save_log_btn.clicked.connect(lambda: save_log(log_text))
    log_btn_layout.addWidget(save_log_btn)
    
    log_layout.addLayout(log_btn_layout)
    layout.addWidget(log_group)
    
    # 将控件保存到tab对象中，以便外部访问
    tab.log_text = log_text
    tab.clear_log_btn = clear_log_btn
    tab.save_log_btn = save_log_btn
    
    return tab


def clear_log(log_text):
    """清空日志"""
    log_text.clear()
    log_message(log_text, "[INFO] 日誌已清空 Log cleared")


def save_log(log_text):
    """保存日志"""
    file_path, _ = QFileDialog.getSaveFileName(
        None, "保存日誌 Save Log",
        f"yolo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "文本文件 (*.txt)"
    )
    
    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_text.toPlainText())
            log_message(log_text, f"[SUCCESS] 日誌已保存: {file_path}")
            QMessageBox.information(
                None, "成功 Success",
                f"日誌已保存\nLog saved to:\n{file_path}"
            )
        except Exception as e:
            log_message(log_text, f"[ERROR] 保存日誌失敗: {e}")
            QMessageBox.critical(
                None, "錯誤 Error",
                f"保存日誌失敗 Failed to save log:\n{str(e)}"
            )


def log_message(log_text, message):
    """记录日志消息"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    log_text.append(formatted_message)
    
    # 自动滚动到底部
    scrollbar = log_text.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())


def get_global_style():
    """获取全局样式"""
    return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            color: #2c3e50;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QPushButton:pressed {
            background-color: #004085;
        }
        QPushButton:disabled {
            background-color: #6c757d;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 6px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #0078d4;
        }
    """


def get_tab_style():
    """获取标签页样式"""
    return """
        QTabWidget::pane {
            border: 2px solid #dee2e6;
            border-radius: 8px;
            background: white;
        }
        QTabBar::tab {
            background: #f8f9fa;
            color: #495057;
            padding: 10px 20px;
            margin-right: 2px;
            border: 2px solid #dee2e6;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background: white;
            color: #0078d4;
            border-bottom: 2px solid white;
        }
        QTabBar::tab:hover {
            background: #e9ecef;
        }
    """


def get_title_style():
    """获取标题样式"""
    return """
        QLabel {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #3498db, stop:1 #2ecc71);
            color: white;
            border-radius: 8px;
            margin-bottom: 10px;
        }
    """
