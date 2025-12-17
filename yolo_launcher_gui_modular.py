"""
YOLO 統一啟動器 - 模块化版本
Modular YOLO Launcher GUI
基於PyQt5的現代化圖形界面 - 采用模块化设计
"""

import sys
import os
import torch
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QLabel, QPushButton,
                            QStatusBar, QTextEdit, QGroupBox,
                            QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QFont

# 添加Code目录到Python路径
code_dir = os.path.join(os.path.dirname(__file__), "Code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)


# 导入功能模块
from gui.modules import (
    DataConversionModule,
    TrainingModule,
    InferenceModule,
    ModelAnalyzerModule,
    ModelModifierModule,
    StereoTrainingModule,
    StereoInferenceModule
)

# 导入设置管理器
from gui.config.settings import SettingsManager

# 导入工具函数
from gui.utils import get_gpu_name, create_log_tab, get_global_style, get_tab_style, get_title_style

# 导入工作线程
from gui.workers import WorkerThread


class YOLOLauncherModular(QMainWindow):
    """YOLO 統一啟動器 - 模块化版本"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 統一啟動器 - 模块化版本")
        
        # 初始化设置管理器
        self.settings_manager = SettingsManager()
        
        # 初始化功能模块
        self.data_conversion_module = DataConversionModule(self)
        self.training_module = TrainingModule(self)
        self.inference_module = InferenceModule(self)
        self.model_analyzer_module = ModelAnalyzerModule(self)
        self.model_modifier_module = ModelModifierModule(self)
        self.stereo_training_module = StereoTrainingModule(self)
        self.stereo_inference_module = StereoInferenceModule(self)
        
        # 连接所有模块的信号
        self._connect_module_signals()
        
        # 设置UI
        self.setup_ui()
        self.setup_style()
        
        # 加载保存的设置
        self.load_settings()
        
        # 设置设备信息显示
        self._update_device_display()
        
        # 显示欢迎信息 (在UI设置完成后)
        self.log_message("="*60)
        self.log_message("🎯 YOLO 統一啟動器 - 模块化版本")
        self.log_message("="*60)
        self.log_message(f"📅 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"🖥️  GPU: {get_gpu_name()}")
        self.log_message("="*60)
        
    def _connect_module_signals(self):
        """连接所有功能模块的信号"""
        modules = [
            self.data_conversion_module,
            self.training_module,
            self.inference_module,
            self.model_analyzer_module,
            self.model_modifier_module,
            self.stereo_training_module,
            self.stereo_inference_module
        ]
        
        for module in modules:
            module.connect_signals(self)
    
    def _update_device_display(self):
        """更新设备信息显示"""
        try:
            from gui.utils import get_device_info
            device_info = get_device_info()
            
            if device_info['device'] == 'cuda':
                # 提取GPU型号名称（去掉括号中的信息）
                gpu_name = device_info['name']
                if '(' in gpu_name:
                    gpu_name = gpu_name.split('(')[0].strip()
                
                # 格式化显示
                device_text = f"🖥️ {gpu_name}"
                if device_info['memory_gb'] > 0:
                    device_text += f"\n💾 {device_info['memory_gb']:.1f}GB"
                
                self.device_label.setText(device_text)
                self.device_label.setToolTip(f"GPU: {device_info['name']}\nMemory: {device_info['memory_gb']:.1f}GB\nCount: {device_info['count']}")
            else:
                self.device_label.setText("💻 CPU")
                self.device_label.setToolTip("Using CPU for computation")
                
        except Exception as e:
            self.device_label.setText("❓ Unknown")
            self.device_label.setToolTip(f"Device detection failed: {e}")
            
            
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # 标题和设备信息
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        # 主标题
        title_label = QLabel("YOLO 統一啟動器 - 模块化版本")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(get_title_style())
        title_layout.addWidget(title_label)
        
        # 设备信息标签
        self.device_label = QLabel()
        self.device_label.setAlignment(Qt.AlignCenter)
        self.device_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #e8f5e8, stop:1 #f0f8ff);
                border: 2px solid #28a745;
                border-radius: 6px;
                padding: 8px 12px;
                margin-left: 10px;
                min-width: 120px;
            }
        """)
        title_layout.addWidget(self.device_label)
        
        main_layout.addWidget(title_container)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(get_tab_style())
        
        # 添加各功能标签页
        self.tab_widget.addTab(
            self.data_conversion_module.create_tab(),
            "🔄 數據轉換"
        )
        self.tab_widget.addTab(
            self.training_module.create_tab(),
            "🚀 模型訓練"
        )
        self.tab_widget.addTab(
            self.inference_module.create_tab(),
            "🔍 模型推理"
        )
        self.tab_widget.addTab(
            self.model_analyzer_module.create_tab(),
            "📊 模型分析"
        )
        self.tab_widget.addTab(
            self.model_modifier_module.create_tab(),
            "🔧 模型修改"
        )
        self.tab_widget.addTab(
            self.stereo_training_module.create_tab(),
            "👁️ 立體視覺訓練"
        )
        self.tab_widget.addTab(
            self.stereo_inference_module.create_tab(),
            "🔍 立體視覺推理"
        )
        
        # 日志标签页
        log_tab = create_log_tab(self)
        self.tab_widget.addTab(log_tab, "📋 運行日誌")
        
        # 保存日志控件引用
        if hasattr(log_tab, 'log_text'):
            self.log_text = log_tab.log_text
            self.clear_log_btn = log_tab.clear_log_btn
            self.save_log_btn = log_tab.save_log_btn
        else:
            # 如果create_log_tab没有正确返回控件，创建一个简单的日志控件
            from PyQt5.QtWidgets import QTextEdit
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.clear_log_btn = None
            self.save_log_btn = None
        
        main_layout.addWidget(self.tab_widget)
        
        # 创建状态栏
        self.create_status_bar()
        
        # 初始化模型列表
        self.model_analyzer_module.refresh_analyzer_model_list()
        
        # 初始化训练模块的模型列表
        self.training_module.refresh_model_list()
        
        # 初始化立體視覺訓練模組的模型列表
        self.stereo_training_module.refresh_stereo_model_list()
        
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态标签
        self.status_label = QLabel("就緒 Ready")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.status_bar.addWidget(self.status_label)
        
    def setup_style(self):
        """设置全局样式"""
        self.setStyleSheet(get_global_style())
        
    def log_message(self, message):
        """记录日志消息"""
        try:
            if hasattr(self, 'log_text') and self.log_text is not None:
                timestamp = datetime.now().strftime('%H:%M:%S')
                formatted_message = f"[{timestamp}] {message}"
                self.log_text.append(formatted_message)
                
                # 自动滚动到底部
                scrollbar = self.log_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            else:
                # 如果log_text不可用，打印到控制台
                print(f"[LOG] {message}")
        except Exception as e:
            # 如果日志记录失败，打印到控制台
            print(f"[LOG ERROR] {message} (Error: {e})")
        
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_label.setText(message)
        
    def show_progress(self, show, current=0, total=0, text=""):
        """显示进度信息（通过状态栏文字显示）"""
        try:
            if show and total > 0:
                # 计算百分比
                percentage = int((current / total) * 100) if total > 0 else 0
                # 在状态栏显示进度信息
                if text:
                    status_text = f"{text} - {current}/{total} ({percentage}%)"
                else:
                    status_text = f"進行中 In Progress - {current}/{total} ({percentage}%)"
                self.update_status(status_text)
            elif show:
                # 只显示文本，不显示具体进度
                if text:
                    self.update_status(text)
                else:
                    self.update_status("處理中 Processing...")
            else:
                # 隐藏进度，恢复就绪状态
                self.update_status("就緒 Ready")
        except Exception as e:
            print(f"[ERROR] Progress update failed: {e}")
            
                
    def load_settings(self):
        """加载保存的设置"""
        try:
            # 加载窗口几何信息
            window_geometry = self.settings_manager.get('window.geometry')
            if window_geometry:
                self.setGeometry(
                    window_geometry.get('x', 100),
                    window_geometry.get('y', 100),
                    window_geometry.get('width', 1400),
                    window_geometry.get('height', 900)
                )
                self.log_message("✅ 已加载保存的窗口位置和大小")
            else:
                self.setGeometry(100, 100, 1400, 900)
                self.log_message("ℹ️ 使用默认窗口大小")
            
            # 加载最后使用的标签页
            last_tab = self.settings_manager.get('window.last_tab_index', 0)
            if 0 <= last_tab < self.tab_widget.count():
                self.tab_widget.setCurrentIndex(last_tab)
            
            # 通知所有模块加载设置
            modules = [
                self.data_conversion_module,
                self.training_module,
                self.inference_module,
                self.model_analyzer_module,
                self.model_modifier_module,
                self.stereo_training_module,
                self.stereo_inference_module
            ]
            
            for module in modules:
                if hasattr(module, 'load_settings'):
                    module.load_settings(self.settings_manager)
            
            self.log_message("✅ 设置加载完成")
            
        except Exception as e:
            self.log_message(f"[WARNING] 加载设置失败: {e}")
    
    def save_settings(self):
        """保存当前设置"""
        try:
            # 保存窗口几何信息
            geometry = self.geometry()
            self.settings_manager.set('window.geometry', {
                'x': geometry.x(),
                'y': geometry.y(),
                'width': geometry.width(),
                'height': geometry.height()
            })
            
            # 保存当前标签页
            self.settings_manager.set('window.last_tab_index', self.tab_widget.currentIndex())
            
            # 通知所有模块保存设置
            modules = [
                self.data_conversion_module,
                self.training_module,
                self.inference_module,
                self.model_analyzer_module,
                self.model_modifier_module,
                self.stereo_training_module,
                self.stereo_inference_module
            ]
            
            for module in modules:
                if hasattr(module, 'save_settings'):
                    try:
                        module.save_settings(self.settings_manager)
                        self.log_message(f"✅ {module.__class__.__name__} 設定已保存")
                    except Exception as e:
                        self.log_message(f"[WARNING] {module.__class__.__name__} 設定保存失敗: {e}")
            
            # 保存到文件
            if self.settings_manager.save():
                self.log_message("✅ 设置保存成功")
            else:
                self.log_message("[WARNING] 设置保存失败")
                
        except Exception as e:
            self.log_message(f"[ERROR] 保存设置失败: {e}")
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        reply = QMessageBox.question(
            self, "確認退出 Confirm Exit",
            "確定要退出YOLO統一啟動器嗎？\nAre you sure to exit YOLO Launcher?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 保存设置
            self.save_settings()
            
            self.log_message("="*60)
            self.log_message("👋 退出YOLO統一啟動器 Exiting YOLO Launcher")
            self.log_message("="*60)
            event.accept()
        else:
            event.ignore()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = YOLOLauncherModular()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

