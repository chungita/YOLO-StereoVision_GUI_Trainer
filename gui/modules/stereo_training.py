

import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
                            QFileDialog, QMessageBox, QCheckBox, QTabWidget,
                            QScrollArea, QTabBar, QDoubleSpinBox, QSpinBox,
                            QCheckBox, QDoubleSpinBox, QComboBox, QLabel,
                            QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout,
                            QDialog)
from PyQt5.QtCore import Qt
from .base_module import BaseModule

# 添加Code目录到Python路径
code_dir = Path(__file__).parent.parent.parent / "Code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))


class StereoParamsDialog(QDialog):
    """立體視覺參數設置對話框 - 包含高級參數和增廣參數"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 立體視覺參數設置 Stereo Vision Parameters")
        self.setModal(True)
        self.setMinimumSize(1000, 800)
        self.setMaximumSize(1400, 1000)
        
        # 添加關閉事件處理
        self.finished.connect(self.on_dialog_finished)
        
        # 設置窗口樣式
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
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
            QPushButton:pressed {
                background-color: #004085;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        self.setup_ui()
        
    def on_dialog_finished(self, result):
        """對話框關閉時的回調函數"""
        try:
            if result == QDialog.Accepted:
                self.log("✅ 參數設置已確認")
            else:
                self.log("ℹ️ 參數設置已取消")
                
        except Exception as e:
            self.log(f"[WARNING] 處理對話框關閉事件時發生錯誤: {e}")
    
    def log(self, message):
        """記錄日誌消息"""
        try:
            # 嘗試從父窗口獲取日誌功能
            if hasattr(self.parent(), 'log'):
                self.parent().log(message)
            else:
                print(f"[StereoParamsDialog] {message}")
        except:
            print(f"[StereoParamsDialog] {message}")
        
    def setup_ui(self):
        """設置用戶界面"""
        layout = QVBoxLayout(self)
        
        # 創建標籤頁
        tab_widget = QTabWidget()
        
        # 高級參數標籤頁
        advanced_tab = self.create_advanced_tab()
        tab_widget.addTab(advanced_tab, "🔧 高級參數 Advanced")
        
        # 增廣參數標籤頁
        augmentation_tab = self.create_augmentation_tab()
        tab_widget.addTab(augmentation_tab, "🎨 增廣參數 Augmentation")
        
        layout.addWidget(tab_widget)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 重置為默認值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("✅ 確定")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        
    def create_advanced_tab(self):
        """創建高級參數標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 創建滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 主內容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 模型架構參數組
        architecture_group = QGroupBox("模型架構參數 Model Architecture Parameters")
        architecture_layout = QGridLayout(architecture_group)
        
        # 相關實現選項
        architecture_layout.addWidget(QLabel("相關實現 Corr Implementation:"), 0, 0)
        self.corr_implementation_combo = QComboBox()
        self.corr_implementation_combo.addItems([
            'reg (默認 Default)',
            'alt', 
            'reg_cuda', 
            'alt_cuda'
        ])
        self.corr_implementation_combo.setCurrentText('reg (默認 Default)')
        self.corr_implementation_combo.setToolTip("相關體積實現方式 Correlation volume implementation")
        architecture_layout.addWidget(self.corr_implementation_combo, 0, 1)
        
        # 下採樣層數
        architecture_layout.addWidget(QLabel("下採樣層數 N Downsample:"), 0, 2)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 4)
        self.downsample_spin.setValue(2)
        self.downsample_spin.setToolTip("視差場分辨率 (1/2^K)")
        architecture_layout.addWidget(self.downsample_spin, 0, 3)
        
        # 相關體積參數
        architecture_layout.addWidget(QLabel("相關層數 Corr Levels:"), 1, 0)
        self.corr_levels_spin = QSpinBox()
        self.corr_levels_spin.setRange(1, 8)
        self.corr_levels_spin.setValue(4)
        self.corr_levels_spin.setToolTip("相關金字塔層數")
        architecture_layout.addWidget(self.corr_levels_spin, 1, 1)
        
        architecture_layout.addWidget(QLabel("相關半徑 Corr Radius:"), 1, 2)
        self.corr_radius_spin = QSpinBox()
        self.corr_radius_spin.setRange(1, 8)
        self.corr_radius_spin.setValue(4)
        self.corr_radius_spin.setToolTip("相關金字塔寬度")
        architecture_layout.addWidget(self.corr_radius_spin, 1, 3)
        
        # GRU層數
        architecture_layout.addWidget(QLabel("GRU層數 N GRU Layers:"), 2, 0)
        self.gru_layers_spin = QSpinBox()
        self.gru_layers_spin.setRange(1, 5)
        self.gru_layers_spin.setValue(3)
        self.gru_layers_spin.setToolTip("隱藏GRU層數")
        architecture_layout.addWidget(self.gru_layers_spin, 2, 1)
        
        # 共享骨幹網絡
        architecture_layout.addWidget(QLabel("共享骨幹 Shared Backbone:"), 2, 2)
        self.shared_backbone_cb = QCheckBox()
        self.shared_backbone_cb.setToolTip("為上下文和特徵編碼器使用單一骨幹")
        architecture_layout.addWidget(self.shared_backbone_cb, 2, 3)
        
        # 上下文正規化
        architecture_layout.addWidget(QLabel("上下文正規化 Context Norm:"), 3, 0)
        self.context_norm_combo = QComboBox()
        self.context_norm_combo.addItems([
            'batch',
            'group', 
            'instance',
            'none'
        ])
        self.context_norm_combo.setCurrentText('batch')
        self.context_norm_combo.setToolTip("上下文編碼器正規化方式")
        architecture_layout.addWidget(self.context_norm_combo, 3, 1)
        
        # 慢快GRU
        architecture_layout.addWidget(QLabel("慢快GRU Slow Fast GRU:"), 3, 2)
        self.slow_fast_gru_cb = QCheckBox()
        self.slow_fast_gru_cb.setToolTip("更頻繁地迭代低分辨率GRU")
        architecture_layout.addWidget(self.slow_fast_gru_cb, 3, 3)
        
        # 隱藏維度
        architecture_layout.addWidget(QLabel("隱藏維度 Hidden Dims:"), 4, 0)
        self.hidden_dims_combo = QComboBox()
        self.hidden_dims_combo.addItems([
            "128x128x128 (默認)",
            "64x64x64",
            "96x96x96", 
            "160x160x160",
            "192x192x192"
        ])
        self.hidden_dims_combo.setToolTip("隱藏狀態和上下文維度")
        architecture_layout.addWidget(self.hidden_dims_combo, 4, 1)
        
        content_layout.addWidget(architecture_group)
        
        # 優化參數組
        optimization_group = QGroupBox("優化參數 Optimization Parameters")
        optimization_layout = QGridLayout(optimization_group)
        
        # 混合精度
        optimization_layout.addWidget(QLabel("混合精度 Mixed Precision:"), 0, 0)
        self.mixed_precision_cb = QCheckBox()
        self.mixed_precision_cb.setChecked(False)
        self.mixed_precision_cb.setToolTip("使用混合精度訓練")
        optimization_layout.addWidget(self.mixed_precision_cb, 0, 1)
        
        # 權重衰減
        optimization_layout.addWidget(QLabel("權重衰減 Weight Decay:"), 0, 2)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.001)
        self.weight_decay_spin.setValue(0.00001)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setToolTip("優化器權重衰減")
        optimization_layout.addWidget(self.weight_decay_spin, 0, 3)
        
        content_layout.addWidget(optimization_group)
        
        # 訓練控制參數組
        training_control_group = QGroupBox("訓練控制參數 Training Control Parameters")
        training_control_layout = QGridLayout(training_control_group)
        
        # 訓練迭代
        training_control_layout.addWidget(QLabel("訓練迭代 Train Iters:"), 0, 0)
        self.train_iters_spin = QSpinBox()
        self.train_iters_spin.setRange(1, 100)
        self.train_iters_spin.setValue(16)
        self.train_iters_spin.setToolTip("訓練時的迭代次數")
        training_control_layout.addWidget(self.train_iters_spin, 0, 1)
        
        # 驗證迭代
        training_control_layout.addWidget(QLabel("驗證迭代 Valid Iters:"), 0, 2)
        self.valid_iters_spin = QSpinBox()
        self.valid_iters_spin.setRange(1, 100)
        self.valid_iters_spin.setValue(32)
        self.valid_iters_spin.setToolTip("驗證時的迭代次數")
        training_control_layout.addWidget(self.valid_iters_spin, 0, 3)
        
        # 學習率
        training_control_layout.addWidget(QLabel("學習率 Learning Rate:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 0.01)
        self.learning_rate_spin.setValue(0.0002)
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setToolTip("最大學習率")
        training_control_layout.addWidget(self.learning_rate_spin, 1, 1)
        
        # 圖像尺寸
        training_control_layout.addWidget(QLabel("圖像尺寸 Image Size:"), 1, 2)
        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems([
            "320x720 (默認 Default)",
            "640x480 (原始尺寸 Original)",
            "256x512",
            "384x768", 
            "512x1024",
            "640x1280"
        ])
        self.image_size_combo.setToolTip("訓練時隨機裁剪的圖像尺寸")
        training_control_layout.addWidget(self.image_size_combo, 1, 3)
        
        content_layout.addWidget(training_control_group)
        
        # 設置滾動區域內容
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
        
    def create_augmentation_tab(self):
        """創建增廣參數標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 創建滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 主內容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 空間變換參數組
        spatial_group = QGroupBox("空間變換參數 Spatial Transformation Parameters")
        spatial_layout = QGridLayout(spatial_group)
        
        # 空間縮放
        spatial_layout.addWidget(QLabel("空間縮放 Spatial Scale:"), 0, 0)
        spatial_layout.addWidget(QLabel("最小值 Min:"), 0, 1)
        self.spatial_scale_min_spin = QDoubleSpinBox()
        self.spatial_scale_min_spin.setRange(-1.0, 1.0)
        self.spatial_scale_min_spin.setValue(-0.2)
        self.spatial_scale_min_spin.setDecimals(1)
        self.spatial_scale_min_spin.setToolTip("空間縮放最小值")
        spatial_layout.addWidget(self.spatial_scale_min_spin, 0, 2)
        
        spatial_layout.addWidget(QLabel("最大值 Max:"), 0, 3)
        self.spatial_scale_max_spin = QDoubleSpinBox()
        self.spatial_scale_max_spin.setRange(-1.0, 1.0)
        self.spatial_scale_max_spin.setValue(0.4)
        self.spatial_scale_max_spin.setDecimals(1)
        self.spatial_scale_max_spin.setToolTip("空間縮放最大值")
        spatial_layout.addWidget(self.spatial_scale_max_spin, 0, 4)
        
        content_layout.addWidget(spatial_group)
        
        # 顏色變換參數組
        color_group = QGroupBox("顏色變換參數 Color Transformation Parameters")
        color_layout = QGridLayout(color_group)
        
        # 飽和度範圍
        color_layout.addWidget(QLabel("飽和度範圍 Saturation Range:"), 0, 0)
        color_layout.addWidget(QLabel("最小值 Min:"), 0, 1)
        self.saturation_min_spin = QDoubleSpinBox()
        self.saturation_min_spin.setRange(0.0, 2.0)
        self.saturation_min_spin.setValue(0.0)
        self.saturation_min_spin.setDecimals(1)
        self.saturation_min_spin.setToolTip("飽和度最小值")
        color_layout.addWidget(self.saturation_min_spin, 0, 2)
        
        color_layout.addWidget(QLabel("最大值 Max:"), 0, 3)
        self.saturation_max_spin = QDoubleSpinBox()
        self.saturation_max_spin.setRange(0.0, 2.0)
        self.saturation_max_spin.setValue(1.4)
        self.saturation_max_spin.setDecimals(1)
        self.saturation_max_spin.setToolTip("飽和度最大值")
        color_layout.addWidget(self.saturation_max_spin, 0, 4)
        
        # Gamma範圍
        color_layout.addWidget(QLabel("Gamma範圍 Gamma Range:"), 1, 0)
        color_layout.addWidget(QLabel("最小值 Min:"), 1, 1)
        self.gamma_min_spin = QDoubleSpinBox()
        self.gamma_min_spin.setRange(0.5, 2.0)
        self.gamma_min_spin.setValue(0.8)
        self.gamma_min_spin.setDecimals(1)
        self.gamma_min_spin.setToolTip("Gamma最小值")
        color_layout.addWidget(self.gamma_min_spin, 1, 2)
        
        color_layout.addWidget(QLabel("最大值 Max:"), 1, 3)
        self.gamma_max_spin = QDoubleSpinBox()
        self.gamma_max_spin.setRange(0.5, 2.0)
        self.gamma_max_spin.setValue(1.2)
        self.gamma_max_spin.setDecimals(1)
        self.gamma_max_spin.setToolTip("Gamma最大值")
        color_layout.addWidget(self.gamma_max_spin, 1, 4)
        
        content_layout.addWidget(color_group)
        
        # 翻轉和變換參數組
        transform_group = QGroupBox("翻轉和變換參數 Flip and Transform Parameters")
        transform_layout = QGridLayout(transform_group)
        
        # 翻轉選項
        transform_layout.addWidget(QLabel("圖像翻轉 Image Flip:"), 0, 0)
        self.do_flip_combo = QComboBox()
        self.do_flip_combo.addItems([
            "無 None",
            "水平翻轉 Horizontal",
            "垂直翻轉 Vertical"
        ])
        self.do_flip_combo.setToolTip("圖像翻轉方式")
        transform_layout.addWidget(self.do_flip_combo, 0, 1)
        
        # 其他選項
        transform_layout.addWidget(QLabel("其他選項 Other Options:"), 0, 2)
        self.noyjitter_cb = QCheckBox("禁用Y抖動")
        self.noyjitter_cb.setToolTip("不模擬不完美的校正")
        transform_layout.addWidget(self.noyjitter_cb, 0, 3)
        
        content_layout.addWidget(transform_group)
        
        # 設置滾動區域內容
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        return tab
        
    def reset_to_defaults(self):
        """重置為默認值"""
        # 高級參數默認值
        self.corr_implementation_combo.setCurrentText('reg (默認 Default)')
        self.downsample_spin.setValue(2)
        self.corr_levels_spin.setValue(4)
        self.corr_radius_spin.setValue(4)
        self.gru_layers_spin.setValue(3)
        self.shared_backbone_cb.setChecked(False)
        self.context_norm_combo.setCurrentText('batch')
        self.slow_fast_gru_cb.setChecked(False)
        self.hidden_dims_combo.setCurrentText("128x128x128 (默認)")
        
        self.mixed_precision_cb.setChecked(False)
        self.weight_decay_spin.setValue(0.00001)
        
        self.train_iters_spin.setValue(16)
        self.valid_iters_spin.setValue(32)
        self.learning_rate_spin.setValue(0.0002)
        self.image_size_combo.setCurrentText("640x480 (原始尺寸 Original)")
        
        # 增廣參數默認值
        self.spatial_scale_min_spin.setValue(-0.2)
        self.spatial_scale_max_spin.setValue(0.4)
        self.saturation_min_spin.setValue(0.0)
        self.saturation_max_spin.setValue(1.4)
        self.gamma_min_spin.setValue(0.8)
        self.gamma_max_spin.setValue(1.2)
        self.do_flip_combo.setCurrentText("無 None")
        self.noyjitter_cb.setChecked(False)
    
    def get_all_params(self):
        """獲取所有參數值"""
        return {
            # 高級參數
            'corr_implementation': self.corr_implementation_combo.currentText().split(' ')[0],
            'n_downsample': self.downsample_spin.value(),
            'corr_levels': self.corr_levels_spin.value(),
            'corr_radius': self.corr_radius_spin.value(),
            'n_gru_layers': self.gru_layers_spin.value(),
            'shared_backbone': self.shared_backbone_cb.isChecked(),
            'context_norm': self.context_norm_combo.currentText(),
            'slow_fast_gru': self.slow_fast_gru_cb.isChecked(),
            'hidden_dims': self.hidden_dims_combo.currentText(),
            'mixed_precision': self.mixed_precision_cb.isChecked(),
            'weight_decay': self.weight_decay_spin.value(),
            'train_iters': self.train_iters_spin.value(),
            'valid_iters': self.valid_iters_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'image_size': self.parse_image_size(self.image_size_combo.currentText()),
            
            # 增廣參數
            'spatial_scale_min': self.spatial_scale_min_spin.value(),
            'spatial_scale_max': self.spatial_scale_max_spin.value(),
            'saturation_min': self.saturation_min_spin.value(),
            'saturation_max': self.saturation_max_spin.value(),
            'gamma_min': self.gamma_min_spin.value(),
            'gamma_max': self.gamma_max_spin.value(),
            'do_flip': self.do_flip_combo.currentText(),
            'noyjitter': self.noyjitter_cb.isChecked()
        }
    
    def set_all_params(self, params):
        """設置所有參數值"""
        # 如果沒有提供參數，使用空字典（會使用默認值）
        if not params:
            params = {}
        
        # 高級參數
        if 'corr_implementation' in params:
            for i in range(self.corr_implementation_combo.count()):
                if self.corr_implementation_combo.itemText(i).startswith(params['corr_implementation']):
                    self.corr_implementation_combo.setCurrentIndex(i)
                    break
        
        if 'n_downsample' in params:
            self.downsample_spin.setValue(params['n_downsample'])
        if 'corr_levels' in params:
            self.corr_levels_spin.setValue(params['corr_levels'])
        if 'corr_radius' in params:
            self.corr_radius_spin.setValue(params['corr_radius'])
        if 'n_gru_layers' in params:
            self.gru_layers_spin.setValue(params['n_gru_layers'])
        if 'shared_backbone' in params:
            self.shared_backbone_cb.setChecked(params['shared_backbone'])
        if 'context_norm' in params:
            self.context_norm_combo.setCurrentText(params['context_norm'])
        if 'slow_fast_gru' in params:
            self.slow_fast_gru_cb.setChecked(params['slow_fast_gru'])
        if 'hidden_dims' in params:
            self.hidden_dims_combo.setCurrentText(params['hidden_dims'])
        if 'mixed_precision' in params:
            self.mixed_precision_cb.setChecked(params['mixed_precision'])
        if 'weight_decay' in params:
            self.weight_decay_spin.setValue(params['weight_decay'])
        if 'train_iters' in params:
            self.train_iters_spin.setValue(params['train_iters'])
        if 'valid_iters' in params:
            self.valid_iters_spin.setValue(params['valid_iters'])
        if 'learning_rate' in params:
            self.learning_rate_spin.setValue(params['learning_rate'])
        if 'image_size' in params:
            # 如果参数是列表格式 [width, height]，转换为文本格式
            if isinstance(params['image_size'], list) and len(params['image_size']) == 2:
                size_text = f"{params['image_size'][0]}x{params['image_size'][1]}"
                # 查找匹配的选项
                for i in range(self.image_size_combo.count()):
                    if self.image_size_combo.itemText(i).startswith(size_text):
                        self.image_size_combo.setCurrentIndex(i)
                        break
                else:
                    # 如果没有找到匹配的选项，使用默认值
                    self.image_size_combo.setCurrentText("320x720 (默認 Default)")
            else:
                self.image_size_combo.setCurrentText(params['image_size'])
        
        # 增廣參數
        if 'spatial_scale_min' in params:
            self.spatial_scale_min_spin.setValue(params['spatial_scale_min'])
        if 'spatial_scale_max' in params:
            self.spatial_scale_max_spin.setValue(params['spatial_scale_max'])
        if 'saturation_min' in params:
            self.saturation_min_spin.setValue(params['saturation_min'])
        if 'saturation_max' in params:
            self.saturation_max_spin.setValue(params['saturation_max'])
        if 'gamma_min' in params:
            self.gamma_min_spin.setValue(params['gamma_min'])
        if 'gamma_max' in params:
            self.gamma_max_spin.setValue(params['gamma_max'])
        if 'do_flip' in params:
            self.do_flip_combo.setCurrentText(params['do_flip'])
        if 'noyjitter' in params:
            self.noyjitter_cb.setChecked(params['noyjitter'])
    
    def update_image_size_options(self, detected_sizes):
        """根據檢測到的圖像尺寸動態更新圖像尺寸選項"""
        try:
            # 清空現有選項
            self.image_size_combo.clear()
            
            # 添加檢測到的尺寸選項
            if detected_sizes:
                for i, (width, height) in enumerate(detected_sizes):
                    if i == 0:
                        # 第一個尺寸設為推薦選項
                        self.image_size_combo.addItem(f"{width}x{height} (檢測到 - 推薦)")
                    else:
                        self.image_size_combo.addItem(f"{width}x{height} (檢測到)")
                
                # 添加分隔線
                self.image_size_combo.addItem("───────────────")
            
            # 添加標準尺寸選項
            standard_sizes = [
                ("320x720", "320x720 (默認 Default)"),
                ("640x480", "640x480 (原始尺寸 Original)"),
                ("256x512", "256x512"),
                ("384x768", "384x768"), 
                ("512x1024", "512x1024"),
                ("640x1280", "640x1280")
            ]
            
            for size_value, size_text in standard_sizes:
                self.image_size_combo.addItem(size_text)
            
            # 如果沒有檢測到尺寸，選擇默認值
            if not detected_sizes:
                self.image_size_combo.setCurrentText("320x720 (默認 Default)")
            else:
                # 選擇第一個檢測到的尺寸
                self.image_size_combo.setCurrentIndex(0)
                
        except Exception as e:
            self.log(f"[ERROR] 更新圖像尺寸選項時發生錯誤: {e}")
            # 如果出錯，恢復默認選項
            self.image_size_combo.clear()
            self.image_size_combo.addItems([
                "320x720 (默認 Default)",
                "640x480 (原始尺寸 Original)",
                "256x512",
                "384x768", 
                "512x1024",
                "640x1280"
            ])
            self.image_size_combo.setCurrentText("640x480 (原始尺寸 Original)")
    
    def parse_image_size(self, size_text):
        """解析圖像尺寸文本，返回 [width, height] 格式"""
        try:
            # 提取尺寸部分（例如：從 "320x720 (檢測到 - 推薦)" 提取 "320x720"）
            size_part = size_text.split(' ')[0]  # 取第一個空格前的部分
            if 'x' in size_part:
                width, height = size_part.split('x')
                return [int(width), int(height)]
            else:
                # 如果解析失敗，返回默認尺寸
                return [320, 720]
        except Exception as e:
            self.log(f"[WARNING] 解析圖像尺寸失敗: {size_text}, 使用默認尺寸")
            return [320, 720]


class StereoTrainingModule(BaseModule):
    """立体视觉训练功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.advanced_params = {}  # 存儲高級參數（將在 load_settings 時填充）
        self.available_datasets = []  # 存儲檢測到的資料集
        
        # 不再在初始化時載入默認值，改為在 load_settings 時從配置文件載入，缺漏的再用默認值
    
    def _get_default_params_from_config(self):
        """從 config.py 獲取默認參數（不直接設置 self.advanced_params，僅返回字典）"""
        try:
            # 添加項目根目錄到 Python 路徑（config/config.py 在根目錄）
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # 嘗試從 config.config 載入 RAFT_STEREO_CONFIG
            try:
                from config.config import RAFT_STEREO_CONFIG
                
                # 將配置轉換為 advanced_params 格式
                default_params = {
                    'corr_implementation': RAFT_STEREO_CONFIG.get('corr_implementation', 'reg'),
                    'n_downsample': RAFT_STEREO_CONFIG.get('n_downsample', 2),
                    'corr_levels': RAFT_STEREO_CONFIG.get('corr_levels', 4),
                    'corr_radius': RAFT_STEREO_CONFIG.get('corr_radius', 4),
                    'n_gru_layers': RAFT_STEREO_CONFIG.get('n_gru_layers', 3),
                    'shared_backbone': RAFT_STEREO_CONFIG.get('shared_backbone', False),
                    'context_norm': RAFT_STEREO_CONFIG.get('context_norm', 'batch'),
                    'slow_fast_gru': RAFT_STEREO_CONFIG.get('slow_fast_gru', False),
                    'hidden_dims': f"{RAFT_STEREO_CONFIG.get('hidden_dims', [128, 128, 128])[0]}x{RAFT_STEREO_CONFIG.get('hidden_dims', [128, 128, 128])[1]}x{RAFT_STEREO_CONFIG.get('hidden_dims', [128, 128, 128])[2]} (默認)",
                    'mixed_precision': RAFT_STEREO_CONFIG.get('mixed_precision', False),
                    'weight_decay': RAFT_STEREO_CONFIG.get('wdecay', 0.00001),
                    'train_iters': RAFT_STEREO_CONFIG.get('train_iters', 16),
                    'valid_iters': RAFT_STEREO_CONFIG.get('valid_iters', 32),
                    'learning_rate': RAFT_STEREO_CONFIG.get('lr', 0.0002),
                    'image_size': list(RAFT_STEREO_CONFIG.get('image_size', (320, 720))),
                    # spatial_scale 在 config.py 中是 (0.0, 0.0)，但我們使用代碼中的默認值
                    'spatial_scale_min': -0.2,  # 默認值
                    'spatial_scale_max': 0.4,  # 默認值
                    'saturation_min': 0.0,  # 默認值
                    'saturation_max': 1.4,  # 默認值
                    'gamma_min': 0.8,  # 默認值
                    'gamma_max': 1.2,  # 默認值
                    'do_flip': '無 None',  # 默認值
                    'noyjitter': RAFT_STEREO_CONFIG.get('noyjitter', False)
                }
                
                return default_params
            except ImportError:
                self.log("[WARNING] 無法從 config.config 載入默認參數，使用本地默認值")
                # 使用本地默認值
                return self._get_local_default_params()
        except Exception as e:
            self.log(f"[WARNING] 載入默認參數失敗: {e}")
            return self._get_local_default_params()
    
    def _get_local_default_params(self):
        """獲取本地默認參數"""
        return {
            'corr_implementation': 'reg',
            'n_downsample': 2,
            'corr_levels': 4,
            'corr_radius': 4,
            'n_gru_layers': 3,
            'shared_backbone': False,
            'context_norm': 'batch',
            'slow_fast_gru': False,
            'hidden_dims': '128x128x128 (默認)',
            'mixed_precision': False,
            'weight_decay': 0.00001,
            'train_iters': 16,
            'valid_iters': 32,
            'learning_rate': 0.0002,
            'image_size': [640,480],
            'spatial_scale_min': -0.2,
            'spatial_scale_max': 0.4,
            'saturation_min': 0.0,
            'saturation_max': 1.4,
            'gamma_min': 0.8,
            'gamma_max': 1.2,
            'do_flip': '無 None',
            'noyjitter': False
        }
    
    def _get_initial_basic_params(self):
        """獲取基本參數的初始值（優先從配置文件，否則使用 config.py 默認值）"""
        try:
            # 嘗試從父窗口的 settings_manager 讀取配置
            if hasattr(self, 'parent') and self.parent:
                if hasattr(self.parent, 'settings_manager'):
                    stereo_settings = self.parent.settings_manager.get_section('stereo_training')
                    if not stereo_settings:
                        stereo_settings = {}
                    if stereo_settings:
                        batch_size = stereo_settings.get('batch_size')
                        num_steps = stereo_settings.get('num_steps')
                        if batch_size is not None and num_steps is not None:
                            return batch_size, num_steps
        except:
            pass
        
        # 如果無法從配置文件讀取，使用 config.py 的默認值
        try:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from config.config import STEREO_TRAIN_GUI_DEFAULTS
            return (
                STEREO_TRAIN_GUI_DEFAULTS.get('batch_size', 6),
                STEREO_TRAIN_GUI_DEFAULTS.get('num_steps', 100000)
            )
        except:
            # 最終後備默認值
            return 6, 100000
    
    def create_tab(self):
        """创建立体视觉训练标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 数据集选择
        dataset_group = QGroupBox("數據集設置")
        dataset_layout = QGridLayout(dataset_group)
        
        # 自動檢測的資料集選擇
        dataset_layout.addWidget(QLabel("可用資料集:"), 0, 0)
        self.stereo_dataset_combo = QComboBox()
        self.stereo_dataset_combo.setToolTip("選擇立體視覺資料集")
        dataset_layout.addWidget(self.stereo_dataset_combo, 0, 1)
        
        self.refresh_datasets_btn = QPushButton("🔄 刷新")
        self.refresh_datasets_btn.clicked.connect(self.refresh_stereo_datasets)
        self.refresh_datasets_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        dataset_layout.addWidget(self.refresh_datasets_btn, 0, 2)
        
        # 手動選擇資料集路徑
        dataset_layout.addWidget(QLabel("或手動選擇:"), 1, 0)
        self.stereo_dataset_edit = QLineEdit()
        self.stereo_dataset_edit.setPlaceholderText("選擇立體視覺數據集路徑")
        dataset_layout.addWidget(self.stereo_dataset_edit, 1, 1)
        
        self.stereo_dataset_btn = QPushButton("瀏覽")
        self.stereo_dataset_btn.clicked.connect(self.browse_stereo_dataset)
        dataset_layout.addWidget(self.stereo_dataset_btn, 1, 2)
        
        # 資料集信息顯示
        self.dataset_info_label = QLabel("")
        self.dataset_info_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_info_label, 2, 0, 1, 3)
        
        # 連接信號
        self.stereo_dataset_combo.currentTextChanged.connect(self.on_dataset_selected)
        self.stereo_dataset_edit.textChanged.connect(self.on_manual_dataset_changed)
        
        layout.addWidget(dataset_group)
        
        # 模型选择
        model_group = QGroupBox("模型設置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("預訓練模型:"), 0, 0)
        self.stereo_model_combo = QComboBox()
        self.stereo_model_combo.setToolTip("將從 Model_file/Stereo_Vision 搜索對應檔案；亦可提供完整路徑")
        model_layout.addWidget(self.stereo_model_combo, 0, 1)
        
        # 添加刷新按鈕
        self.refresh_stereo_models_btn = QPushButton("🔄 刷新模型")
        self.refresh_stereo_models_btn.clicked.connect(self.refresh_stereo_model_list)
        self.refresh_stereo_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        model_layout.addWidget(self.refresh_stereo_models_btn, 0, 2)
        
        layout.addWidget(model_group)
        
        # 基本訓練參數
        params_group = QGroupBox("基本訓練參數 Basic Training Parameters")
        params_layout = QGridLayout(params_group)
        
        # 獲取初始值（優先從配置文件，否則使用 config.py 默認值）
        initial_batch_size, initial_num_steps = self._get_initial_basic_params()
        
        params_layout.addWidget(QLabel("批次大小 Batch Size:"), 0, 0)
        self.stereo_batch_spin = QSpinBox()
        self.stereo_batch_spin.setRange(1, 32)
        self.stereo_batch_spin.setValue(initial_batch_size)
        self.stereo_batch_spin.setToolTip("訓練批次大小 Training batch size")
        params_layout.addWidget(self.stereo_batch_spin, 0, 1)
        
        params_layout.addWidget(QLabel("訓練步數 Num Steps:"), 0, 2)
        self.stereo_num_steps_spin = QSpinBox()
        self.stereo_num_steps_spin.setRange(1, 1000000)
        self.stereo_num_steps_spin.setValue(initial_num_steps)
        self.stereo_num_steps_spin.setToolTip("總訓練步數 Total number of training steps")
        params_layout.addWidget(self.stereo_num_steps_spin, 0, 3)
        
        # 高級參數按鈕
        self.advanced_params_btn = QPushButton("⚙️ 高級參數")
        self.advanced_params_btn.clicked.connect(self.open_advanced_params)
        self.advanced_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        params_layout.addWidget(self.advanced_params_btn, 1, 0, 1, 4)
        
        layout.addWidget(params_group)
        
        # 輸出設置
        output_group = QGroupBox("輸出設置")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出目錄:"), 0, 0)
        self.stereo_output_edit = QLineEdit()
        self.stereo_output_edit.setPlaceholderText("留空使用默認路徑 (checkpoints)")
        output_layout.addWidget(self.stereo_output_edit, 1, 0)
        
        layout.addWidget(output_group)
        
        # 訓練說明
        info_group = QGroupBox("💡 訓練說明")
        info_group.setStyleSheet("QGroupBox { padding-top: 5px; }")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(10, 0, 10, 10)
        
        info_text = QTextEdit()
        info_text.setPlainText("""📌 快速開始指南

▶ 數據集要求
  • 資料夾結構：Img0/（左圖）、Img1/（右圖）、Disparity/（視差圖）
  • 每個資料夾需包含 train/、val/、test/ 子目錄

▶ 基本參數說明
  • 批次大小：建議 4-8（取決於GPU記憶體）
  • 訓練步數：建議 50,000-200,000 步
  • 圖像尺寸：點擊"⚙️ 高級參數"可根據資料集自動檢測並調整

▶ 預訓練模型選擇
  • sceneflow：通用場景（推薦新手使用）
  • middlebury：室內高精度
  • eth3d：戶外場景
  • realtime：即時處理優化版本

▶ 進階設置
  點擊"⚙️ 高級參數"可調整：
  • 模型架構參數（相關層數、GRU層數等）
  • 優化參數（學習率、權重衰減、混合精度等）
  • 數據增廣參數（縮放、顏色變換、翻轉等）

💾 輸出位置：checkpoints/ 目錄（可在上方自訂）
📊 訓練日誌：可在"📋 運行日誌"標籤頁查看即時進度""")
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(280)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 训练控制
        control_group = QGroupBox("訓練控制")
        control_layout = QHBoxLayout(control_group)
        
        self.stereo_start_btn = QPushButton("🚀 開始訓練")
        self.stereo_start_btn.clicked.connect(self.start_stereo_training)
        self.stereo_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e55a25;
            }
        """)
        control_layout.addWidget(self.stereo_start_btn)
        
        self.stereo_stop_btn = QPushButton("⏹️ 停止訓練")
        self.stereo_stop_btn.clicked.connect(self.stop_stereo_training)
        self.stereo_stop_btn.setEnabled(False)
        self.stereo_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        control_layout.addWidget(self.stereo_stop_btn)
        
        self.stereo_clear_btn = QPushButton("🗑️ 清空設置")
        self.stereo_clear_btn.clicked.connect(self.clear_stereo_settings)
        control_layout.addWidget(self.stereo_clear_btn)
        
        layout.addWidget(control_group)
        
        # 初始化時自動檢測資料集
        self.refresh_stereo_datasets()
        
        # 初始化時自動檢測模型
        self.refresh_stereo_model_list()
        
        # 標記標籤頁已創建，可以載入設置
        self.tab_created = True
        
        # 如果有待載入的設置，現在載入它們
        if hasattr(self, '_pending_settings') and self._pending_settings:
            self.log("🔄 載入待處理的設置...")
            self._load_pending_settings()
        
        self.tab_widget = tab
        return tab
    
    def _load_pending_settings(self):
        """載入待處理的設置"""
        try:
            stereo_settings = self._pending_settings
            
            # 載入基本參數
            if hasattr(self, 'stereo_dataset_edit') and 'dataset_path' in stereo_settings:
                self.stereo_dataset_edit.setText(stereo_settings['dataset_path'])
            if hasattr(self, 'stereo_model_combo') and 'model_name' in stereo_settings:
                self.stereo_model_combo.setCurrentText(stereo_settings['model_name'])
            if hasattr(self, 'stereo_batch_spin') and 'batch_size' in stereo_settings:
                self.stereo_batch_spin.setValue(stereo_settings['batch_size'])
            
            # 載入訓練步數 num_steps
            if hasattr(self, 'stereo_num_steps_spin'):
                if 'num_steps' in stereo_settings:
                    self.stereo_num_steps_spin.setValue(stereo_settings['num_steps'])
            
            if hasattr(self, 'stereo_output_edit') and 'output_path' in stereo_settings:
                self.stereo_output_edit.setText(stereo_settings['output_path'])
            
            self.log("✅ 待處理的立體視覺基本參數已載入")
            
            # 清除待處理設置
            self._pending_settings = None
            
        except Exception as e:
            self.log(f"[ERROR] 載入待處理設置時發生錯誤: {e}")
    
    def open_advanced_params(self):
        """打開高級參數設置對話框"""
        dialog = StereoParamsDialog(self.parent)
        
        # 獲取當前選擇的資料集信息
        dataset_path = self.stereo_dataset_edit.text()
        if dataset_path:
            try:
                info = self.get_stereo_dataset_info(dataset_path)
                if 'image_sizes' in info and info['image_sizes']:
                    # 根據檢測到的圖像尺寸更新選項
                    dialog.update_image_size_options(info['image_sizes'])
                    self.log(f"✅ 已根據資料集圖像尺寸更新選項: {info['image_sizes']}")
            except Exception as e:
                self.log(f"[WARNING] 無法獲取資料集圖像尺寸: {e}")
        
        # 設置當前參數值（使用當前 advanced_params）
        # 如果 advanced_params 不存在或為空，先嘗試從配置文件加載
        if not hasattr(self, 'advanced_params') or not self.advanced_params:
            try:
                if hasattr(self, 'parent') and self.parent and hasattr(self.parent, 'settings_manager'):
                    stereo_settings_temp = self.parent.settings_manager.get_section('stereo_training')
                    if stereo_settings_temp and 'advanced_params' in stereo_settings_temp and stereo_settings_temp['advanced_params']:
                        self.advanced_params = stereo_settings_temp['advanced_params'].copy()
                        # 處理 image_size 格式
                        if 'image_size' in self.advanced_params:
                            img_size = self.advanced_params['image_size']
                            if isinstance(img_size, str) and 'x' in img_size:
                                width, height = img_size.split('x')
                                self.advanced_params['image_size'] = [int(width), int(height)]
                        self.log("✅ 已從配置文件加載高級參數到對話框")
            except Exception as e:
                self.log(f"[WARNING] 從配置文件加載高級參數失敗: {e}")
        
        # 設置對話框參數
        if hasattr(self, 'advanced_params') and self.advanced_params:
            dialog.set_all_params(self.advanced_params)
            self.log(f"✅ 已設置對話框參數（共 {len(self.advanced_params)} 個參數）")
        else:
            # 如果 advanced_params 仍然為空，使用空字典（對話框會使用默認值）
            dialog.set_all_params({})
            self.log("ℹ️ advanced_params 為空，對話框使用默認值")
        
        # 執行對話框
        result = dialog.exec_()
        
        # 如果用戶點擊確定，直接從對話框獲取參數並更新
        if result == QDialog.Accepted:
            try:
                dialog_params = dialog.get_all_params()
                if dialog_params:
                    # 確保 advanced_params 存在
                    if not hasattr(self, 'advanced_params') or not isinstance(self.advanced_params, dict):
                        self.advanced_params = {}
                    
                    # 更新參數（使用完整的參數字典替換，而不是 update）
                    self.advanced_params = dialog_params.copy()
                    self.log(f"✅ 高級參數已確認並更新（共 {len(self.advanced_params)} 個參數） Advanced parameters confirmed and updated")
                    self.log(f"   主要參數: corr_implementation={self.advanced_params.get('corr_implementation')}, "
                           f"learning_rate={self.advanced_params.get('learning_rate')}, "
                           f"train_iters={self.advanced_params.get('train_iters')}")
                else:
                    self.log("[WARNING] 對話框返回的參數為空")
            except Exception as e:
                self.log(f"[WARNING] 獲取對話框參數失敗: {e}")
                import traceback
                self.log(f"   詳細錯誤: {traceback.format_exc()}")
        else:
            self.log("ℹ️ 高級參數設置已取消 Advanced parameters dialog cancelled")
    
    def start_stereo_training(self):
        """开始立体视觉训练"""
        # 驗證數據集路徑
        dataset_path = self.stereo_dataset_edit.text()
        if not dataset_path:
            self.log("[WARNING] 請選擇訓練數據集")
            QMessageBox.warning(self.parent, "警告 Warning", "請選擇訓練數據集")
            return
        
        # 获取训练参数
        model_name = self.stereo_model_combo.currentText().strip()
        
        # 檢查模型是否有效
        if not model_name or model_name.startswith("("):
            self.log("[WARNING] 請先放置預訓練模型文件")
            QMessageBox.warning(
                self.parent, 
                "警告 Warning", 
                "未找到可用的預訓練模型！\n\n請將模型文件放置在 Model_file/Stereo_Vision 目錄下\n然後點擊「🔄 刷新模型」按鈕。"
            )
            return
        
        # 禁用按鈕
        self.stereo_start_btn.setEnabled(False)
        self.stereo_stop_btn.setEnabled(True)
        
        batch_size = self.stereo_batch_spin.value()
        num_steps = self.stereo_num_steps_spin.value()
        
        # 初始化进度条：设置总数和初始值
        self.show_progress(True, current=0, total=num_steps, text="準備訓練 Preparing training...")
        output_dir = self.stereo_output_edit.text() if self.stereo_output_edit.text() else "checkpoints"
        
        # 從高級參數中獲取詳細設置
        train_iters = self.advanced_params.get('train_iters', 16)
        valid_iters = self.advanced_params.get('valid_iters', 32)
        corr_implementation = self.advanced_params.get('corr_implementation', 'reg')
        mixed_precision = self.advanced_params.get('mixed_precision', False)
        n_downsample = self.advanced_params.get('n_downsample', 2)
        corr_levels = self.advanced_params.get('corr_levels', 4)
        corr_radius = self.advanced_params.get('corr_radius', 4)
        n_gru_layers = self.advanced_params.get('n_gru_layers', 3)
        learning_rate = self.advanced_params.get('learning_rate', 0.0002)
        weight_decay = self.advanced_params.get('weight_decay', 0.00001)
        hidden_dims = self.advanced_params.get('hidden_dims', '128x128x128 (默認)')  # 添加 hidden_dims 提取
        
        # 獲取圖像尺寸設置（默認使用原始尺寸640x480）
        image_size = self.advanced_params.get('image_size', [640, 480])
        if isinstance(image_size, str):
            # 如果是字符串格式，解析為列表
            try:
                if 'x' in image_size:
                    width, height = image_size.split('x')
                    image_size = [int(width), int(height)]
                else:
                    image_size = [640, 480]  # 默認使用原始尺寸
            except:
                image_size = [640, 480]  # 默認使用原始尺寸
        
        self.log(f"🚀 開始立體視覺訓練")
        self.log(f"   數據集: {dataset_path}")
        self.log(f"   預訓練模型: {model_name}")
        self.log(f"   訓練參數: 步數={num_steps}, 批次={batch_size}")
        self.log(f"   迭代參數: 訓練={train_iters}, 驗證={valid_iters}")
        self.log(f"   圖像尺寸: {image_size[0]}x{image_size[1]} (width x height, 將轉換為 height x width)")
        self.log(f"   相關實現: {corr_implementation}")
        self.log(f"   模型架構: n_downsample={n_downsample}, corr_levels={corr_levels}, corr_radius={corr_radius}")
        self.log(f"   GRU層數: {n_gru_layers}, Hidden Dims: {hidden_dims}")
        self.log(f"   優化選項: 混合精度={mixed_precision}, 學習率={learning_rate}, 權重衰減={weight_decay}")
        
        # 記錄完整的訓練參數
        self.log_training_parameters({
            'dataset_path': dataset_path,
            'model_name': model_name,
            'batch_size': batch_size,
            'num_steps': num_steps,
            'train_iters': train_iters,
            'valid_iters': valid_iters,
            'corr_implementation': corr_implementation,
            'mixed_precision': mixed_precision,
            'n_downsample': n_downsample,
            'corr_levels': corr_levels,
            'corr_radius': corr_radius,
            'n_gru_layers': n_gru_layers,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'image_size': image_size,
            'spatial_scale_min': self.advanced_params.get('spatial_scale_min', -0.2),
            'spatial_scale_max': self.advanced_params.get('spatial_scale_max', 0.4),
            'saturation_min': self.advanced_params.get('saturation_min', 0.0),
            'saturation_max': self.advanced_params.get('saturation_max', 1.4),
            'gamma_min': self.advanced_params.get('gamma_min', 0.8),
            'gamma_max': self.advanced_params.get('gamma_max', 1.2),
            'do_flip': self.advanced_params.get('do_flip', '無 None'),
            'noyjitter': self.advanced_params.get('noyjitter', False),
            'output_dir': output_dir
        })
        
        # 使用統一的WorkerThread執行訓練
        self.start_stereo_worker_thread(
            dataset_path=dataset_path,
            model_name=model_name,
            batch_size=batch_size,
            num_steps=num_steps,
            train_iters=train_iters,
            valid_iters=valid_iters,
            output_dir=output_dir,
            corr_implementation=corr_implementation,
            mixed_precision=mixed_precision,
            n_downsample=n_downsample,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            n_gru_layers=n_gru_layers,
            hidden_dims=hidden_dims,  # 添加 hidden_dims 參數
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            image_size=image_size,
            spatial_scale_min=self.advanced_params.get('spatial_scale_min', -0.2),
            spatial_scale_max=self.advanced_params.get('spatial_scale_max', 0.4),
            saturation_min=self.advanced_params.get('saturation_min', 0.0),
            saturation_max=self.advanced_params.get('saturation_max', 1.4),
            gamma_min=self.advanced_params.get('gamma_min', 0.8),
            gamma_max=self.advanced_params.get('gamma_max', 1.2),
            do_flip=self.advanced_params.get('do_flip', '無 None'),
            noyjitter=self.advanced_params.get('noyjitter', False)
        )
    
    def start_stereo_worker_thread(self, **kwargs):
        """啟動立體視覺訓練工作線程"""
        try:
            # 導入WorkerThread
            from gui.workers.worker_thread import WorkerThread
            import traceback
            
            self.log("🔄 正在創建工作線程... Creating worker thread...")
            
            # 創建工作線程
            self.worker_thread = WorkerThread(task_type="train_stereo", **kwargs)
            
            # 連接信號
            self.worker_thread.progress.connect(self.on_stereo_progress)
            self.worker_thread.finished.connect(self.on_stereo_training_finished)
            self.worker_thread.log_message.connect(self.log)
            self.worker_thread.epoch_progress.connect(self.on_stereo_epoch_progress)
            
            self.log("🔄 正在啟動工作線程... Starting worker thread...")
            
            # 啟動線程
            self.worker_thread.start()
            
            self.log("✅ 立體視覺訓練工作線程啟動命令已發送 Worker thread start command sent")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = f"啟動立體視覺訓練工作線程失敗: {e}"
            self.log(f"[ERROR] {error_msg}")
            self.log(f"[ERROR] 詳細錯誤信息 Detailed error:")
            self.log(error_detail)
            
            # 確保按鈕狀態正確恢復
            self.stereo_start_btn.setEnabled(True)
            self.stereo_stop_btn.setEnabled(False)
            self.show_progress(False)
            
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"啟動訓練失敗 Failed to start training:\n{error_msg}\n\n詳細信息 See log for details"
            )
    
    def on_stereo_progress(self, message):
        """立體視覺訓練進度回調"""
        self.log(message)
        
        # 嘗試從消息中解析進度信息
        # 格式可能類似：Step 1000/200000, Loss: 0.5
        import re
        # 匹配步數信息
        step_match = re.search(r'(\d+)\s*/\s*(\d+)', message)
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))
            # 更新進度條
            self.show_progress(True, current=current, total=total, text=f"訓練中 Training... ({current}/{total})")
    
    def on_stereo_epoch_progress(self, current, total, text):
        """立體視覺訓練輪次進度回調"""
        self.log(f"Epoch {current}/{total}: {text}")
        
        # 更新進度條（如果是步數進度）
        if total > 0:
            # 嘗試從 text 中提取步數信息
            import re
            step_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if step_match:
                step_current = int(step_match.group(1))
                step_total = int(step_match.group(2))
                self.show_progress(True, current=step_current, total=step_total, text=f"訓練中 Training... ({step_current}/{step_total})")
    
    def log_training_parameters(self, params):
        """記錄訓練參數到日誌和文件"""
        import json
        from datetime import datetime
        
        # 創建參數記錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        param_record = {
            'timestamp': timestamp,
            'training_type': 'stereo_vision',
            'parameters': params
        }
        
        # 記錄到日誌
        self.log("=" * 60)
        self.log("📋 訓練參數記錄 Training Parameters Log")
        self.log("=" * 60)
        for key, value in params.items():
            self.log(f"   {key}: {value}")
        self.log("=" * 60)
        
        # 保存到文件
        try:
            import os
            os.makedirs("training_logs", exist_ok=True)
            log_file = f"training_logs/stereo_training_params_{timestamp}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(param_record, f, indent=2, ensure_ascii=False)
            self.log(f"✅ 參數已保存到: {log_file}")
        except Exception as e:
            self.log(f"⚠️ 保存參數文件失敗: {e}")
    
    def run_training(self, dataset_path, model_name, batch_size, num_steps, train_iters, valid_iters, 
                    output_dir, corr_implementation, mixed_precision, n_downsample, corr_levels, 
                    corr_radius, n_gru_layers, learning_rate, weight_decay, image_size, 
                    spatial_scale_min, spatial_scale_max, saturation_min, saturation_max, 
                    gamma_min, gamma_max, do_flip, noyjitter):
        """Run the raft_stereo_trainer using object-oriented approach"""
        import sys
        import os
        from datetime import datetime
        
        # 檢查訓練腳本是否存在
        train_script = "Code/raft_stereo_trainer.py"
        if not os.path.exists(train_script):
            self.log(f"錯誤: 找不到訓練腳本 {train_script}")
            self.log(f"Error: Training script {train_script} not found")
            return False, None
        
        # 創建帶時間戳的輸出資料夾
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        output_folder = f"raft_stereo_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        
        self.log(f"創建輸出資料夾: {output_folder}")
        self.log(f"Created output folder: {output_folder}")
        
        # 導入必要的模組
        try:
            # 添加 Code 目錄到 Python 路徑
            code_dir = Path(__file__).parent.parent.parent / "Code"
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
            
            from config.config import TrainingConfig
            from Code.raft_stereo_trainer import RAFTStereoTrainer
            
        except ImportError as e:
            self.log(f"導入訓練模組失敗: {e}")
            self.log(f"Failed to import training modules: {e}")
            return False, output_folder
        
        # 創建訓練配置
        config = TrainingConfig(
            name=f"raft-stereo-{timestamp}",
            train_datasets=['drone'],
            dataset_root=dataset_path,
            batch_size=batch_size,
            num_steps=num_steps,
            train_iters=train_iters,
            valid_iters=valid_iters,
            lr=learning_rate,
            wdecay=weight_decay,
            image_size=image_size,
            corr_implementation=corr_implementation,
            mixed_precision=mixed_precision,
            n_downsample=n_downsample,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            n_gru_layers=n_gru_layers,
            spatial_scale=(spatial_scale_min, spatial_scale_max),
            saturation_range=[saturation_min, saturation_max] if saturation_min != 0.0 or saturation_max != 1.4 else None,
            img_gamma=[gamma_min, gamma_max] if gamma_min != 0.8 or gamma_max != 1.2 else None,
            do_flip=do_flip if do_flip != '無 None' and do_flip != 'None' else None,
            noyjitter=noyjitter,
            output_dir=output_folder
        )
        
        # 驗證配置
        if not config.validate():
            self.log("配置驗證失敗，請檢查參數設置")
            self.log("Configuration validation failed, please check parameters")
            return False, output_folder
        
        self.log("準備開始訓練...")
        self.log("Prepare to start training...")
        self.log(f"使用配置: {config.name}")
        self.log(f"Using configuration: {config.name}")
        self.log("-" * 50)
        
        try:
            # 設置日誌
            import logging
            logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
            
            # 創建訓練器並執行訓練
            trainer = RAFTStereoTrainer(config)
            result_path = trainer.train()
            
            self.log("-" * 50)
            self.log("訓練完成！")
            self.log("Training completed!")
            self.log(f"模型保存路徑: {result_path}")
            self.log(f"Model saved to: {result_path}")
            
            return True, output_folder
            
        except Exception as e:
            self.log(f"訓練失敗: {e}")
            self.log(f"Training failed: {e}")
            import traceback
            self.log(f"詳細錯誤信息: {traceback.format_exc()}")
            self.log(f"Detailed error: {traceback.format_exc()}")
            return False, output_folder
    
    def stop_stereo_training(self):
        """停止立体视觉训练"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.log("⏹️ 正在停止立體視覺訓練...")
        else:
            self.log("ℹ️ 沒有正在運行的立體視覺訓練")
        
        self.stereo_start_btn.setEnabled(True)
        self.stereo_stop_btn.setEnabled(False)
        self.show_progress(False)
        self.log("⏹️ 立體視覺訓練已停止")
        
    def on_stereo_training_finished(self, success, message):
        """立体视觉训练完成回调"""
        self.stereo_start_btn.setEnabled(True)
        self.stereo_stop_btn.setEnabled(False)
        self.show_progress(False)
        
        if success:
            self.log(f"[SUCCESS] 立體視覺訓練完成: {message}")
            QMessageBox.information(
                self.parent, "成功 Success",
                f"立體視覺訓練完成！\n\n{message}"
            )
        else:
            self.log(f"[ERROR] 立體視覺訓練失敗: {message}")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"立體視覺訓練失敗:\n{message}"
            )
            
    def clear_stereo_settings(self):
        """清空立体视觉设置"""
        self.stereo_dataset_edit.clear()
        self.stereo_dataset_combo.setCurrentIndex(0)
        self.dataset_info_label.setText("")
        self.stereo_model_combo.setCurrentIndex(0)
        self.stereo_batch_spin.setValue(6)
        self.stereo_num_steps_spin.setValue(100000)
        self.stereo_output_edit.clear()
        
        # 清空高級參數
        self.advanced_params = {}
        
        # 重新檢測資料集
        self.refresh_stereo_datasets()
        
        self.log("[INFO] 已清空立體視覺設置")
    
    def load_settings(self, settings_manager):
        """載入立體視覺訓練模組設定"""
        try:
            # 從 gui_settings.yaml 讀取立體視覺訓練設定
            stereo_settings = settings_manager.get_section('stereo_training')
            if not stereo_settings:
                stereo_settings = {}
            
            # 載入基本參數
            try:
                # 確保控件已創建（如果尚未創建，延後載入）
                if not (hasattr(self, 'stereo_batch_spin') and hasattr(self, 'stereo_num_steps_spin')):
                    self.log("⚠️ 控件尚未創建，將在創建後重新載入設置")
                    self._pending_settings = stereo_settings
                    return
                
                if hasattr(self, 'stereo_dataset_edit') and 'dataset_path' in stereo_settings:
                    self.stereo_dataset_edit.setText(stereo_settings['dataset_path'])
                if hasattr(self, 'stereo_model_combo') and 'model_name' in stereo_settings:
                    self.stereo_model_combo.setCurrentText(stereo_settings['model_name'])
                
                # 獲取 config.py 的基本參數默認值（用於填充缺漏的參數）
                project_root = Path(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from config.config import STEREO_TRAIN_GUI_DEFAULTS
                default_batch_size = STEREO_TRAIN_GUI_DEFAULTS.get('batch_size', 6)
                default_num_steps = STEREO_TRAIN_GUI_DEFAULTS.get('num_steps', 100000)
                
                # 載入批次大小：優先從 gui_settings.yaml 讀取，缺漏則使用 config.py 默認值
                if hasattr(self, 'stereo_batch_spin'):
                    if 'batch_size' in stereo_settings:
                        self.stereo_batch_spin.setValue(stereo_settings['batch_size'])
                        self.log(f"✅ 已從 gui_settings.yaml 載入批次大小: {stereo_settings['batch_size']}")
                    else:
                        self.stereo_batch_spin.setValue(default_batch_size)
                        self.log(f"ℹ️ gui_settings.yaml 未包含批次大小，使用 config.py 默認值: {default_batch_size}")
                
                # 載入訓練步數 num_steps
                if hasattr(self, 'stereo_num_steps_spin'):
                    if 'num_steps' in stereo_settings:
                        self.stereo_num_steps_spin.setValue(stereo_settings['num_steps'])
                        self.log(f"✅ 已從 gui_settings.yaml 載入訓練步數: {stereo_settings['num_steps']}")
                    else:
                        self.stereo_num_steps_spin.setValue(default_num_steps)
                        self.log(f"ℹ️ gui_settings.yaml 未包含訓練步數，使用 config.py 默認值: {default_num_steps}")
                
                if hasattr(self, 'stereo_output_edit') and 'output_path' in stereo_settings:
                    self.stereo_output_edit.setText(stereo_settings['output_path'])
                
                self.log("✅ 立體視覺基本參數已載入:")
                self.log(f"   資料集路徑: {stereo_settings.get('dataset_path', '未設置')}")
                self.log(f"   模型名稱: {stereo_settings.get('model_name', '未設置')}")
                self.log(f"   批次大小: {stereo_settings.get('batch_size', '未設置')}")
                self.log(f"   訓練步數: {stereo_settings.get('num_steps', '未設置')}")
                self.log(f"   輸出路徑: {stereo_settings.get('output_path', '未設置')}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 載入基本參數時發生錯誤: {e}")
                self.log("   控件可能尚未創建，將在創建後重新載入")
                # 保存設置以供稍後載入
                self._pending_settings = stereo_settings
            
            # 載入高級參數（優先順序：gui_settings.yaml > config.py 默認值）
            # 先獲取 config.py 的默認值作為基礎
            default_params = self._get_default_params_from_config()
            
            # 從 gui_settings.yaml 載入保存的參數
            loaded_from_file = {}
            
            # 檢查是否有高級參數（即使是空字典也要處理）
            if 'advanced_params' in stereo_settings:
                # 優先使用保存的設置
                if stereo_settings['advanced_params']:
                    loaded_from_file = stereo_settings['advanced_params'].copy()
                else:
                    # 如果是空字典，跳過
                    loaded_from_file = {}
                
                # 處理 image_size 格式（可能是字符串 "320x720" 或列表 [320, 720]）
                if 'image_size' in loaded_from_file:
                    img_size = loaded_from_file['image_size']
                    if isinstance(img_size, str) and 'x' in img_size:
                        width, height = img_size.split('x')
                        loaded_from_file['image_size'] = [int(width), int(height)]
                    elif isinstance(img_size, list) and len(img_size) == 2:
                        loaded_from_file['image_size'] = img_size
                
                self.log("✅ 已從 gui_settings.yaml 載入高級參數")
            else:
                # 載入個別高級參數（向後兼容）
                for key in ['corr_implementation', 'n_downsample', 'corr_levels', 'corr_radius', 
                           'n_gru_layers', 'shared_backbone', 'context_norm', 'slow_fast_gru', 
                           'hidden_dims', 'mixed_precision', 'weight_decay', 'train_iters', 
                           'valid_iters', 'learning_rate', 'image_size', 'spatial_scale_min', 
                           'spatial_scale_max', 'saturation_min', 'saturation_max', 
                           'gamma_min', 'gamma_max', 'do_flip', 'noyjitter']:
                    if key in stereo_settings:
                        loaded_from_file[key] = stereo_settings[key]
                
                if loaded_from_file:
                    self.log("✅ 已從 gui_settings.yaml 載入個別參數（向後兼容）")
            
            # 合併參數：先用 config.py 默認值作為基礎，再用 gui_settings.yaml 的值覆蓋
            self.advanced_params = default_params.copy()
            if loaded_from_file:
                self.advanced_params.update(loaded_from_file)
                missing_keys = set(default_params.keys()) - set(loaded_from_file.keys())
                if missing_keys:
                    self.log(f"ℹ️ 以下參數使用 config.py 默認值: {', '.join(sorted(missing_keys))}")
                self.log(f"✅ 立體視覺高級參數已合併載入（共 {len(self.advanced_params)} 個參數）")
            else:
                self.log("ℹ️ gui_settings.yaml 中未找到高級參數，使用 config.py 默認值")
                self.log(f"✅ 立體視覺高級參數已從 config.py 載入（共 {len(self.advanced_params)} 個參數）")
            
            self.log("✅ 立體視覺訓練設定載入完成")
            
        except Exception as e:
            self.log(f"[WARNING] 載入立體視覺訓練設定失敗: {e}")
            import traceback
            self.log(f"   詳細錯誤: {traceback.format_exc()}")
    
    def save_settings(self, settings_manager):
        """保存立體視覺訓練模組設定"""
        try:
            # 如果控件尚未創建，不保存設置（避免保存默認值覆蓋配置文件）
            if not (hasattr(self, 'stereo_batch_spin') and hasattr(self, 'stereo_num_steps_spin')):
                self.log("⚠️ 控件尚未創建，跳過保存設置（避免覆蓋配置文件）")
                return
            
            stereo_settings = {}
            
            # 保存基本參數 - 必須從控件讀取當前值
            try:
                # 使用 hasattr 檢查控件是否存在
                if hasattr(self, 'stereo_dataset_edit'):
                    stereo_settings['dataset_path'] = self.stereo_dataset_edit.text()
                else:
                    stereo_settings['dataset_path'] = ""
                    
                if hasattr(self, 'stereo_model_combo'):
                    stereo_settings['model_name'] = self.stereo_model_combo.currentText()
                else:
                    stereo_settings['model_name'] = 'raftstereo-sceneflow.pth'
                    
                # 確保從控件讀取當前值（這是關鍵：必須從控件讀取，不是默認值）
                if hasattr(self, 'stereo_batch_spin') and self.stereo_batch_spin is not None:
                    batch_value = self.stereo_batch_spin.value()
                    stereo_settings['batch_size'] = batch_value
                    self.log(f"📝 從控件讀取批次大小: {batch_value}")
                else:
                    stereo_settings['batch_size'] = 6
                    self.log("[WARNING] stereo_batch_spin 控件不存在或為 None，使用默認值 6")
                    
                if hasattr(self, 'stereo_num_steps_spin') and self.stereo_num_steps_spin is not None:
                    steps_value = self.stereo_num_steps_spin.value()
                    stereo_settings['num_steps'] = steps_value
                    self.log(f"📝 從控件讀取訓練步數: {steps_value}")
                else:
                    stereo_settings['num_steps'] = 100000
                    self.log("[WARNING] stereo_num_steps_spin 控件不存在或為 None，使用默認值 100000")
                    
                if hasattr(self, 'stereo_output_edit'):
                    stereo_settings['output_path'] = self.stereo_output_edit.text()
                else:
                    stereo_settings['output_path'] = ""
                
                # 記錄保存的基本參數（用於驗證）
                self.log("✅ 立體視覺基本參數已準備保存:")
                self.log(f"   資料集路徑: {stereo_settings['dataset_path']}")
                self.log(f"   模型名稱: {stereo_settings['model_name']}")
                self.log(f"   批次大小: {stereo_settings['batch_size']} (從控件: {hasattr(self, 'stereo_batch_spin')})")
                self.log(f"   訓練步數: {stereo_settings['num_steps']} (從控件: {hasattr(self, 'stereo_num_steps_spin')})")
                self.log(f"   輸出路徑: {stereo_settings['output_path']}")
                
            except Exception as e:
                self.log(f"[ERROR] 保存基本參數時發生錯誤: {e}")
                import traceback
                self.log(f"   詳細錯誤: {traceback.format_exc()}")
                # 如果控件不存在，設置默認值
                stereo_settings['dataset_path'] = ""
                stereo_settings['model_name'] = 'raftstereo-sceneflow.pth'
                stereo_settings['batch_size'] = 6
                stereo_settings['num_steps'] = 100000
                stereo_settings['output_path'] = ""
            
            # 保存高級參數（確保 advanced_params 存在且為字典）
            if not hasattr(self, 'advanced_params') or not isinstance(self.advanced_params, dict):
                # 如果 advanced_params 不存在或不是字典，嘗試從配置文件加載
                self.log("⚠️ advanced_params 不存在或格式錯誤，嘗試從配置文件加載")
                try:
                    stereo_settings_temp = settings_manager.get_section('stereo_training')
                    if stereo_settings_temp and 'advanced_params' in stereo_settings_temp:
                        self.advanced_params = stereo_settings_temp['advanced_params'].copy()
                        self.log("✅ 已從配置文件加載 advanced_params")
                    else:
                        # 如果配置文件中也沒有，使用默認值
                        self.advanced_params = self._get_default_params_from_config()
                        self.log("ℹ️ 使用 config.py 默認值")
                except Exception as e:
                    self.log(f"[WARNING] 從配置文件加載 advanced_params 失敗: {e}")
                    self.advanced_params = self._get_default_params_from_config()
            
            # 如果 advanced_params 為空字典，嘗試從 config.py 獲取默認值
            if not self.advanced_params:
                self.log("ℹ️ advanced_params 為空，使用 config.py 默認值")
                self.advanced_params = self._get_default_params_from_config()
                
            # 確保 advanced_params 是完整的（如果仍然為空，使用本地默認值）
            if not self.advanced_params:
                self.log("ℹ️ 使用本地默認值")
                self.advanced_params = self._get_local_default_params()
            
            # 保存高級參數（序列化 image_size 為字符串格式以便 YAML 保存）
            saved_advanced_params = self.advanced_params.copy()
            if 'image_size' in saved_advanced_params and isinstance(saved_advanced_params['image_size'], list):
                saved_advanced_params['image_size'] = f"{saved_advanced_params['image_size'][0]}x{saved_advanced_params['image_size'][1]}"
            
            stereo_settings['advanced_params'] = saved_advanced_params
            
            if saved_advanced_params:
                self.log("✅ 立體視覺高級參數已保存:")
                self.log(f"   共 {len(saved_advanced_params)} 個參數")
            else:
                self.log("ℹ️ 立體視覺高級參數為空，已保存默認值")
            
            # 保存到設定管理器
            settings_manager.set_section('stereo_training', stereo_settings)
            
            # 驗證保存的值（用於調試）
            self.log("=" * 60)
            self.log("📋 保存的立體視覺訓練設定摘要:")
            self.log(f"   batch_size: {stereo_settings.get('batch_size', 'N/A')}")
            self.log(f"   num_steps: {stereo_settings.get('num_steps', 'N/A')}")
            self.log(f"   model_name: {stereo_settings.get('model_name', 'N/A')}")
            self.log(f"   dataset_path: {stereo_settings.get('dataset_path', 'N/A')}")
            self.log(f"   advanced_params 數量: {len(stereo_settings.get('advanced_params', {}))}")
            self.log("=" * 60)
            
            self.log("✅ 立體視覺訓練設定已保存到 settings_manager")
            
        except Exception as e:
            self.log(f"[WARNING] 保存立體視覺訓練設定失敗: {e}")
            import traceback
            self.log(f"   詳細錯誤: {traceback.format_exc()}")

    def browse_stereo_dataset(self):
        """瀏覽立體視覺數據集資料夾"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇立體視覺數據集資料夾"
        )
        if folder_path:
            # 檢查是否為有效的立體視覺數據集結構
            from pathlib import Path
            
            if self.is_stereo_dataset(Path(folder_path)):
                self.stereo_dataset_edit.setText(folder_path)
                
                # 獲取並顯示資料集信息
                info = self.get_stereo_dataset_info(folder_path)
                info_text = f"資料集: {info['name']}\n"
                info_text += f"訓練樣本: {info['train_samples']}\n"
                info_text += f"驗證樣本: {info['val_samples']}\n"
                info_text += f"測試樣本: {info['test_samples']}\n"
                info_text += f"總樣本: {info['total_samples']}"
                
                self.dataset_info_label.setText(info_text)
                self.log(f"✅ 立體視覺數據集已選擇: {folder_path}")
                self.log(f"   訓練樣本: {info['train_samples']}, 驗證樣本: {info['val_samples']}, 測試樣本: {info['test_samples']}")
            else:
                QMessageBox.warning(
                    self.parent, 
                    "警告 Warning", 
                    "選擇的資料夾不是有效的立體視覺數據集結構。\n"
                    "請確保資料夾包含 Img0、Img1 和 Disparity 子資料夾。\n\n"
                    "The selected folder is not a valid stereo dataset structure.\n"
                    "Please ensure the folder contains Img0, Img1, and Disparity subfolders."
                )
    
    def refresh_stereo_datasets(self):
        """刷新立體視覺資料集列表"""
        try:
            # 檢測可用的資料集
            self.available_datasets = self.detect_stereo_datasets()
            
            # 清空下拉框
            self.stereo_dataset_combo.clear()
            
            if self.available_datasets:
                for dataset in self.available_datasets:
                    display_name = f"{dataset['name']} - {dataset['description']}"
                    self.stereo_dataset_combo.addItem(display_name, dataset['path'])
                
                self.log(f"✅ 檢測到 {len(self.available_datasets)} 個立體視覺資料集")
                
                # 自動選擇第一個資料集
                if self.available_datasets:
                    self.stereo_dataset_combo.setCurrentIndex(0)
                    self.on_dataset_selected()
            else:
                self.stereo_dataset_combo.addItem("未檢測到立體視覺資料集")
                self.dataset_info_label.setText("未檢測到立體視覺資料集，請手動選擇資料集路徑")
                self.log("⚠️ 未檢測到立體視覺資料集")
                
        except Exception as e:
            self.log(f"[ERROR] 檢測資料集時發生錯誤: {e}")
            self.stereo_dataset_combo.clear()
            self.stereo_dataset_combo.addItem("檢測資料集時發生錯誤")
            self.dataset_info_label.setText(f"檢測資料集時發生錯誤: {e}")
    
    def on_dataset_selected(self):
        """當選擇資料集時的回調"""
        try:
            current_data = self.stereo_dataset_combo.currentData()
            if current_data:
                # 更新手動輸入框
                self.stereo_dataset_edit.setText(current_data)
                
                # 獲取資料集信息
                info = self.get_stereo_dataset_info(current_data)
                
                # 顯示資料集信息
                info_text = f"資料集: {info['name']}\n"
                info_text += f"訓練樣本: {info['train_samples']}\n"
                info_text += f"驗證樣本: {info['val_samples']}\n"
                info_text += f"測試樣本: {info['test_samples']}\n"
                info_text += f"總樣本: {info['total_samples']}\n"
                
                # 添加圖像尺寸信息
                if 'image_sizes' in info and info['image_sizes']:
                    sizes_text = ", ".join([f"{w}x{h}" for w, h in info['image_sizes'][:3]])  # 顯示前3個尺寸
                    if len(info['image_sizes']) > 3:
                        sizes_text += f" (+{len(info['image_sizes'])-3} more)"
                    info_text += f"\n圖像尺寸: {sizes_text}"
                else:
                    info_text += "\n圖像尺寸: 檢測中..."
                
                self.dataset_info_label.setText(info_text)
                self.log(f"✅ 已選擇資料集: {info['name']} (訓練樣本: {info['train_samples']})")
            else:
                self.dataset_info_label.setText("")
                
        except Exception as e:
            self.log(f"[ERROR] 獲取資料集信息時發生錯誤: {e}")
            self.dataset_info_label.setText(f"獲取資料集信息時發生錯誤: {e}")
    
    def on_manual_dataset_changed(self):
        """當手動輸入資料集路徑時的回調"""
        try:
            dataset_path = self.stereo_dataset_edit.text()
            if dataset_path:
                # 檢查是否為有效的立體視覺資料集
                from pathlib import Path
                
                if self.is_stereo_dataset(Path(dataset_path)):
                    info = self.get_stereo_dataset_info(dataset_path)
                    
                    # 顯示資料集信息
                    info_text = f"資料集: {info['name']}\n"
                    info_text += f"訓練樣本: {info['train_samples']}\n"
                    info_text += f"驗證樣本: {info['val_samples']}\n"
                    info_text += f"測試樣本: {info['test_samples']}\n"
                    info_text += f"總樣本: {info['total_samples']}\n"
                    
                    # 添加圖像尺寸信息
                    if 'image_sizes' in info and info['image_sizes']:
                        sizes_text = ", ".join([f"{w}x{h}" for w, h in info['image_sizes'][:3]])  # 顯示前3個尺寸
                        if len(info['image_sizes']) > 3:
                            sizes_text += f" (+{len(info['image_sizes'])-3} more)"
                        info_text += f"\n圖像尺寸: {sizes_text}"
                    else:
                        info_text += "\n圖像尺寸: 檢測中..."
                    
                    self.dataset_info_label.setText(info_text)
                    self.log(f"✅ 手動選擇的資料集有效: {info['name']} (訓練樣本: {info['train_samples']})")
                else:
                    self.dataset_info_label.setText("無效的立體視覺資料集格式")
                    self.log("⚠️ 手動選擇的資料集格式無效")
            else:
                self.dataset_info_label.setText("")
                
        except Exception as e:
            self.log(f"[ERROR] 驗證手動資料集時發生錯誤: {e}")
            self.dataset_info_label.setText(f"驗證資料集時發生錯誤: {e}")
    
    def is_stereo_dataset(self, dataset_path):
        """檢查是否為有效的立體視覺資料集結構"""
        try:
            from pathlib import Path
            
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                return False
            
            # 檢查必需的子資料夾
            required_dirs = ['Img0', 'Img1', 'Disparity']
            for dir_name in required_dirs:
                dir_path = dataset_path / dir_name
                if not dir_path.exists() or not dir_path.is_dir():
                    return False
            
            # 檢查每個子資料夾是否包含 train, val, test 子資料夾
            for dir_name in required_dirs:
                dir_path = dataset_path / dir_name
                subdirs = ['train', 'val', 'test']
                for subdir in subdirs:
                    subdir_path = dir_path / subdir
                    if not subdir_path.exists() or not subdir_path.is_dir():
                        return False
            
            return True
            
        except Exception as e:
            self.log(f"[ERROR] 檢查資料集結構時發生錯誤: {e}")
            return False
    
    def get_stereo_dataset_info(self, dataset_path):
        """獲取立體視覺資料集信息"""
        try:
            from pathlib import Path
            import os
            from PIL import Image
            
            dataset_path = Path(dataset_path)
            dataset_name = dataset_path.name
            
            # 計算各分割的樣本數量
            train_samples = 0
            val_samples = 0
            test_samples = 0
            
            # 檢查 Img0/train 資料夾中的文件數量
            train_img0_path = dataset_path / 'Img0' / 'train'
            if train_img0_path.exists():
                train_samples = len([f for f in os.listdir(train_img0_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # 檢查 Img0/val 資料夾中的文件數量
            val_img0_path = dataset_path / 'Img0' / 'val'
            if val_img0_path.exists():
                val_samples = len([f for f in os.listdir(val_img0_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # 檢查 Img0/test 資料夾中的文件數量
            test_img0_path = dataset_path / 'Img0' / 'test'
            if test_img0_path.exists():
                test_samples = len([f for f in os.listdir(test_img0_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            total_samples = train_samples + val_samples + test_samples
            
            # 檢測圖像尺寸
            image_sizes = self.detect_image_sizes(dataset_path)
            
            return {
                'name': dataset_name,
                'path': str(dataset_path),
                'description': f"立體視覺資料集 ({total_samples} 樣本)",
                'train_samples': train_samples,
                'val_samples': val_samples,
                'test_samples': test_samples,
                'total_samples': total_samples,
                'image_sizes': image_sizes
            }
            
        except Exception as e:
            self.log(f"[ERROR] 獲取資料集信息時發生錯誤: {e}")
            return {
                'name': 'Unknown',
                'path': str(dataset_path),
                'description': '未知資料集',
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0,
                'total_samples': 0,
                'image_sizes': []
            }
    
    def detect_image_sizes(self, dataset_path):
        """檢測資料集中的圖像尺寸"""
        try:
            from pathlib import Path
            from PIL import Image
            import os
            
            dataset_path = Path(dataset_path)
            sizes = set()
            
            # 檢查所有分割的圖像尺寸
            for split in ['train', 'val', 'test']:
                img0_path = dataset_path / 'Img0' / split
                if img0_path.exists():
                    # 檢查前幾張圖像的尺寸
                    image_files = [f for f in os.listdir(img0_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    # 檢查前10張圖像的尺寸
                    for i, img_file in enumerate(image_files[:10]):
                        try:
                            img_path = img0_path / img_file
                            with Image.open(img_path) as img:
                                sizes.add(img.size)  # (width, height)
                        except Exception as e:
                            self.log(f"[WARNING] 無法讀取圖像 {img_file}: {e}")
                            continue
                        
                        # 如果已經檢測到足夠的尺寸變化，可以提前停止
                        if len(sizes) >= 5:
                            break
            
            # 轉換為列表並排序
            sizes_list = sorted(list(sizes), key=lambda x: x[0] * x[1])  # 按面積排序
            
            self.log(f"檢測到圖像尺寸: {sizes_list}")
            return sizes_list
            
        except Exception as e:
            self.log(f"[ERROR] 檢測圖像尺寸時發生錯誤: {e}")
            return []
    
    def detect_stereo_datasets(self):
        """檢測可用的立體視覺資料集"""
        try:
            from pathlib import Path
            
            datasets = []
            dataset_root = Path("Dataset")
            
            if not dataset_root.exists():
                return datasets
            
            # 遍歷 Dataset 目錄下的所有子目錄
            for item in dataset_root.iterdir():
                if item.is_dir():
                    # 檢查是否為立體視覺資料集
                    if self.is_stereo_dataset(item):
                        info = self.get_stereo_dataset_info(item)
                        datasets.append(info)
            
            return datasets
            
        except Exception as e:
            self.log(f"[ERROR] 檢測資料集時發生錯誤: {e}")
            return []
    
    def refresh_stereo_model_list(self):
        """刷新立體視覺預訓練模型列表"""
        try:
            # 檢查控件是否已創建
            if not hasattr(self, 'stereo_model_combo'):
                return  # 控件尚未創建，跳過刷新
            
            from pathlib import Path
            import os
            
            self.log("🔄 正在刷新立體視覺模型列表...")
            
            # 檢查模型目錄
            model_dirs = [
                Path("Model_file/Stereo_Vision"),
                Path("Model_file/PTH_File"),  # 向後兼容舊目錄
                Path("Model_file"),
            ]
            
            # 收集所有可用的 .pth 文件
            available_models = set()
            
            for model_dir in model_dirs:
                if model_dir.exists():
                    # 查找所有 .pth 文件
                    pth_files = list(model_dir.glob("*.pth"))
                    for pth_file in pth_files:
                        # 只添加包含 "stereo" 或 "raft" 的模型
                        file_name = pth_file.name.lower()
                        if 'stereo' in file_name or 'raft' in file_name:
                            available_models.add(pth_file.name)
            
            # 默認模型列表（如果沒有找到任何模型）
            default_models = [
                'raftstereo-sceneflow.pth',
                'raftstereo-middlebury.pth',
                'raftstereo-eth3d.pth',
                'iraftstereo_rvc.pth',
                'raftstereo-realtime.pth'
            ]
            
            # 保存當前選中的模型
            current_model = self.stereo_model_combo.currentText() if hasattr(self, 'stereo_model_combo') else None
            
            # 清空下拉框
            self.stereo_model_combo.clear()
            
            if available_models:
                # 先添加找到的模型（按字母順序）
                sorted_models = sorted(available_models)
                
                # 將默認模型排在前面
                priority_models = []
                other_models = []
                
                for model in sorted_models:
                    if model in default_models:
                        priority_models.append(model)
                    else:
                        other_models.append(model)
                
                # 組合列表：優先模型 + 其他模型
                all_models = priority_models + other_models
                
                # 只添加找到的模型到下拉框
                for model in all_models:
                    self.stereo_model_combo.addItem(model)
                
                self.log(f"✅ 找到 {len(available_models)} 個立體視覺模型")
                
                # 恢復之前的選擇
                if current_model:
                    # 嘗試匹配原來的選擇
                    for i in range(self.stereo_model_combo.count()):
                        item_text = self.stereo_model_combo.itemText(i)
                        if current_model in item_text:
                            self.stereo_model_combo.setCurrentIndex(i)
                            break
            else:
                # 沒有找到任何模型，只顯示提示信息
                self.stereo_model_combo.addItem("(無可用模型 - 請放置模型文件)")
                
                self.log("⚠️ 未在 Model_file/Stereo_Vision 目錄中找到模型文件")
                self.log("   請下載預訓練模型並放置在該目錄")
                self.log(f"   支持的模型: {', '.join(default_models)}")
            
            self.log("✅ 模型列表刷新完成")
            
        except Exception as e:
            self.log(f"[ERROR] 刷新模型列表時發生錯誤: {e}")
            import traceback
            self.log(f"   詳細錯誤: {traceback.format_exc()}")
            
            # 發生錯誤時顯示提示信息
            self.stereo_model_combo.clear()
            self.stereo_model_combo.addItem("(模型列表加載失敗)")

