"""
YOLO 訓練模組
Training Module
處理YOLO模型的標準訓練功能
"""

from pathlib import Path
import yaml
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                            QRadioButton, QButtonGroup, QTextEdit,
                            QFileDialog, QMessageBox, QFrame, QDialog,
                            QScrollArea, QSplitter)
from PyQt5.QtCore import Qt
from .base_module import BaseModule


class AdvancedParamsDialog(QDialog):
    """高級參數設置對話框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔧 高級訓練參數設置 Advanced Training Parameters")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.setMaximumSize(1200, 800)
        
        # 設置窗口圖標和樣式
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
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """設置用戶界面"""
        main_layout = QVBoxLayout(self)
        
        # 創建滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 創建內容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 創建高級參數組
        self._create_advanced_params_groups(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # 創建按鈕區域
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 重置為默認值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("✅ 確定")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(self.ok_btn)
        
        main_layout.addLayout(button_layout)
    
    def _create_advanced_params_groups(self, main_layout):
        """創建高級參數組"""
        # 數據增強參數組
        aug_group = QGroupBox("📊 數據增強參數")
        aug_layout = QGridLayout(aug_group)
        
        # 基礎增強參數
        aug_layout.addWidget(QLabel("Mosaic (%):"), 0, 0)
        self.mosaic_spin = QSpinBox()
        self.mosaic_spin.setRange(0, 100)
        self.mosaic_spin.setValue(100)
        self.mosaic_spin.setToolTip("Mosaic數據增強概率")
        aug_layout.addWidget(self.mosaic_spin, 0, 1)
        
        aug_layout.addWidget(QLabel("Mixup (%):"), 0, 2)
        self.mixup_spin = QSpinBox()
        self.mixup_spin.setRange(0, 100)
        self.mixup_spin.setValue(0)
        self.mixup_spin.setToolTip("Mixup數據增強概率")
        aug_layout.addWidget(self.mixup_spin, 0, 3)
        
        aug_layout.addWidget(QLabel("Copy-paste (%):"), 1, 0)
        self.copy_paste_spin = QSpinBox()
        self.copy_paste_spin.setRange(0, 100)
        self.copy_paste_spin.setValue(0)
        self.copy_paste_spin.setToolTip("Copy-paste數據增強概率")
        aug_layout.addWidget(self.copy_paste_spin, 1, 1)
        
        aug_layout.addWidget(QLabel("Scale (%):"), 1, 2)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(0, 100)
        self.scale_spin.setValue(50)
        self.scale_spin.setToolTip("圖像縮放範圍")
        aug_layout.addWidget(self.scale_spin, 1, 3)
        
        # HSV增強參數
        aug_layout.addWidget(QLabel("HSV色相 (%):"), 2, 0)
        self.hsv_h_spin = QSpinBox()
        self.hsv_h_spin.setRange(0, 100)
        self.hsv_h_spin.setValue(15)
        self.hsv_h_spin.setToolTip("HSV色相增強範圍")
        aug_layout.addWidget(self.hsv_h_spin, 2, 1)
        
        aug_layout.addWidget(QLabel("HSV飽和度 (%):"), 2, 2)
        self.hsv_s_spin = QSpinBox()
        self.hsv_s_spin.setRange(0, 100)
        self.hsv_s_spin.setValue(70)
        self.hsv_s_spin.setToolTip("HSV飽和度增強範圍")
        aug_layout.addWidget(self.hsv_s_spin, 2, 3)
        
        aug_layout.addWidget(QLabel("HSV明度 (%):"), 3, 0)
        self.hsv_v_spin = QSpinBox()
        self.hsv_v_spin.setRange(0, 100)
        self.hsv_v_spin.setValue(40)
        self.hsv_v_spin.setToolTip("HSV明度增強範圍")
        aug_layout.addWidget(self.hsv_v_spin, 3, 1)
        
        aug_layout.addWidget(QLabel("BGR通道 (%):"), 3, 2)
        self.bgr_spin = QSpinBox()
        self.bgr_spin.setRange(0, 100)
        self.bgr_spin.setValue(0)
        self.bgr_spin.setToolTip("BGR通道增強範圍")
        aug_layout.addWidget(self.bgr_spin, 3, 3)
        
        # 幾何變換參數
        aug_layout.addWidget(QLabel("旋轉角度 (°):"), 4, 0)
        self.degrees_spin = QSpinBox()
        self.degrees_spin.setRange(0, 180)
        self.degrees_spin.setValue(0)
        self.degrees_spin.setToolTip("圖像旋轉角度範圍")
        aug_layout.addWidget(self.degrees_spin, 4, 1)
        
        aug_layout.addWidget(QLabel("平移距離 (%):"), 4, 2)
        self.translate_spin = QSpinBox()
        self.translate_spin.setRange(0, 100)
        self.translate_spin.setValue(10)
        self.translate_spin.setToolTip("圖像平移距離範圍")
        aug_layout.addWidget(self.translate_spin, 4, 3)
        
        aug_layout.addWidget(QLabel("剪切角度 (°):"), 5, 0)
        self.shear_spin = QSpinBox()
        self.shear_spin.setRange(0, 45)
        self.shear_spin.setValue(0)
        self.shear_spin.setToolTip("圖像剪切角度範圍")
        aug_layout.addWidget(self.shear_spin, 5, 1)
        
        aug_layout.addWidget(QLabel("透視變換 (%):"), 5, 2)
        self.perspective_spin = QSpinBox()
        self.perspective_spin.setRange(0, 100)
        self.perspective_spin.setValue(0)
        self.perspective_spin.setToolTip("透視變換範圍")
        aug_layout.addWidget(self.perspective_spin, 5, 3)
        
        # 翻轉和裁剪參數
        aug_layout.addWidget(QLabel("上下翻轉 (%):"), 6, 0)
        self.flipud_spin = QSpinBox()
        self.flipud_spin.setRange(0, 100)
        self.flipud_spin.setValue(0)
        self.flipud_spin.setToolTip("上下翻轉概率")
        aug_layout.addWidget(self.flipud_spin, 6, 1)
        
        aug_layout.addWidget(QLabel("左右翻轉 (%):"), 6, 2)
        self.fliplr_spin = QSpinBox()
        self.fliplr_spin.setRange(0, 100)
        self.fliplr_spin.setValue(50)
        self.fliplr_spin.setToolTip("左右翻轉概率")
        aug_layout.addWidget(self.fliplr_spin, 6, 3)
        
        aug_layout.addWidget(QLabel("隨機擦除 (%):"), 7, 0)
        self.erasing_spin = QSpinBox()
        self.erasing_spin.setRange(0, 100)
        self.erasing_spin.setValue(0)
        self.erasing_spin.setToolTip("隨機擦除概率")
        aug_layout.addWidget(self.erasing_spin, 7, 1)
        
        aug_layout.addWidget(QLabel("裁剪比例 (%):"), 7, 2)
        self.crop_fraction_spin = QSpinBox()
        self.crop_fraction_spin.setRange(0, 100)
        self.crop_fraction_spin.setValue(100)
        self.crop_fraction_spin.setToolTip("裁剪比例")
        aug_layout.addWidget(self.crop_fraction_spin, 7, 3)
        
        # 自動增強策略
        aug_layout.addWidget(QLabel("自動增強策略:"), 8, 0)
        self.auto_augment_combo = QComboBox()
        self.auto_augment_combo.addItems(['None', 'randaugment', 'autoaugment', 'augmix'])
        self.auto_augment_combo.setToolTip("自動增強策略選擇")
        aug_layout.addWidget(self.auto_augment_combo, 8, 1, 1, 3)
        
        main_layout.addWidget(aug_group)
        
        # 優化器參數組
        opt_group = QGroupBox("⚙️ 優化器參數")
        opt_layout = QGridLayout(opt_group)
        
        opt_layout.addWidget(QLabel("權重衰減 (Weight Decay):"), 0, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(4)
        self.weight_decay_spin.setToolTip("權重衰減系數")
        opt_layout.addWidget(self.weight_decay_spin, 0, 1)
        
        opt_layout.addWidget(QLabel("動量 (Momentum):"), 0, 2)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setToolTip("SGD優化器動量")
        opt_layout.addWidget(self.momentum_spin, 0, 3)
        
        opt_layout.addWidget(QLabel("β1 (Adam):"), 1, 0)
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.0, 1.0)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setValue(0.9)
        self.beta1_spin.setDecimals(2)
        self.beta1_spin.setToolTip("Adam優化器β1參數")
        opt_layout.addWidget(self.beta1_spin, 1, 1)
        
        opt_layout.addWidget(QLabel("β2 (Adam):"), 1, 2)
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.0, 1.0)
        self.beta2_spin.setSingleStep(0.01)
        self.beta2_spin.setValue(0.999)
        self.beta2_spin.setDecimals(3)
        self.beta2_spin.setToolTip("Adam優化器β2參數")
        opt_layout.addWidget(self.beta2_spin, 1, 3)
        
        main_layout.addWidget(opt_group)
        
        # 學習率調度參數組
        lr_group = QGroupBox("📈 學習率調度")
        lr_layout = QGridLayout(lr_group)
        
        lr_layout.addWidget(QLabel("學習率調度器:"), 0, 0)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(['auto', 'linear', 'cosine', 'step', 'exp'])
        self.lr_scheduler_combo.setToolTip("學習率調度策略")
        lr_layout.addWidget(self.lr_scheduler_combo, 0, 1)
        
        lr_layout.addWidget(QLabel("學習率衰減:"), 0, 2)
        self.lr_decay_spin = QDoubleSpinBox()
        self.lr_decay_spin.setRange(0.0, 1.0)
        self.lr_decay_spin.setSingleStep(0.01)
        self.lr_decay_spin.setValue(0.1)
        self.lr_decay_spin.setDecimals(2)
        self.lr_decay_spin.setToolTip("學習率衰減系數")
        lr_layout.addWidget(self.lr_decay_spin, 0, 3)
        
        lr_layout.addWidget(QLabel("Warmup Epochs:"), 1, 0)
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 50)
        self.warmup_epochs_spin.setValue(3)
        self.warmup_epochs_spin.setToolTip("學習率預熱輪數")
        lr_layout.addWidget(self.warmup_epochs_spin, 1, 1)
        
        lr_layout.addWidget(QLabel("Warmup動量:"), 1, 2)
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setSingleStep(0.01)
        self.warmup_momentum_spin.setValue(0.8)
        self.warmup_momentum_spin.setDecimals(2)
        self.warmup_momentum_spin.setToolTip("預熱期動量")
        lr_layout.addWidget(self.warmup_momentum_spin, 1, 3)
        
        main_layout.addWidget(lr_group)
        
        # 驗證參數組
        val_group = QGroupBox("✅ 驗證參數")
        val_layout = QGridLayout(val_group)
        
        val_layout.addWidget(QLabel("驗證頻率:"), 0, 0)
        self.val_frequency_spin = QSpinBox()
        self.val_frequency_spin.setRange(1, 100)
        self.val_frequency_spin.setValue(1)
        self.val_frequency_spin.setToolTip("每N個epoch驗證一次")
        val_layout.addWidget(self.val_frequency_spin, 0, 1)
        
        val_layout.addWidget(QLabel("驗證迭代次數:"), 0, 2)
        self.val_iters_spin = QSpinBox()
        self.val_iters_spin.setRange(1, 1000)
        self.val_iters_spin.setValue(32)
        self.val_iters_spin.setToolTip("驗證時的迭代次數")
        val_layout.addWidget(self.val_iters_spin, 0, 3)
        
        val_layout.addWidget(QLabel("早停耐心值:"), 1, 0)
        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(0, 100)
        self.early_stopping_patience_spin.setValue(50)
        self.early_stopping_patience_spin.setToolTip("早停耐心值，0表示禁用早停")
        val_layout.addWidget(self.early_stopping_patience_spin, 1, 1)
        
        val_layout.addWidget(QLabel("早停最小改善:"), 1, 2)
        self.early_stopping_min_delta_spin = QDoubleSpinBox()
        self.early_stopping_min_delta_spin.setRange(0.0, 1.0)
        self.early_stopping_min_delta_spin.setSingleStep(0.001)
        self.early_stopping_min_delta_spin.setValue(0.001)
        self.early_stopping_min_delta_spin.setDecimals(3)
        self.early_stopping_min_delta_spin.setToolTip("早停最小改善閾值")
        val_layout.addWidget(self.early_stopping_min_delta_spin, 1, 3)
        
        main_layout.addWidget(val_group)
        
        # 設備參數組
        device_group = QGroupBox("🖥️ 設備設置")
        device_layout = QGridLayout(device_group)
        
        device_layout.addWidget(QLabel("設備選擇:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
        self.device_combo.setToolTip("選擇訓練設備")
        device_layout.addWidget(self.device_combo, 0, 1)
        
        device_layout.addWidget(QLabel("多GPU訓練:"), 0, 2)
        self.multi_gpu_checkbox = QCheckBox("啟用多GPU訓練")
        self.multi_gpu_checkbox.setToolTip("啟用多GPU並行訓練")
        device_layout.addWidget(self.multi_gpu_checkbox, 0, 3)
        
        device_layout.addWidget(QLabel("GPU內存優化:"), 1, 0)
        self.gpu_memory_optimization_checkbox = QCheckBox("啟用GPU內存優化")
        self.gpu_memory_optimization_checkbox.setChecked(True)
        self.gpu_memory_optimization_checkbox.setToolTip("啟用GPU內存優化")
        device_layout.addWidget(self.gpu_memory_optimization_checkbox, 1, 1)
        
        device_layout.addWidget(QLabel("數據加載優化:"), 1, 2)
        self.data_loading_optimization_checkbox = QCheckBox("啟用數據加載優化")
        self.data_loading_optimization_checkbox.setChecked(True)
        self.data_loading_optimization_checkbox.setToolTip("啟用數據加載優化")
        device_layout.addWidget(self.data_loading_optimization_checkbox, 1, 3)
        
        main_layout.addWidget(device_group)
        
        # 其他高級參數組
        advanced_group = QGroupBox("🔧 其他高級參數")
        advanced_layout = QGridLayout(advanced_group)
        
        advanced_layout.addWidget(QLabel("關閉Mosaic Epoch:"), 0, 0)
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(10)
        self.close_mosaic_spin.setToolTip("最後N個epoch關閉Mosaic增強")
        advanced_layout.addWidget(self.close_mosaic_spin, 0, 1)
        
        advanced_layout.addWidget(QLabel("單類別訓練:"), 0, 2)
        self.single_cls_checkbox = QCheckBox("啟用單類別訓練")
        self.single_cls_checkbox.setToolTip("將所有類別視為單一類別")
        advanced_layout.addWidget(self.single_cls_checkbox, 0, 3)
        
        advanced_layout.addWidget(QLabel("緩存數據:"), 1, 0)
        self.cache_checkbox = QCheckBox("啟用數據緩存")
        self.cache_checkbox.setToolTip("緩存數據到內存以加速訓練")
        advanced_layout.addWidget(self.cache_checkbox, 1, 1)
        
        advanced_layout.addWidget(QLabel("恢復訓練:"), 1, 2)
        self.resume_checkbox = QCheckBox("從檢查點恢復")
        self.resume_checkbox.setToolTip("從最後一個檢查點恢復訓練")
        advanced_layout.addWidget(self.resume_checkbox, 1, 3)
        
        main_layout.addWidget(advanced_group)
    
    def reset_to_defaults(self):
        """重置為默認值"""
        # 數據增強參數
        self.mosaic_spin.setValue(100)
        self.mixup_spin.setValue(0)
        self.copy_paste_spin.setValue(0)
        self.scale_spin.setValue(50)
        self.hsv_h_spin.setValue(15)
        self.hsv_s_spin.setValue(70)
        self.hsv_v_spin.setValue(40)
        self.bgr_spin.setValue(0)
        self.auto_augment_combo.setCurrentText('None')
        
        # 幾何變換參數
        self.degrees_spin.setValue(0)
        self.translate_spin.setValue(10)
        self.shear_spin.setValue(0)
        self.perspective_spin.setValue(0)
        
        # 翻轉和裁剪參數
        self.flipud_spin.setValue(0)
        self.fliplr_spin.setValue(50)
        self.erasing_spin.setValue(0)
        self.crop_fraction_spin.setValue(100)
        
        # 優化器參數
        self.weight_decay_spin.setValue(0.0005)
        self.momentum_spin.setValue(0.937)
        self.beta1_spin.setValue(0.9)
        self.beta2_spin.setValue(0.999)
        
        # 學習率調度參數
        self.lr_scheduler_combo.setCurrentText('auto')
        self.lr_decay_spin.setValue(0.1)
        self.warmup_epochs_spin.setValue(3)
        self.warmup_momentum_spin.setValue(0.8)
        
        # 驗證參數
        self.val_frequency_spin.setValue(1)
        self.val_iters_spin.setValue(32)
        self.early_stopping_patience_spin.setValue(50)
        self.early_stopping_min_delta_spin.setValue(0.001)
        
        # 設備參數
        self.device_combo.setCurrentText('auto')
        self.multi_gpu_checkbox.setChecked(False)
        self.gpu_memory_optimization_checkbox.setChecked(True)
        self.data_loading_optimization_checkbox.setChecked(True)
        
        # 其他高級參數
        self.close_mosaic_spin.setValue(10)
        self.single_cls_checkbox.setChecked(False)
        self.cache_checkbox.setChecked(False)
        self.resume_checkbox.setChecked(False)
    
    def get_advanced_params(self):
        """獲取高級參數值"""
        return {
            # 數據增強參數
            'scale': self.scale_spin.value() * 0.01,
            'mosaic': self.mosaic_spin.value() * 0.01,
            'mixup': self.mixup_spin.value() * 0.01,
            'copy_paste': self.copy_paste_spin.value() * 0.01,
            'hsv_h': self.hsv_h_spin.value() * 0.01,
            'hsv_s': self.hsv_s_spin.value() * 0.01,
            'hsv_v': self.hsv_v_spin.value() * 0.01,
            'bgr': self.bgr_spin.value() * 0.01,
            'auto_augment': self.auto_augment_combo.currentText() if self.auto_augment_combo.currentText() != 'None' else None,
            # 幾何變換參數
            'degrees': self.degrees_spin.value(),
            'translate': self.translate_spin.value() * 0.01,
            'shear': self.shear_spin.value() * 0.01,
            'perspective': self.perspective_spin.value() * 0.01,
            # 翻轉和裁剪參數
            'flipud': self.flipud_spin.value() * 0.01,
            'fliplr': self.fliplr_spin.value() * 0.01,
            'erasing': self.erasing_spin.value() * 0.01,
            'crop_fraction': self.crop_fraction_spin.value() * 0.01,
            # 優化器參數
            'weight_decay': self.weight_decay_spin.value(),
            'momentum': self.momentum_spin.value(),
            'beta1': self.beta1_spin.value(),
            'beta2': self.beta2_spin.value(),
            # 學習率調度參數
            'lr_scheduler': self.lr_scheduler_combo.currentText(),
            'lr_decay': self.lr_decay_spin.value(),
            'warmup_epochs': self.warmup_epochs_spin.value(),
            'warmup_momentum': self.warmup_momentum_spin.value(),
            # 驗證參數
            'val_frequency': self.val_frequency_spin.value(),
            'val_iters': self.val_iters_spin.value(),
            'early_stopping_patience': self.early_stopping_patience_spin.value(),
            'early_stopping_min_delta': self.early_stopping_min_delta_spin.value(),
            # 設備參數
            'device': self.device_combo.currentText(),
            'multi_gpu': self.multi_gpu_checkbox.isChecked(),
            'gpu_memory_optimization': self.gpu_memory_optimization_checkbox.isChecked(),
            'data_loading_optimization': self.data_loading_optimization_checkbox.isChecked(),
            # 其他高級參數
            'close_mosaic': self.close_mosaic_spin.value(),
            'single_cls': self.single_cls_checkbox.isChecked(),
            'cache': self.cache_checkbox.isChecked(),
            'resume': self.resume_checkbox.isChecked()
        }


class TrainingModule(BaseModule):
    """訓練功能模組"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.auto_refresh_timer = None
        
    def create_tab(self):
        """創建訓練標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 資料集選擇
        dataset_group = self._create_dataset_group()
        layout.addWidget(dataset_group)
        
        # 模型選擇
        model_group = self._create_model_group()
        layout.addWidget(model_group)
        
        # 訓練參數
        params_group = self._create_params_group()
        layout.addWidget(params_group)
        
        # 訓練控制
        control_group = self._create_control_group()
        layout.addWidget(control_group)
        
        self.tab_widget = tab
        
        # 啟動自動刷新
        self.start_auto_refresh()
        
        return tab
    
    def load_settings(self, settings_manager):
        """載入訓練模組設定"""
        try:
            # 載入標準訓練設定
            standard_settings = settings_manager.get_section('standard_training')
            if standard_settings:
                # 基礎參數
                if hasattr(self, 'epochs_spin') and 'epochs' in standard_settings:
                    self.epochs_spin.setValue(standard_settings['epochs'])
                if hasattr(self, 'batch_size_spin') and 'batch_size' in standard_settings:
                    self.batch_size_spin.setValue(standard_settings['batch_size'])
                if hasattr(self, 'learning_rate_spin') and 'learning_rate' in standard_settings:
                    self.learning_rate_spin.setValue(standard_settings['learning_rate'])
                if hasattr(self, 'imgsz_spin') and 'imgsz' in standard_settings:
                    self.imgsz_spin.setValue(standard_settings['imgsz'])
                if hasattr(self, 'save_period_spin') and 'save_period' in standard_settings:
                    self.save_period_spin.setValue(standard_settings['save_period'])
                
                # 數據增強參數
                if hasattr(self, 'scale_spin') and 'scale' in standard_settings:
                    self.scale_spin.setValue(standard_settings['scale'])
                if hasattr(self, 'mosaic_spin') and 'mosaic' in standard_settings:
                    self.mosaic_spin.setValue(standard_settings['mosaic'])
                if hasattr(self, 'mixup_spin') and 'mixup' in standard_settings:
                    self.mixup_spin.setValue(standard_settings['mixup'])
                if hasattr(self, 'copy_paste_spin') and 'copy_paste' in standard_settings:
                    self.copy_paste_spin.setValue(standard_settings['copy_paste'])
                
                # HSV和BGR參數
                if hasattr(self, 'hsv_h_spin') and 'hsv_h' in standard_settings:
                    self.hsv_h_spin.setValue(standard_settings['hsv_h'])
                if hasattr(self, 'hsv_s_spin') and 'hsv_s' in standard_settings:
                    self.hsv_s_spin.setValue(standard_settings['hsv_s'])
                if hasattr(self, 'hsv_v_spin') and 'hsv_v' in standard_settings:
                    self.hsv_v_spin.setValue(standard_settings['hsv_v'])
                if hasattr(self, 'bgr_spin') and 'bgr' in standard_settings:
                    self.bgr_spin.setValue(standard_settings['bgr'])
                
                # 幾何變換參數
                if hasattr(self, 'degrees_spin') and 'degrees' in standard_settings:
                    self.degrees_spin.setValue(standard_settings['degrees'])
                if hasattr(self, 'translate_spin') and 'translate' in standard_settings:
                    self.translate_spin.setValue(standard_settings['translate'])
                if hasattr(self, 'shear_spin') and 'shear' in standard_settings:
                    self.shear_spin.setValue(standard_settings['shear'])
                if hasattr(self, 'perspective_spin') and 'perspective' in standard_settings:
                    self.perspective_spin.setValue(standard_settings['perspective'])
                
                # 翻轉和裁剪參數
                if hasattr(self, 'flipud_spin') and 'flipud' in standard_settings:
                    self.flipud_spin.setValue(standard_settings['flipud'])
                if hasattr(self, 'fliplr_spin') and 'fliplr' in standard_settings:
                    self.fliplr_spin.setValue(standard_settings['fliplr'])
                if hasattr(self, 'erasing_spin') and 'erasing' in standard_settings:
                    self.erasing_spin.setValue(standard_settings['erasing'])
                if hasattr(self, 'crop_fraction_spin') and 'crop_fraction' in standard_settings:
                    self.crop_fraction_spin.setValue(standard_settings['crop_fraction'])
                
                # 訓練控制參數
                if hasattr(self, 'close_mosaic_spin') and 'close_mosaic' in standard_settings:
                    self.close_mosaic_spin.setValue(standard_settings['close_mosaic'])
                if hasattr(self, 'workers_spin') and 'workers' in standard_settings:
                    self.workers_spin.setValue(standard_settings['workers'])
                
                # 優化器和AMP設定
                if hasattr(self, 'optimizer_combo') and 'optimizer' in standard_settings:
                    # 查找對應的索引
                    for i in range(self.optimizer_combo.count()):
                        if self.optimizer_combo.itemText(i) == standard_settings['optimizer']:
                            self.optimizer_combo.setCurrentIndex(i)
                            break
                if hasattr(self, 'amp_checkbox') and 'amp' in standard_settings:
                    self.amp_checkbox.setChecked(standard_settings['amp'])
                
                # 模型和資料集路徑
                if hasattr(self, 'train_dataset_edit') and 'dataset_path' in standard_settings:
                    self.train_dataset_edit.setText(standard_settings['dataset_path'])
                if hasattr(self, 'train_model_edit') and 'model_file' in standard_settings:
                    self.train_model_edit.setText(standard_settings['model_file'])
                
                # 訓練模式
                if hasattr(self, 'pretrained_radio') and 'training_mode' in standard_settings:
                    if standard_settings['training_mode'] == 'pretrained':
                        self.pretrained_radio.setChecked(True)
                    elif standard_settings['training_mode'] == 'yaml':
                        self.yaml_radio.setChecked(True)
                
                # 載入高級參數
                if 'advanced_params' in standard_settings and standard_settings['advanced_params']:
                    self.advanced_params = standard_settings['advanced_params']
                    self.log("✅ 高級參數已載入")
                
                self.log("✅ 訓練設定載入成功")
        except Exception as e:
            self.log(f"[WARNING] 載入訓練設定失敗: {e}")
    
    def save_settings(self, settings_manager):
        """保存訓練模組設定"""
        try:
            standard_settings = {}
            
            # 基礎參數
            try:
                standard_settings['epochs'] = self.epochs_spin.value()
                standard_settings['batch_size'] = self.batch_size_spin.value()
                standard_settings['learning_rate'] = self.learning_rate_spin.value()
                standard_settings['imgsz'] = self.imgsz_spin.value()
                standard_settings['save_period'] = self.save_period_spin.value()
                
                self.log("✅ 基本訓練參數已保存:")
                self.log(f"   訓練輪數: {standard_settings['epochs']}")
                self.log(f"   批次大小: {standard_settings['batch_size']}")
                self.log(f"   學習率: {standard_settings['learning_rate']}")
                self.log(f"   圖像尺寸: {standard_settings['imgsz']}")
                self.log(f"   保存週期: {standard_settings['save_period']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存基本訓練參數時發生錯誤: {e}")
                # 設置默認值
                standard_settings['epochs'] = 100
                standard_settings['batch_size'] = 16
                standard_settings['learning_rate'] = 0.01
                standard_settings['imgsz'] = 640
                standard_settings['save_period'] = -1
            
            # 數據增強參數
            if hasattr(self, 'scale_spin'):
                standard_settings['scale'] = self.scale_spin.value()
            if hasattr(self, 'mosaic_spin'):
                standard_settings['mosaic'] = self.mosaic_spin.value()
            if hasattr(self, 'mixup_spin'):
                standard_settings['mixup'] = self.mixup_spin.value()
            if hasattr(self, 'copy_paste_spin'):
                standard_settings['copy_paste'] = self.copy_paste_spin.value()
            
            # HSV和BGR參數
            if hasattr(self, 'hsv_h_spin'):
                standard_settings['hsv_h'] = self.hsv_h_spin.value()
            if hasattr(self, 'hsv_s_spin'):
                standard_settings['hsv_s'] = self.hsv_s_spin.value()
            if hasattr(self, 'hsv_v_spin'):
                standard_settings['hsv_v'] = self.hsv_v_spin.value()
            if hasattr(self, 'bgr_spin'):
                standard_settings['bgr'] = self.bgr_spin.value()
            
            # 幾何變換參數
            if hasattr(self, 'degrees_spin'):
                standard_settings['degrees'] = self.degrees_spin.value()
            if hasattr(self, 'translate_spin'):
                standard_settings['translate'] = self.translate_spin.value()
            if hasattr(self, 'shear_spin'):
                standard_settings['shear'] = self.shear_spin.value()
            if hasattr(self, 'perspective_spin'):
                standard_settings['perspective'] = self.perspective_spin.value()
            
            # 翻轉和裁剪參數
            if hasattr(self, 'flipud_spin'):
                standard_settings['flipud'] = self.flipud_spin.value()
            if hasattr(self, 'fliplr_spin'):
                standard_settings['fliplr'] = self.fliplr_spin.value()
            if hasattr(self, 'erasing_spin'):
                standard_settings['erasing'] = self.erasing_spin.value()
            if hasattr(self, 'crop_fraction_spin'):
                standard_settings['crop_fraction'] = self.crop_fraction_spin.value()
            
            # 訓練控制參數
            if hasattr(self, 'close_mosaic_spin'):
                standard_settings['close_mosaic'] = self.close_mosaic_spin.value()
            if hasattr(self, 'workers_spin'):
                standard_settings['workers'] = self.workers_spin.value()
            
            # 優化器和AMP設定
            if hasattr(self, 'optimizer_combo'):
                standard_settings['optimizer'] = self.optimizer_combo.currentText()
            if hasattr(self, 'amp_checkbox'):
                standard_settings['amp'] = self.amp_checkbox.isChecked()
            
            # 模型和資料集路徑
            try:
                standard_settings['dataset_path'] = self.train_dataset_edit.text()
                standard_settings['model_file'] = self.train_model_edit.text()
                
                self.log("✅ 路徑參數已保存:")
                self.log(f"   資料集路徑: {standard_settings['dataset_path']}")
                self.log(f"   模型文件: {standard_settings['model_file']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存路徑參數時發生錯誤: {e}")
                standard_settings['dataset_path'] = ""
                standard_settings['model_file'] = ""
            
            # 訓練模式
            try:
                if self.pretrained_radio.isChecked():
                    standard_settings['training_mode'] = 'pretrained'
                elif self.yaml_radio.isChecked():
                    standard_settings['training_mode'] = 'yaml'
                else:
                    standard_settings['training_mode'] = 'pretrained'  # 默認值
                
                self.log(f"✅ 訓練模式已保存: {standard_settings['training_mode']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存訓練模式時發生錯誤: {e}")
                standard_settings['training_mode'] = 'pretrained'
            
            # 保存高級參數（即使為空也要保存，以保持一致性）
            if hasattr(self, 'advanced_params'):
                standard_settings['advanced_params'] = self.advanced_params
                if self.advanced_params:
                    self.log("✅ 高級參數已保存")
                else:
                    self.log("ℹ️ 高級參數為空，已保存空值")
            
            # 保存到設定管理器
            settings_manager.set_section('standard_training', standard_settings)
            self.log("✅ 訓練設定保存成功")
            
        except Exception as e:
            self.log(f"[WARNING] 保存訓練設定失敗: {e}")
        
    def _create_dataset_group(self):
        """創建資料集選擇組"""
        group = QGroupBox("訓練資料集")
        layout = QGridLayout(group)
        
        layout.addWidget(QLabel("選擇資料集:"), 0, 0)
        self.train_dataset_combo = QComboBox()
        self.train_dataset_combo.setPlaceholderText("請選擇或輸入資料集路徑")
        self.train_dataset_combo.setEditable(True)
        self.train_dataset_combo.setMinimumWidth(300)
        self.train_dataset_combo.currentTextChanged.connect(self.update_train_dataset_info)
        layout.addWidget(self.train_dataset_combo, 1, 0)
        
        self.train_dataset_btn = QPushButton("瀏覽")
        self.train_dataset_btn.clicked.connect(self.browse_train_dataset)
        layout.addWidget(self.train_dataset_btn, 1, 1)
        
        self.auto_find_train_dataset_btn = QPushButton("🔍 自動尋找")
        self.auto_find_train_dataset_btn.clicked.connect(self.auto_find_train_dataset)
        self.auto_find_train_dataset_btn.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold;"
        )
        layout.addWidget(self.auto_find_train_dataset_btn, 1, 2)
        
        # 資料集狀態顯示
        self.train_dataset_status = QLabel("")
        self.train_dataset_status.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(self.train_dataset_status, 2, 0, 1, 3)
        
        # 最後使用資訊
        self.last_used_info = QLabel("")
        self.last_used_info.setStyleSheet(
            "color: #007bff; font-size: 10px; font-style: italic; padding: 2px;"
        )
        layout.addWidget(self.last_used_info, 3, 0, 1, 3)
        
        return group
        
    def _create_model_group(self):
        """創建模型選擇組"""
        group = QGroupBox("模型設定")
        layout = QGridLayout(group)
        
        # 訓練模式選擇
        mode_layout = QHBoxLayout()
        self.training_mode_group = QButtonGroup()
        
        self.pretrained_radio = QRadioButton("預訓練模型 (PT)")
        self.pretrained_radio.setChecked(True)
        self.pretrained_radio.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.training_mode_group.addButton(self.pretrained_radio, 0)
        mode_layout.addWidget(self.pretrained_radio)
        
        self.retrain_radio = QRadioButton("重新訓練 (YAML)")
        self.retrain_radio.setStyleSheet("color: #ff6b35; font-weight: bold;")
        self.training_mode_group.addButton(self.retrain_radio, 1)
        mode_layout.addWidget(self.retrain_radio)
        
        # 連接模式切換事件
        self.pretrained_radio.toggled.connect(self.on_training_mode_changed)
        
        layout.addLayout(mode_layout, 0, 0, 1, 3)
        
        # 模型文件選擇
        layout.addWidget(QLabel("選擇模型:"), 1, 0)
        self.model_file_combo = QComboBox()
        self.model_file_combo.setMinimumWidth(300)
        layout.addWidget(self.model_file_combo, 2, 0)
        
        # 刷新模型按鈕
        self.refresh_model_btn = QPushButton("🔄 刷新")
        self.refresh_model_btn.clicked.connect(self.refresh_model_list)
        layout.addWidget(self.refresh_model_btn, 2, 1)
        
        # 自動刷新控制按鈕
        self.auto_refresh_btn = QPushButton("⏰ 自動刷新")
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        self.auto_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        layout.addWidget(self.auto_refresh_btn, 2, 2)
        
        # 模型大小選擇（僅YAML模式顯示）
        self.train_model_size_label = QLabel("模型大小:")
        self.train_model_size_combo = QComboBox()
        self.train_model_size_combo.addItems([
            "n (nano)", "s (small)", "m (medium)", "l (large)", "x (xlarge)"
        ])
        layout.addWidget(self.train_model_size_label, 3, 0)
        layout.addWidget(self.train_model_size_combo, 3, 1)
        
        # 模型狀態
        self.train_model_status = QLabel("")
        self.train_model_status.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(self.train_model_status, 4, 0, 1, 3)
        
        # 初始隱藏模型大小選擇（預設為PT模式）
        self.train_model_size_label.setVisible(False)
        self.train_model_size_combo.setVisible(False)
        
        return group
        
    def _create_params_group(self):
        """創建訓練參數組"""
        group = QGroupBox("訓練參數")
        layout = QGridLayout(group)
        
        # 基礎參數
        layout.addWidget(QLabel("訓練輪數 (Epochs):"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        layout.addWidget(self.epochs_spin, 0, 1)
        
        layout.addWidget(QLabel("學習率 (×0.001):"), 0, 2)
        self.learning_rate_spin = QSpinBox()
        self.learning_rate_spin.setRange(1, 1000)
        self.learning_rate_spin.setValue(10)  # 實際為0.01
        layout.addWidget(self.learning_rate_spin, 0, 3)
        
        layout.addWidget(QLabel("批次大小 (Batch Size):"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)
        layout.addWidget(self.batch_size_spin, 1, 1)
        
        layout.addWidget(QLabel("圖像大小 (Image Size):"), 1, 2)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        layout.addWidget(self.imgsz_spin, 1, 3)
        
        # 保存週期
        layout.addWidget(QLabel("保存週期 (Save Period):"), 2, 0)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(-1, 1000)
        self.save_period_spin.setValue(-1)
        self.save_period_spin.setToolTip("-1 = 僅保存最後一個epoch")
        layout.addWidget(self.save_period_spin, 2, 1)
        
        # 工作進程
        layout.addWidget(QLabel("工作進程 (Workers):"), 2, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(8)
        layout.addWidget(self.workers_spin, 2, 3)
        
        # 優化器
        layout.addWidget(QLabel("優化器 (Optimizer):"), 3, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp'])
        layout.addWidget(self.optimizer_combo, 3, 1)
        
        # AMP混合精度
        self.amp_checkbox = QCheckBox("啟用 AMP 混合精度訓練")
        self.amp_checkbox.setChecked(True)
        layout.addWidget(self.amp_checkbox, 3, 2, 1, 2)
        
        # 高級參數控制按鈕
        advanced_control_layout = QHBoxLayout()
        self.show_advanced_btn = QPushButton("🔧 高級參數設置")
        self.show_advanced_btn.clicked.connect(self.open_advanced_params_dialog)
        self.show_advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        advanced_control_layout.addWidget(self.show_advanced_btn)
        advanced_control_layout.addStretch()
        layout.addLayout(advanced_control_layout, 4, 0, 1, 4)
        
        # 存儲高級參數
        self.advanced_params = {}
        
        return group
    
    def open_advanced_params_dialog(self):
        """打開高級參數設置對話框"""
        dialog = AdvancedParamsDialog(self.parent)
        
        # 如果已有高級參數，設置到對話框中
        if self.advanced_params:
            self._set_dialog_params(dialog, self.advanced_params)
        
        if dialog.exec_() == QDialog.Accepted:
            # 用戶點擊確定，保存高級參數
            self.advanced_params = dialog.get_advanced_params()
            self.log("🔧 高級參數已更新 Advanced parameters updated")
        else:
            # 用戶點擊取消，不保存
            self.log("🔧 高級參數設置已取消 Advanced parameters setup cancelled")
    
    def _set_dialog_params(self, dialog, params):
        """設置對話框參數值"""
        try:
            # 數據增強參數
            if 'scale' in params:
                dialog.scale_spin.setValue(int(params['scale'] * 100))
            if 'mosaic' in params:
                dialog.mosaic_spin.setValue(int(params['mosaic'] * 100))
            if 'mixup' in params:
                dialog.mixup_spin.setValue(int(params['mixup'] * 100))
            if 'copy_paste' in params:
                dialog.copy_paste_spin.setValue(int(params['copy_paste'] * 100))
            if 'hsv_h' in params:
                dialog.hsv_h_spin.setValue(int(params['hsv_h'] * 100))
            if 'hsv_s' in params:
                dialog.hsv_s_spin.setValue(int(params['hsv_s'] * 100))
            if 'hsv_v' in params:
                dialog.hsv_v_spin.setValue(int(params['hsv_v'] * 100))
            if 'bgr' in params:
                dialog.bgr_spin.setValue(int(params['bgr'] * 100))
            if 'auto_augment' in params:
                if params['auto_augment']:
                    dialog.auto_augment_combo.setCurrentText(params['auto_augment'])
                else:
                    dialog.auto_augment_combo.setCurrentText('None')
            
            # 幾何變換參數
            if 'degrees' in params:
                dialog.degrees_spin.setValue(int(params['degrees']))
            if 'translate' in params:
                dialog.translate_spin.setValue(int(params['translate'] * 100))
            if 'shear' in params:
                dialog.shear_spin.setValue(int(params['shear'] * 100))
            if 'perspective' in params:
                dialog.perspective_spin.setValue(int(params['perspective'] * 100))
            
            # 翻轉和裁剪參數
            if 'flipud' in params:
                dialog.flipud_spin.setValue(int(params['flipud'] * 100))
            if 'fliplr' in params:
                dialog.fliplr_spin.setValue(int(params['fliplr'] * 100))
            if 'erasing' in params:
                dialog.erasing_spin.setValue(int(params['erasing'] * 100))
            if 'crop_fraction' in params:
                dialog.crop_fraction_spin.setValue(int(params['crop_fraction'] * 100))
            
            # 優化器參數
            if 'weight_decay' in params:
                dialog.weight_decay_spin.setValue(params['weight_decay'])
            if 'momentum' in params:
                dialog.momentum_spin.setValue(params['momentum'])
            if 'beta1' in params:
                dialog.beta1_spin.setValue(params['beta1'])
            if 'beta2' in params:
                dialog.beta2_spin.setValue(params['beta2'])
            
            # 學習率調度參數
            if 'lr_scheduler' in params:
                dialog.lr_scheduler_combo.setCurrentText(params['lr_scheduler'])
            if 'lr_decay' in params:
                dialog.lr_decay_spin.setValue(params['lr_decay'])
            if 'warmup_epochs' in params:
                dialog.warmup_epochs_spin.setValue(params['warmup_epochs'])
            if 'warmup_momentum' in params:
                dialog.warmup_momentum_spin.setValue(params['warmup_momentum'])
            
            # 驗證參數
            if 'val_frequency' in params:
                dialog.val_frequency_spin.setValue(params['val_frequency'])
            if 'val_iters' in params:
                dialog.val_iters_spin.setValue(params['val_iters'])
            if 'early_stopping_patience' in params:
                dialog.early_stopping_patience_spin.setValue(params['early_stopping_patience'])
            if 'early_stopping_min_delta' in params:
                dialog.early_stopping_min_delta_spin.setValue(params['early_stopping_min_delta'])
            
            # 設備參數
            if 'device' in params:
                dialog.device_combo.setCurrentText(params['device'])
            if 'multi_gpu' in params:
                dialog.multi_gpu_checkbox.setChecked(params['multi_gpu'])
            if 'gpu_memory_optimization' in params:
                dialog.gpu_memory_optimization_checkbox.setChecked(params['gpu_memory_optimization'])
            if 'data_loading_optimization' in params:
                dialog.data_loading_optimization_checkbox.setChecked(params['data_loading_optimization'])
            
            # 其他高級參數
            if 'close_mosaic' in params:
                dialog.close_mosaic_spin.setValue(params['close_mosaic'])
            if 'single_cls' in params:
                dialog.single_cls_checkbox.setChecked(params['single_cls'])
            if 'cache' in params:
                dialog.cache_checkbox.setChecked(params['cache'])
            if 'resume' in params:
                dialog.resume_checkbox.setChecked(params['resume'])
                
        except Exception as e:
            self.log(f"[WARNING] 設置高級參數失敗: {e}")
    
        
    def _create_additional_params(self):
        """創建額外參數（隱藏，用於保持兼容性）"""
        # 這些參數在簡化版中使用默認值
        self.scale_spin = QSpinBox()
        self.scale_spin.setValue(50)  # 0.5
        
        self.copy_paste_spin = QSpinBox()
        self.copy_paste_spin.setValue(0)
        
        self.hsv_h_spin = QSpinBox()
        self.hsv_h_spin.setValue(15)  # 0.015
        
        self.hsv_s_spin = QSpinBox()
        self.hsv_s_spin.setValue(70)  # 0.7
        
        self.hsv_v_spin = QSpinBox()
        self.hsv_v_spin.setValue(40)  # 0.4
        
        self.bgr_spin = QSpinBox()
        self.bgr_spin.setValue(0)
        
        self.auto_augment_combo = QComboBox()
        self.auto_augment_combo.addItems(['randaugment', 'autoaugment', 'augmix', 'None'])
        
        self.degrees_spin = QSpinBox()
        self.degrees_spin.setValue(0)
        
        self.translate_spin = QSpinBox()
        self.translate_spin.setValue(10)  # 0.1
        
        self.shear_spin = QSpinBox()
        self.shear_spin.setValue(0)
        
        self.perspective_spin = QSpinBox()
        self.perspective_spin.setValue(0)
        
        self.flipud_spin = QSpinBox()
        self.flipud_spin.setValue(0)
        
        self.fliplr_spin = QSpinBox()
        self.fliplr_spin.setValue(50)  # 0.5
        
        self.erasing_spin = QSpinBox()
        self.erasing_spin.setValue(0)
        
        self.crop_fraction_spin = QSpinBox()
        self.crop_fraction_spin.setValue(100)  # 1.0
        
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setValue(10)
        
    def _create_control_group(self):
        """創建訓練控制組"""
        group = QGroupBox("訓練控制")
        layout = QHBoxLayout(group)
        
        self.train_start_btn = QPushButton("🚀 開始訓練")
        self.train_start_btn.clicked.connect(self.start_training)
        self.train_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        layout.addWidget(self.train_start_btn)
        
        self.train_stop_btn = QPushButton("⏹️ 停止訓練")
        self.train_stop_btn.clicked.connect(self.stop_training)
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.setStyleSheet("""
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
        layout.addWidget(self.train_stop_btn)
        
        return group
        
    def browse_train_dataset(self):
        """瀏覽訓練資料集"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇資料集資料夾"
        )
        if folder_path:
            config_file = Path(folder_path) / "data_config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    channels = config_data.get('channels', 3)
                    nc = config_data.get('nc', 1)
                    
                    display_name = f"{Path(folder_path).name} ({channels}通道, {nc}類別)"
                    self.train_dataset_combo.addItem(display_name, folder_path)
                    self.train_dataset_combo.setCurrentText(display_name)
                    
                    self.log(f"[OK] 已添加資料集: {Path(folder_path).name}")
                    self.update_train_dataset_info()
                    
                except Exception as e:
                    self.log(f"[WARNING] 讀取配置文件失敗: {e}")
            else:
                self.log("[WARNING] 選擇的資料夾不包含data_config.yaml文件")
                
    def auto_find_train_dataset(self):
        """自動尋找訓練資料集"""
        self.train_dataset_combo.clear()
        
        try:
            dataset_dirs = list(Path("Dataset").glob("dataset_*"))
            standard_datasets = []
            
            for dataset_dir in dataset_dirs:
                config_file = dataset_dir / 'data_config.yaml'
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)
                        
                        channels = config_data.get('channels', 3)
                        nc = config_data.get('nc', 1)
                        
                        standard_datasets.append({
                            'path': str(dataset_dir),
                            'name': dataset_dir.name,
                            'channels': channels,
                            'nc': nc
                        })
                    except Exception as e:
                        self.log(f"[WARNING] 讀取配置失敗 {dataset_dir.name}: {e}")
                        continue
            
            if standard_datasets:
                standard_datasets.sort(
                    key=lambda x: Path(x['path']).stat().st_mtime,
                    reverse=True
                )
                
                for dataset in standard_datasets:
                    display_name = (
                        f"{dataset['name']} "
                        f"({dataset['channels']}通道, {dataset['nc']}類別)"
                    )
                    self.train_dataset_combo.addItem(display_name, dataset['path'])
                
                self.log(f"[OK] 找到 {len(standard_datasets)} 個有效資料集")
            else:
                self.log("[WARNING] 未找到有效的訓練資料集")
                
        except Exception as e:
            self.log(f"[ERROR] 自動尋找資料集失敗: {e}")
            
    def update_train_dataset_info(self):
        """更新訓練資料集資訊"""
        dataset_path = self.train_dataset_combo.currentData()
        if not dataset_path:
            return
        
        try:
            config_file = Path(dataset_path) / "data_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                channels = config_data.get('channels', 3)
                nc = config_data.get('nc', 1)
                
                self.train_dataset_status.setText(
                    f"✓ 通道: {channels} | 類別: {nc}"
                )
                self.train_dataset_status.setStyleSheet(
                    "color: #28a745; font-size: 11px;"
                )
        except Exception as e:
            self.train_dataset_status.setText(f"✗ 讀取失敗: {str(e)}")
            self.train_dataset_status.setStyleSheet(
                "color: #dc3545; font-size: 11px;"
            )
            
    def start_training(self):
        """開始訓練"""
        # 獲取資料集路徑
        dataset_path = self.train_dataset_combo.currentData()
        if not dataset_path:
            self.log("[WARNING] 請選擇資料集")
            return
        
        # 檢查資料集
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            self.log("[WARNING] 資料集資料夾不存在")
            return
        
        config_file = dataset_path / "data_config.yaml"
        if not config_file.exists():
            self.log("[WARNING] 資料集中未找到data_config.yaml文件")
            return
        
        # 獲取模型路徑
        selected_model = self.model_file_combo.currentData()
        if not selected_model or not Path(selected_model).exists():
            self.log("[WARNING] 請選擇有效的模型文件")
            return
        
        # 禁用按鈕
        self.train_start_btn.setEnabled(False)
        self.train_stop_btn.setEnabled(True)
        self.show_progress(True)
        
        # 獲取訓練參數
        training_mode = 'retrain' if self.retrain_radio.isChecked() else 'pretrained'
        epochs = self.epochs_spin.value()
        learning_rate = self.learning_rate_spin.value() * 0.001
        batch_size = self.batch_size_spin.value()
        imgsz = self.imgsz_spin.value()
        
        self.log(f"🎯 訓練模式: {'重新訓練 (YAML)' if training_mode == 'retrain' else '預訓練模型 (PT)'}")
        self.log(f"🎯 訓練參數: 輪數={epochs}, 學習率={learning_rate}, 批次={batch_size}")
        
        # 導入統一的工作線程
        from gui.workers import WorkerThread
        
        # 創建默認高級參數（如果沒有設置）
        if not self.advanced_params:
            self._create_additional_params()
            self.advanced_params = {
                'scale': self.scale_spin.value() * 0.01,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': self.copy_paste_spin.value() * 0.01,
                'hsv_h': self.hsv_h_spin.value() * 0.01,
                'hsv_s': self.hsv_s_spin.value() * 0.01,
                'hsv_v': self.hsv_v_spin.value() * 0.01,
                'bgr': self.bgr_spin.value() * 0.01,
                'auto_augment': self.auto_augment_combo.currentText() if self.auto_augment_combo.currentText() != 'None' else None,
                'degrees': self.degrees_spin.value(),
                'translate': self.translate_spin.value() * 0.01,
                'shear': self.shear_spin.value() * 0.01,
                'perspective': self.perspective_spin.value() * 0.01,
                'flipud': self.flipud_spin.value() * 0.01,
                'fliplr': self.fliplr_spin.value() * 0.01,
                'erasing': self.erasing_spin.value() * 0.01,
                'crop_fraction': self.crop_fraction_spin.value() * 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'beta1': 0.9,
                'beta2': 0.999,
                'lr_scheduler': 'auto',
                'lr_decay': 0.1,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'val_frequency': 1,
                'val_iters': 32,
                'early_stopping_patience': 50,
                'early_stopping_min_delta': 0.001,
                'device': 'auto',
                'multi_gpu': False,
                'gpu_memory_optimization': True,
                'data_loading_optimization': True,
                'close_mosaic': self.close_mosaic_spin.value(),
                'single_cls': False,
                'cache': False,
                'resume': False
            }
        
        # 創建工作線程 - 使用新的任務類型
        task_type = 'train_yaml' if training_mode == 'retrain' else 'train_pretrained'
        self.worker_thread = WorkerThread(
            task_type=task_type,
            config_path=str(config_file),
            model_file=selected_model,
            model_size=self.train_model_size_combo.currentText().split()[0] if training_mode == 'retrain' else None,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            imgsz=imgsz,
            save_period=self.save_period_spin.value(),
            # 數據增強參數
            scale=self.advanced_params.get('scale', 0.5),
            mosaic=self.advanced_params.get('mosaic', 1.0),
            mixup=self.advanced_params.get('mixup', 0.0),
            copy_paste=self.advanced_params.get('copy_paste', 0.0),
            hsv_h=self.advanced_params.get('hsv_h', 0.015),
            hsv_s=self.advanced_params.get('hsv_s', 0.7),
            hsv_v=self.advanced_params.get('hsv_v', 0.4),
            bgr=self.advanced_params.get('bgr', 0.0),
            auto_augment=self.advanced_params.get('auto_augment'),
            # 幾何變換參數
            degrees=self.advanced_params.get('degrees', 0),
            translate=self.advanced_params.get('translate', 0.1),
            shear=self.advanced_params.get('shear', 0.0),
            perspective=self.advanced_params.get('perspective', 0.0),
            # 翻轉和裁剪參數
            flipud=self.advanced_params.get('flipud', 0.0),
            fliplr=self.advanced_params.get('fliplr', 0.5),
            erasing=self.advanced_params.get('erasing', 0.0),
            crop_fraction=self.advanced_params.get('crop_fraction', 1.0),
            # 優化器參數
            weight_decay=self.advanced_params.get('weight_decay', 0.0005),
            momentum=self.advanced_params.get('momentum', 0.937),
            beta1=self.advanced_params.get('beta1', 0.9),
            beta2=self.advanced_params.get('beta2', 0.999),
            # 學習率調度參數
            lr_scheduler=self.advanced_params.get('lr_scheduler', 'auto'),
            lr_decay=self.advanced_params.get('lr_decay', 0.1),
            warmup_epochs=self.advanced_params.get('warmup_epochs', 3),
            warmup_momentum=self.advanced_params.get('warmup_momentum', 0.8),
            # 驗證參數
            val_frequency=self.advanced_params.get('val_frequency', 1),
            val_iters=self.advanced_params.get('val_iters', 32),
            early_stopping_patience=self.advanced_params.get('early_stopping_patience', 50),
            early_stopping_min_delta=self.advanced_params.get('early_stopping_min_delta', 0.001),
            # 設備參數
            device=self.advanced_params.get('device', 'auto'),
            multi_gpu=self.advanced_params.get('multi_gpu', False),
            gpu_memory_optimization=self.advanced_params.get('gpu_memory_optimization', True),
            data_loading_optimization=self.advanced_params.get('data_loading_optimization', True),
            # 其他高級參數
            close_mosaic=self.advanced_params.get('close_mosaic', 10),
            single_cls=self.advanced_params.get('single_cls', False),
            cache=self.advanced_params.get('cache', False),
            resume=self.advanced_params.get('resume', False),
            workers=self.workers_spin.value(),
            optimizer=self.optimizer_combo.currentText(),
            amp=self.amp_checkbox.isChecked()
        )
        self.worker_thread.progress.connect(lambda msg: self.update_status(msg))
        self.worker_thread.finished.connect(self.on_training_finished)
        self.worker_thread.log_message.connect(lambda msg: self.log(msg))
        self.worker_thread.start()
        
    def stop_training(self):
        """停止訓練"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.show_progress(False)
        self.log("⏹️ 訓練已停止")
        
    def on_training_finished(self, success, message):
        """訓練完成回調"""
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.show_progress(False)
        
        if success:
            self.log(f"[SUCCESS] 訓練完成: {message}")
            QMessageBox.information(
                self.parent, "成功 Success",
                f"訓練完成！Training completed!\n\n{message}"
            )
        else:
            self.log(f"[ERROR] 訓練失敗: {message}")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"訓練失敗 Training failed:\n{message}"
            )
    
    def on_training_mode_changed(self, checked):
        """訓練模式切換時的處理"""
        if checked:  # Pretrained mode is checked
            # 顯示PT文件，隱藏模型大小選擇
            self.train_model_size_label.setVisible(False)
            self.train_model_size_combo.setVisible(False)
            self.refresh_model_list()
        else:  # Retrain mode
            # 顯示YAML文件，顯示模型大小選擇
            self.train_model_size_label.setVisible(True)
            self.train_model_size_combo.setVisible(True)
            self.refresh_model_list()
    
    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_file_combo.clear()
        
        try:
            if self.pretrained_radio.isChecked():
                # 預訓練模式 - 只從 Model_file/PT_File 載入PT文件
                pt_files = list(Path("Model_file/PT_File").glob("*.pt"))
                
                if pt_files:
                    # 按文件名排序（字母順序）
                    pt_files.sort(key=lambda x: x.name)
                    
                    for pt_file in pt_files:
                        file_size = pt_file.stat().st_size / (1024 * 1024)
                        display_name = f"{pt_file.name} ({file_size:.1f} MB)"
                        self.model_file_combo.addItem(display_name, str(pt_file))
                    
                    self.log(f"[OK] 找到 {len(pt_files)} 個PT模型文件")
                else:
                    self.log("[WARNING] 未找到PT模型文件")
                    
            else:
                # 重新訓練模式 - 載入YAML文件
                yaml_files = list(Path("Model_file/YAML").glob("*.yaml"))
                
                if yaml_files:
                    for yaml_file in yaml_files:
                        display_name = f"{yaml_file.name}"
                        self.model_file_combo.addItem(display_name, str(yaml_file))
                    
                    self.log(f"[OK] 找到 {len(yaml_files)} 個YAML配置文件")
                else:
                    self.log("[WARNING] 未找到YAML配置文件")
                    
        except Exception as e:
            self.log(f"[ERROR] 刷新模型列表失敗: {e}")
    
    def start_auto_refresh(self):
        """啟動自動刷新功能"""
        try:
            from PyQt5.QtCore import QTimer
            
            # 創建定時器，每30秒自動刷新一次
            self.auto_refresh_timer = QTimer()
            self.auto_refresh_timer.timeout.connect(self.auto_refresh)
            self.auto_refresh_timer.start(30000)  # 30秒
            
            # 立即執行一次自動刷新
            self.auto_refresh()
            
            self.log("✅ 自動刷新功能已啟用 (30秒間隔)")
        except Exception as e:
            self.log(f"[WARNING] 啟動自動刷新失敗: {e}")
    
    def stop_auto_refresh(self):
        """停止自動刷新功能"""
        if self.auto_refresh_timer:
            self.auto_refresh_timer.stop()
            self.auto_refresh_timer = None
            self.log("⏹️ 自動刷新功能已停止")
    
    def auto_refresh(self):
        """執行自動刷新"""
        try:
            # 自動刷新模型列表
            if hasattr(self, 'model_file_combo'):
                self.refresh_model_list()
            
            # 自動刷新資料集列表
            if hasattr(self, 'train_dataset_combo'):
                self.auto_find_train_dataset()
                
        except Exception as e:
            self.log(f"[WARNING] 自動刷新執行失敗: {e}")
    
    def toggle_auto_refresh(self):
        """切換自動刷新功能"""
        if self.auto_refresh_timer and self.auto_refresh_timer.isActive():
            # 停止自動刷新
            self.stop_auto_refresh()
            self.auto_refresh_btn.setText("⏰ 啟用自動刷新")
            self.auto_refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    font-weight: bold;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
        else:
            # 啟動自動刷新
            self.start_auto_refresh()
            self.auto_refresh_btn.setText("⏰ 停止自動刷新")
            self.auto_refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    font-weight: bold;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)