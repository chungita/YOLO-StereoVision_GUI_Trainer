"""
推理模块
Inference Module
处理YOLO模型的推理预测功能
"""

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QDoubleSpinBox, QComboBox, QTextEdit,
                            QFileDialog, QMessageBox, QRadioButton, QButtonGroup)
from .base_module import BaseModule


class InferenceModule(BaseModule):
    """推理功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        
    def create_tab(self):
        """创建推理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型选择
        model_group = QGroupBox("模型設置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.inference_model_edit = QLineEdit()
        self.inference_model_edit.setPlaceholderText("選擇訓練好的模型文件 (.pt)")
        model_layout.addWidget(self.inference_model_edit, 1, 0)
        
        self.inference_model_btn = QPushButton("瀏覽")
        self.inference_model_btn.clicked.connect(self.browse_inference_model)
        model_layout.addWidget(self.inference_model_btn, 1, 1)
        
        layout.addWidget(model_group)
        
        # 推理模式选择
        mode_group = QGroupBox("推理模式")
        mode_layout = QVBoxLayout(mode_group)
        
        self.inference_mode_group = QButtonGroup()
        
        self.single_image_radio = QRadioButton("單張圖像推理")
        self.single_image_radio.setChecked(True)
        self.single_image_radio.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.inference_mode_group.addButton(self.single_image_radio, 0)
        mode_layout.addWidget(self.single_image_radio)
        
        self.batch_image_radio = QRadioButton("批次圖像推理")
        self.batch_image_radio.setStyleSheet("color: #28a745; font-weight: bold;")
        self.inference_mode_group.addButton(self.batch_image_radio, 1)
        mode_layout.addWidget(self.batch_image_radio)
        
        self.video_radio = QRadioButton("視頻推理")
        self.video_radio.setStyleSheet("color: #ff6b35; font-weight: bold;")
        self.inference_mode_group.addButton(self.video_radio, 2)
        mode_layout.addWidget(self.video_radio)
        
        layout.addWidget(mode_group)
        
        # 数据源选择
        data_group = QGroupBox("數據源")
        data_layout = QGridLayout(data_group)
        
        data_layout.addWidget(QLabel("輸入路徑:"), 0, 0)
        self.inference_data_edit = QLineEdit()
        self.inference_data_edit.setPlaceholderText("選擇圖像文件或資料夾")
        data_layout.addWidget(self.inference_data_edit, 1, 0)
        
        self.inference_data_btn = QPushButton("瀏覽")
        self.inference_data_btn.clicked.connect(self.browse_inference_data)
        data_layout.addWidget(self.inference_data_btn, 1, 1)
        
        layout.addWidget(data_group)
        
        # 推理参数
        params_group = QGroupBox("推理參數")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("置信度閾值:"), 0, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        self.confidence_spin.setDecimals(2)
        params_layout.addWidget(self.confidence_spin, 0, 1)
        
        params_layout.addWidget(QLabel("IOU閾值:"), 0, 2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setDecimals(2)
        params_layout.addWidget(self.iou_spin, 0, 3)
        
        params_layout.addWidget(QLabel("最大檢測數:"), 1, 0)
        self.max_det_spin = QComboBox()
        self.max_det_spin.addItems(['300', '500', '1000', '2000'])
        self.max_det_spin.setCurrentText('300')
        params_layout.addWidget(self.max_det_spin, 1, 1)
        
        layout.addWidget(params_group)
        
        # 输出设置
        output_group = QGroupBox("輸出設置")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出路徑:"), 0, 0)
        self.inference_output_edit = QLineEdit()
        self.inference_output_edit.setPlaceholderText("留空則使用默認路徑 (Predict/Result)")
        output_layout.addWidget(self.inference_output_edit, 1, 0)
        
        self.inference_output_btn = QPushButton("瀏覽")
        self.inference_output_btn.clicked.connect(self.browse_inference_output)
        output_layout.addWidget(self.inference_output_btn, 1, 1)
        
        layout.addWidget(output_group)
        
        # 推理说明
        info_group = QGroupBox("推理說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        推理功能說明：
        
        1. 單張圖像推理：對單張圖像進行目標檢測
        2. 批次圖像推理：對多張圖像批次處理
        3. 視頻推理：對視頻文件進行逐幀檢測
        
        參數說明：
        - 置信度閾值：檢測結果的最小置信度
        - IOU閾值：非極大值抑制的IOU閾值
        - 最大檢測數：單張圖像最多檢測目標數量
        
        支持的輸入格式：
        - 圖像：.jpg, .jpeg, .png, .bmp, .npy
        - 視頻：.mp4, .avi, .mov, .mkv
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(180)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 推理控制
        control_group = QGroupBox("推理控制")
        control_layout = QHBoxLayout(control_group)
        
        self.inference_start_btn = QPushButton("🔍 開始推理")
        self.inference_start_btn.clicked.connect(self.start_inference)
        self.inference_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        control_layout.addWidget(self.inference_start_btn)
        
        self.inference_stop_btn = QPushButton("⏹️ 停止推理")
        self.inference_stop_btn.clicked.connect(self.stop_inference)
        self.inference_stop_btn.setEnabled(False)
        self.inference_stop_btn.setStyleSheet("""
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
        control_layout.addWidget(self.inference_stop_btn)
        
        layout.addWidget(control_group)
        
        self.tab_widget = tab
        return tab
    
    def load_settings(self, settings_manager):
        """加载推理模块设置"""
        try:
            inference_settings = settings_manager.get_section('inference')
            if inference_settings:
                if hasattr(self, 'inference_model_edit') and 'model_path' in inference_settings:
                    self.inference_model_edit.setText(inference_settings['model_path'])
                if hasattr(self, 'inference_data_edit') and 'dataset_path' in inference_settings:
                    self.inference_data_edit.setText(inference_settings['dataset_path'])
                if hasattr(self, 'inference_output_edit') and 'output_path' in inference_settings:
                    self.inference_output_edit.setText(inference_settings['output_path'])
                
                # 推理参数
                if hasattr(self, 'confidence_spin') and 'confidence_threshold' in inference_settings:
                    self.confidence_spin.setValue(inference_settings['confidence_threshold'])
                if hasattr(self, 'iou_spin') and 'iou_threshold' in inference_settings:
                    self.iou_spin.setValue(inference_settings['iou_threshold'])
                if hasattr(self, 'max_det_spin') and 'max_det' in inference_settings:
                    # 找到对应的索引
                    for i in range(self.max_det_spin.count()):
                        if self.max_det_spin.itemText(i) == str(inference_settings['max_det']):
                            self.max_det_spin.setCurrentIndex(i)
                            break
                
                # 推理模式
                if hasattr(self, 'single_image_radio') and 'mode' in inference_settings:
                    mode = inference_settings['mode']
                    if mode == "單張圖像推理":
                        self.single_image_radio.setChecked(True)
                    elif mode == "批次圖像推理":
                        self.batch_image_radio.setChecked(True)
                    elif mode == "視頻推理":
                        self.video_radio.setChecked(True)
                
                self.log("✅ 推理设置加载完成")
        except Exception as e:
            self.log(f"[WARNING] 加载推理设置失败: {e}")
    
    def save_settings(self, settings_manager):
        """保存推理模块设置"""
        try:
            inference_settings = {}
            
            # 基本參數
            try:
                inference_settings['model_path'] = self.inference_model_edit.text()
                inference_settings['dataset_path'] = self.inference_data_edit.text()
                inference_settings['output_path'] = self.inference_output_edit.text()
                
                self.log("✅ 推理基本參數已保存:")
                self.log(f"   模型路徑: {inference_settings['model_path']}")
                self.log(f"   資料集路徑: {inference_settings['dataset_path']}")
                self.log(f"   輸出路徑: {inference_settings['output_path']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存基本參數時發生錯誤: {e}")
                inference_settings['model_path'] = ""
                inference_settings['dataset_path'] = ""
                inference_settings['output_path'] = ""
            
            # 推理参数
            try:
                inference_settings['confidence_threshold'] = self.confidence_spin.value()
                inference_settings['iou_threshold'] = self.iou_spin.value()
                inference_settings['max_det'] = int(self.max_det_spin.currentText())
                
                self.log("✅ 推理參數已保存:")
                self.log(f"   置信度閾值: {inference_settings['confidence_threshold']}")
                self.log(f"   IoU閾值: {inference_settings['iou_threshold']}")
                self.log(f"   最大檢測數: {inference_settings['max_det']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存推理參數時發生錯誤: {e}")
                inference_settings['confidence_threshold'] = 0.25
                inference_settings['iou_threshold'] = 0.45
                inference_settings['max_det'] = 300
            
            # 推理模式
            try:
                if self.single_image_radio.isChecked():
                    inference_settings['mode'] = "單張圖像推理"
                elif self.batch_image_radio.isChecked():
                    inference_settings['mode'] = "批次圖像推理"
                elif self.video_radio.isChecked():
                    inference_settings['mode'] = "視頻推理"
                else:
                    inference_settings['mode'] = "單張圖像推理"  # 默認值
                
                self.log(f"✅ 推理模式已保存: {inference_settings['mode']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存推理模式時發生錯誤: {e}")
                inference_settings['mode'] = "單張圖像推理"
            
            settings_manager.set_section('inference', inference_settings)
            self.log("✅ 推理设置保存完成")
            
        except Exception as e:
            self.log(f"[WARNING] 保存推理设置失败: {e}")
        
    def browse_inference_model(self):
        """浏览推理模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "選擇模型文件", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.inference_model_edit.setText(file_path)
            self.log(f"[OK] 已選擇模型: {Path(file_path).name}")
            
    def browse_inference_data(self):
        """浏览推理数据"""
        if self.single_image_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "選擇圖像文件", ".",
                "圖像文件 (*.jpg *.jpeg *.png *.bmp *.npy)"
            )
            if file_path:
                self.inference_data_edit.setText(file_path)
        elif self.batch_image_radio.isChecked():
            folder_path = QFileDialog.getExistingDirectory(
                self.parent, "選擇圖像資料夾"
            )
            if folder_path:
                self.inference_data_edit.setText(folder_path)
        else:  # 视频推理
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "選擇視頻文件", ".",
                "視頻文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.inference_data_edit.setText(file_path)
                
    def browse_inference_output(self):
        """浏览推理输出路径"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇輸出路徑"
        )
        if folder_path:
            self.inference_output_edit.setText(folder_path)
            
    def start_inference(self):
        """开始推理"""
        # 验证模型路径
        model_path = self.inference_model_edit.text()
        if not model_path:
            self.log("[WARNING] 請選擇模型文件")
            QMessageBox.warning(self.parent, "警告 Warning", "請選擇模型文件")
            return
        
        if not Path(model_path).exists():
            self.log("[ERROR] 模型文件不存在")
            QMessageBox.warning(self.parent, "警告 Warning", "模型文件不存在")
            return
        
        # 验证数据源
        data_path = self.inference_data_edit.text()
        if not data_path:
            self.log("[WARNING] 請選擇輸入數據")
            QMessageBox.warning(self.parent, "警告 Warning", "請選擇輸入數據")
            return
        
        if not Path(data_path).exists():
            self.log("[ERROR] 輸入數據不存在")
            QMessageBox.warning(self.parent, "警告 Warning", "輸入數據不存在")
            return
        
        # 禁用按钮
        self.inference_start_btn.setEnabled(False)
        self.inference_stop_btn.setEnabled(True)
        self.show_progress(True)
        
        # 获取推理参数
        confidence = self.confidence_spin.value()
        iou_threshold = self.iou_spin.value()
        max_det = int(self.max_det_spin.currentText())
        
        # 确定推理模式
        if self.single_image_radio.isChecked():
            inference_mode = "single"
        elif self.batch_image_radio.isChecked():
            inference_mode = "batch"
        else:
            inference_mode = "video"
        
        self.log(f"🔍 開始推理 - 模式: {inference_mode}")
        self.log(f"   置信度: {confidence}, IOU: {iou_threshold}, 最大檢測: {max_det}")
        
        # 导入WorkerThread
        from gui.workers import WorkerThread
        
        # 创建工作线程
        self.worker_thread = WorkerThread(
            "inference",
            model_path=model_path,
            data_path=data_path,
            output_path=self.inference_output_edit.text() if self.inference_output_edit.text() else None,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_det=max_det,
            inference_mode=inference_mode
        )
        self.worker_thread.progress.connect(lambda msg: self.update_status(msg))
        self.worker_thread.finished.connect(self.on_inference_finished)
        self.worker_thread.log_message.connect(lambda msg: self.log(msg))
        self.worker_thread.start()
        
    def stop_inference(self):
        """停止推理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        
        self.inference_start_btn.setEnabled(True)
        self.inference_stop_btn.setEnabled(False)
        self.show_progress(False)
        self.log("⏹️ 推理已停止")
        
    def on_inference_finished(self, success, message):
        """推理完成回调"""
        self.inference_start_btn.setEnabled(True)
        self.inference_stop_btn.setEnabled(False)
        self.show_progress(False)
        
        if success:
            self.log(f"[SUCCESS] 推理完成: {message}")
            QMessageBox.information(
                self.parent, "成功 Success",
                f"推理完成！Inference completed!\n\n{message}"
            )
        else:
            self.log(f"[ERROR] 推理失敗: {message}")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"推理失敗 Inference failed:\n{message}"
            )

