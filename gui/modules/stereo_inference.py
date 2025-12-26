"""
立体视觉推理模块
Stereo Inference Module
处理 RAFT-Stereo 模型的推理预测功能
"""

import os
import sys
import glob
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QSpinBox, QComboBox, QTextEdit,
                            QFileDialog, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt
from .base_module import BaseModule

# 添加Code目录到Python路径
code_dir = Path(__file__).parent.parent.parent / "Code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))


class StereoInferenceModule(BaseModule):
    """立体视觉推理功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        
    def find_latest_model(self):
        """自動查找最新的訓練模型"""
        all_model_files = []
        
        # 1. 查找 runs 目錄下所有的 checkpoints（支持嵌套目錄）
        # 使用 pathlib 進行遞歸搜索，更可靠
        runs_dir = Path("runs")
        if runs_dir.exists():
            try:
                # 查找所有可能的 checkpoints 目錄
                checkpoint_dirs = [
                    runs_dir.glob("raft_stereo_*/checkpoints"),
                    runs_dir.glob("raft_stereo_*/checkpoints/*"),  # 嵌套目錄
                    runs_dir.glob("checkpoints"),
                    runs_dir.glob("checkpoints/*"),  # 嵌套目錄
                ]
                
                for dir_pattern in checkpoint_dirs:
                    for checkpoint_dir in dir_pattern:
                        if checkpoint_dir.is_dir():
                            # 在該目錄下查找所有 .pth 文件
                            pth_files = list(checkpoint_dir.rglob("*.pth"))
                            all_model_files.extend([str(p) for p in pth_files])
            except Exception as e:
                # 如果搜索失敗，嘗試使用 glob
                try:
                    patterns = [
                        "runs/raft_stereo_*/checkpoints/**/*.pth",
                        "runs/raft_stereo_*/checkpoints/*.pth",
                        "runs/checkpoints/stereo_training/*.pth",
                        "runs/checkpoints/**/*.pth",
                    ]
                    for pattern in patterns:
                        files = glob.glob(pattern, recursive=True)
                        all_model_files.extend(files)
                except Exception:
                    pass
        
        # 2. 查找 Model_file 目錄下的模型
        model_dirs = [
            Path("Model_file/Stereo_Vision"),
            Path("Model_file/PTH_File"),  # 向後兼容
            Path("Model_file"),
        ]
        
        for model_dir in model_dirs:
            if model_dir.exists():
                try:
                    pth_files = list(model_dir.glob("*.pth"))
                    all_model_files.extend([str(p) for p in pth_files])
                except Exception as e:
                    continue
        
        # 3. 過濾並選擇最新的模型
        if all_model_files:
            # 去重並過濾掉不存在的文件
            unique_files = list(set(all_model_files))
            valid_files = [f for f in unique_files if os.path.exists(f)]
            
            if valid_files:
                # 按修改時間排序，返回最新的
                try:
                    latest = max(valid_files, key=os.path.getmtime)
                    return latest
                except Exception as e:
                    # 如果獲取修改時間失敗，返回第一個找到的文件
                    return valid_files[0]
        
        return None
        
    def create_tab(self):
        """创建立体视觉推理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型选择
        model_group = QGroupBox("模型設置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型文件 (.pth):"), 0, 0)
        self.stereo_model_edit = QLineEdit()
        self.stereo_model_edit.setPlaceholderText("選擇訓練好的 RAFT-Stereo 模型文件")
        model_layout.addWidget(self.stereo_model_edit, 1, 0)
        
        self.stereo_model_btn = QPushButton("瀏覽")
        self.stereo_model_btn.clicked.connect(self.browse_stereo_model)
        model_layout.addWidget(self.stereo_model_btn, 1, 1)
        
        self.auto_find_model_btn = QPushButton("🔍 自動查找")
        self.auto_find_model_btn.setToolTip("自動查找最新的訓練模型")
        self.auto_find_model_btn.clicked.connect(self.auto_find_model)
        model_layout.addWidget(self.auto_find_model_btn, 1, 2)
        
        layout.addWidget(model_group)
        
        # 图像输入设置
        input_group = QGroupBox("圖像輸入設置")
        input_layout = QGridLayout(input_group)
        
        input_layout.addWidget(QLabel("左圖像路徑:"), 0, 0)
        self.left_imgs_edit = QLineEdit()
        self.left_imgs_edit.setPlaceholderText("左圖像文件或路徑模式 (支持通配符 *.png)")
        input_layout.addWidget(self.left_imgs_edit, 1, 0)
        
        self.left_imgs_btn = QPushButton("瀏覽")
        self.left_imgs_btn.clicked.connect(self.browse_left_images)
        input_layout.addWidget(self.left_imgs_btn, 1, 1)
        
        input_layout.addWidget(QLabel("右圖像路徑:"), 2, 0)
        self.right_imgs_edit = QLineEdit()
        self.right_imgs_edit.setPlaceholderText("右圖像文件或路徑模式 (支持通配符 *.png)")
        input_layout.addWidget(self.right_imgs_edit, 3, 0)
        
        self.right_imgs_btn = QPushButton("瀏覽")
        self.right_imgs_btn.clicked.connect(self.browse_right_images)
        input_layout.addWidget(self.right_imgs_btn, 3, 1)
        
        layout.addWidget(input_group)
        
        # 推理参数
        params_group = QGroupBox("推理參數")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("推理迭代次數:"), 0, 0)
        self.valid_iters_spin = QSpinBox()
        self.valid_iters_spin.setRange(1, 128)
        self.valid_iters_spin.setValue(32)
        self.valid_iters_spin.setToolTip("迭代次數越多，精度越高但速度越慢")
        params_layout.addWidget(self.valid_iters_spin, 0, 1)
        
        params_layout.addWidget(QLabel("混合精度:"), 0, 2)
        self.mixed_precision_check = QCheckBox("啟用")
        self.mixed_precision_check.setChecked(True)
        self.mixed_precision_check.setToolTip("使用混合精度可以加快推理速度")
        params_layout.addWidget(self.mixed_precision_check, 0, 3)
        
        params_layout.addWidget(QLabel("保存 NumPy 數組:"), 1, 0)
        self.save_numpy_check = QCheckBox("啟用")
        self.save_numpy_check.setChecked(False)
        self.save_numpy_check.setToolTip("同時保存 .npy 格式的視差數據")
        params_layout.addWidget(self.save_numpy_check, 1, 1)
        
        params_layout.addWidget(QLabel("輸出格式:"), 1, 2)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["PNG", "JPG", "JPEG", "TIFF", "BMP", "PFM"])
        self.output_format_combo.setCurrentText("PNG")
        self.output_format_combo.setToolTip("選擇視差圖輸出格式\nPFM: 原始數據（精確分析）\nPNG/JPG: 彩色可視化")
        params_layout.addWidget(self.output_format_combo, 1, 3)
        
        params_layout.addWidget(QLabel("圖像翻轉:"), 2, 0)
        self.flip_non_pfm_check = QCheckBox("啟用")
        self.flip_non_pfm_check.setChecked(False)
        self.flip_non_pfm_check.setToolTip("對非PFM格式進行水平和垂直翻轉\n（PFM格式保持原始數據，不翻轉）")
        params_layout.addWidget(self.flip_non_pfm_check, 2, 1)
        
        layout.addWidget(params_group)
        
        # 输出设置
        output_group = QGroupBox("輸出設置")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出目錄:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("留空則在當前目錄創建，選擇目錄則在該目錄下創建 stereo_inference_時間戳 子目錄")
        self.output_dir_edit.setText("")  # 默認留空，自動生成
        output_layout.addWidget(self.output_dir_edit, 1, 0)
        
        self.output_dir_btn = QPushButton("瀏覽")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_dir_btn, 1, 1)
        
        layout.addWidget(output_group)
        
        # 推理说明
        info_group = QGroupBox("推理說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        立體視覺推理功能說明：
        
        1. 模型文件：選擇訓練好的 RAFT-Stereo 模型 (.pth 文件)
           - 可以手動選擇或使用「自動查找」功能
           - 自動查找會優先查找 runs/raft_stereo_*/checkpoints/ 目錄
        
        2. 圖像輸入：
           - 左圖像和右圖像必須成對出現
           - 支持單個文件或路徑模式（使用通配符 *.png）
           - 例如：Dataset/dataset_Stereo_20251215/Img0/test/*.png
        
        3. 推理參數：
           - 推理迭代次數：建議值 16-32，更多迭代可提高精度但速度更慢
           - 混合精度：啟用可加快推理速度，建議開啟
           - 保存 NumPy 數組：可選，用於後續分析
           - 輸出格式：選擇視差圖保存格式
           - 圖像翻轉：僅對非PFM格式生效，進行水平和垂直翻轉
        
        4. 輸出目錄：
           - 留空：自動在當前目錄創建 stereo_inference_時間戳 目錄
           - 選擇目錄：在選擇的目錄下創建 stereo_inference_時間戳 子目錄
        
        5. 輸出格式說明：
           ✨ PFM (推薦用於精確分析):
              - 保存原始浮點數視差值，無損失
              - 可用於後續精確計算和分析
              - 始終保持原始數據方向（不受翻轉選項影響）
           
           🎨 PNG/JPG/TIFF/BMP (推薦用於可視化):
              - 使用 jet colormap 進行彩色可視化
              - 適合直觀查看視差圖效果
              - 可選擇是否進行圖像翻轉
              - PNG: 無損壓縮，質量最好
              - JPG/JPEG: 有損壓縮，文件最小
              - TIFF: 無損壓縮，支持高質量
              - BMP: 無壓縮，文件最大
        
        💡 提示：
           - 精確數值分析 → 選擇 PFM 格式
           - 可視化查看 → 選擇 PNG/JPG 格式
           - 圖像方向有問題 → 啟用「圖像翻轉」（僅對非PFM格式）
        
        支持的輸入圖像格式：.png, .jpg, .jpeg, .bmp
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(250)
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
    
    def auto_find_model(self):
        """自動查找最新的模型"""
        self.log("🔍 正在自動查找最新的訓練模型...")
        model_path = self.find_latest_model()
        if model_path:
            # 轉換為絕對路徑以便顯示
            abs_path = os.path.abspath(model_path)
            self.stereo_model_edit.setText(abs_path)
            self.log(f"✅ 自動找到模型: {Path(model_path).name}")
            self.log(f"   完整路徑: {abs_path}")
        else:
            self.log("⚠️  未找到訓練模型，請手動選擇")
            self.log("   請確保模型文件在以下位置之一:")
            self.log("   - runs/raft_stereo_*/checkpoints/**/*.pth")
            self.log("   - runs/checkpoints/stereo_training/*.pth")
            self.log("   - Model_file/Stereo_Vision/*.pth")
            QMessageBox.warning(
                self.parent, "警告 Warning",
                "未找到訓練模型\n請手動選擇模型文件或確保模型在以下位置之一:\n\n"
                "- runs/raft_stereo_*/checkpoints/**/*.pth\n"
                "- runs/checkpoints/stereo_training/*.pth\n"
                "- Model_file/Stereo_Vision/*.pth"
            )
    
    def browse_stereo_model(self):
        """浏览立体视觉模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "選擇 RAFT-Stereo 模型文件", ".",
            "PyTorch模型 (*.pth);;所有文件 (*)"
        )
        if file_path:
            self.stereo_model_edit.setText(file_path)
            self.log(f"[OK] 已選擇模型: {Path(file_path).name}")
    
    def browse_left_images(self):
        """浏览左图像"""
        # 支持选择单个文件或文件夹
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "選擇左圖像文件", ".",
            "圖像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )
        if file_path:
            self.left_imgs_edit.setText(file_path)
    
    def browse_right_images(self):
        """浏览右图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "選擇右圖像文件", ".",
            "圖像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )
        if file_path:
            self.right_imgs_edit.setText(file_path)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇輸出目錄"
        )
        if folder_path:
            self.output_dir_edit.setText(folder_path)
    
    def start_inference(self):
        """开始立体视觉推理"""
        # 验证模型路径
        model_path = self.stereo_model_edit.text()
        if not model_path:
            # 尝试自动查找
            self.log("未指定模型路徑，嘗試自動查找...")
            model_path = self.find_latest_model()
            if model_path:
                self.stereo_model_edit.setText(model_path)
                self.log(f"✅ 自動找到模型: {Path(model_path).name}")
            else:
                self.log("[WARNING] 請選擇模型文件或確保有訓練好的模型")
                QMessageBox.warning(self.parent, "警告 Warning", "請選擇模型文件")
                return
        
        if not Path(model_path).exists():
            self.log(f"[ERROR] 模型文件不存在: {model_path}")
            QMessageBox.warning(self.parent, "警告 Warning", f"模型文件不存在:\n{model_path}")
            return
        
        # 验证图像路径
        left_imgs = self.left_imgs_edit.text()
        right_imgs = self.right_imgs_edit.text()
        
        if not left_imgs:
            self.log("[WARNING] 請選擇左圖像")
            QMessageBox.warning(self.parent, "警告 Warning", "請選擇左圖像")
            return
        
        if not right_imgs:
            self.log("[WARNING] 請選擇右圖像")
            QMessageBox.warning(self.parent, "警告 Warning", "請選擇右圖像")
            return
        
        # 检查图像文件是否存在
        left_files = sorted(glob.glob(left_imgs, recursive=True))
        right_files = sorted(glob.glob(right_imgs, recursive=True))
        
        if not left_files:
            self.log(f"[ERROR] 未找到左圖像: {left_imgs}")
            QMessageBox.warning(self.parent, "警告 Warning", f"未找到左圖像:\n{left_imgs}")
            return
        
        if not right_files:
            self.log(f"[ERROR] 未找到右圖像: {right_imgs}")
            QMessageBox.warning(self.parent, "警告 Warning", f"未找到右圖像:\n{right_imgs}")
            return
        
        if len(left_files) != len(right_files):
            self.log(f"[WARNING] 左圖像數量 ({len(left_files)}) 與右圖像數量 ({len(right_files)}) 不匹配")
            reply = QMessageBox.question(
                self.parent, "確認 Continue",
                f"左圖像數量 ({len(left_files)}) 與右圖像數量 ({len(right_files)}) 不匹配\n"
                "是否繼續處理前 {min(len(left_files), len(right_files))} 對圖像？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 禁用按钮
        self.inference_start_btn.setEnabled(False)
        self.inference_stop_btn.setEnabled(True)
        self.show_progress(True, text="正在進行立體視覺推理...")
        
        # 获取推理参数
        valid_iters = self.valid_iters_spin.value()
        mixed_precision = self.mixed_precision_check.isChecked()
        save_numpy = self.save_numpy_check.isChecked()
        output_format = self.output_format_combo.currentText().lower()
        flip_non_pfm = self.flip_non_pfm_check.isChecked()
        
        # 創建帶時間戳的輸出目錄
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        
        if self.output_dir_edit.text():
            # 在用戶指定的輸出目錄下創建帶時間戳的子目錄
            base_dir = self.output_dir_edit.text().strip()
            output_dir = str(Path(base_dir) / f"stereo_inference_{timestamp}")
        else:
            # 默認：在當前目錄創建帶時間戳的目錄
            output_dir = f"stereo_inference_{timestamp}"
        
        self.log(f"🔍 開始立體視覺推理")
        self.log(f"   模型: {Path(model_path).name}")
        self.log(f"   左圖像: {len(left_files)} 張")
        self.log(f"   右圖像: {len(right_files)} 張")
        self.log(f"   迭代次數: {valid_iters}")
        self.log(f"   混合精度: {'啟用' if mixed_precision else '禁用'}")
        self.log(f"   輸出格式: {output_format.upper()}")
        if output_format != 'pfm':
            self.log(f"   圖像翻轉: {'啟用' if flip_non_pfm else '禁用'}")
        else:
            self.log(f"   圖像翻轉: 不適用（PFM格式保持原始數據）")
        self.log(f"   輸出目錄: {output_dir}")
        
        # 导入WorkerThread
        from gui.workers import WorkerThread
        
        # 创建工作线程
        self.worker_thread = WorkerThread(
            "stereo_inference",
            model_path=model_path,
            left_imgs=left_imgs,
            right_imgs=right_imgs,
            output_dir=output_dir,
            valid_iters=valid_iters,
            mixed_precision=mixed_precision,
            save_numpy=save_numpy,
            output_format=output_format,
            flip_non_pfm=flip_non_pfm
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
                f"立體視覺推理完成！\nStereo inference completed!\n\n{message}"
            )
        else:
            self.log(f"[ERROR] 推理失敗: {message}")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"推理失敗 Inference failed:\n{message}"
            )
    
    def load_settings(self, settings_manager):
        """加载立体视觉推理模块设置"""
        try:
            stereo_inference_settings = settings_manager.get_section('stereo_inference')
            if stereo_inference_settings:
                if hasattr(self, 'stereo_model_edit') and 'model_path' in stereo_inference_settings:
                    self.stereo_model_edit.setText(stereo_inference_settings['model_path'])
                if hasattr(self, 'left_imgs_edit') and 'left_imgs' in stereo_inference_settings:
                    self.left_imgs_edit.setText(stereo_inference_settings['left_imgs'])
                if hasattr(self, 'right_imgs_edit') and 'right_imgs' in stereo_inference_settings:
                    self.right_imgs_edit.setText(stereo_inference_settings['right_imgs'])
                if hasattr(self, 'output_dir_edit') and 'output_dir' in stereo_inference_settings:
                    self.output_dir_edit.setText(stereo_inference_settings['output_dir'])
                
                # 推理参数
                if hasattr(self, 'valid_iters_spin') and 'valid_iters' in stereo_inference_settings:
                    self.valid_iters_spin.setValue(stereo_inference_settings['valid_iters'])
                if hasattr(self, 'mixed_precision_check') and 'mixed_precision' in stereo_inference_settings:
                    self.mixed_precision_check.setChecked(stereo_inference_settings['mixed_precision'])
                if hasattr(self, 'save_numpy_check') and 'save_numpy' in stereo_inference_settings:
                    self.save_numpy_check.setChecked(stereo_inference_settings['save_numpy'])
                if hasattr(self, 'output_format_combo') and 'output_format' in stereo_inference_settings:
                    self.output_format_combo.setCurrentText(stereo_inference_settings['output_format'].upper())
                if hasattr(self, 'flip_non_pfm_check') and 'flip_non_pfm' in stereo_inference_settings:
                    self.flip_non_pfm_check.setChecked(stereo_inference_settings['flip_non_pfm'])
                
                self.log("✅ 立體視覺推理設置加載完成")
        except Exception as e:
            self.log(f"[WARNING] 加載立體視覺推理設置失敗: {e}")
    
    def save_settings(self, settings_manager):
        """保存立体视觉推理模块设置"""
        try:
            stereo_inference_settings = {}
            
            # 基本參數
            try:
                stereo_inference_settings['model_path'] = self.stereo_model_edit.text()
                stereo_inference_settings['left_imgs'] = self.left_imgs_edit.text()
                stereo_inference_settings['right_imgs'] = self.right_imgs_edit.text()
                stereo_inference_settings['output_dir'] = self.output_dir_edit.text()
            except AttributeError as e:
                self.log(f"[ERROR] 保存基本參數時發生錯誤: {e}")
                stereo_inference_settings['model_path'] = ""
                stereo_inference_settings['left_imgs'] = ""
                stereo_inference_settings['right_imgs'] = ""
                stereo_inference_settings['output_dir'] = ""
            
            # 推理参数
            try:
                stereo_inference_settings['valid_iters'] = self.valid_iters_spin.value()
                stereo_inference_settings['mixed_precision'] = self.mixed_precision_check.isChecked()
                stereo_inference_settings['save_numpy'] = self.save_numpy_check.isChecked()
                stereo_inference_settings['output_format'] = self.output_format_combo.currentText().lower()
                stereo_inference_settings['flip_non_pfm'] = self.flip_non_pfm_check.isChecked()
            except AttributeError as e:
                self.log(f"[ERROR] 保存推理參數時發生錯誤: {e}")
                stereo_inference_settings['valid_iters'] = 32
                stereo_inference_settings['mixed_precision'] = True
                stereo_inference_settings['save_numpy'] = False
                stereo_inference_settings['output_format'] = 'png'
                stereo_inference_settings['flip_non_pfm'] = False
            
            settings_manager.set_section('stereo_inference', stereo_inference_settings)
            self.log("✅ 立體視覺推理設置保存完成")
            
        except Exception as e:
            self.log(f"[WARNING] 保存立體視覺推理設置失敗: {e}")