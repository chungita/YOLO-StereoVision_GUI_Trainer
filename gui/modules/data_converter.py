"""
数据转换模块
Data Conversion Module
处理森林数据集的转换，支持RGB、RGBD和立体视觉数据
"""

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup, QTextEdit,
                            QFileDialog, QMessageBox)
from .base_module import BaseModule


class DataConversionModule(BaseModule):
    """数据转换功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        
    def create_tab(self):
        """创建数据转换标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 源数据路径选择
        source_group = QGroupBox("源數據設置")
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel("Forest數據集路徑:"), 0, 0)
        self.convert_source_edit = QLineEdit()
        self.convert_source_edit.setPlaceholderText("選擇Forest數據集根目錄")
        self.convert_source_edit.setText("D:\\DMD\\Forest")  # 默认路径
        source_layout.addWidget(self.convert_source_edit, 1, 0)
        
        self.convert_source_btn = QPushButton("瀏覽")
        self.convert_source_btn.clicked.connect(self.browse_convert_source)
        source_layout.addWidget(self.convert_source_btn, 1, 1)
        
        # 有效样本数量统计标签（基于label文件数量）
        self.image_count_label = QLabel("📊 有效樣本數量 Valid Samples: --")
        self.image_count_label.setStyleSheet("""
            QLabel {
                color: #0078d4;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: #f0f8ff;
                border: 1px solid #0078d4;
                border-radius: 4px;
            }
        """)
        source_layout.addWidget(self.image_count_label, 2, 0, 1, 2)
        
        # 连接文本改变信号以自动更新图片数量
        self.convert_source_edit.textChanged.connect(self.update_image_count)
        
        layout.addWidget(source_group)
        
        # 输出设置
        output_group = QGroupBox("輸出設置")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出路徑 (可選):"), 0, 0)
        self.convert_output_edit = QLineEdit()
        self.convert_output_edit.setPlaceholderText("留空則使用默認路徑 (dataset_時間戳)")
        output_layout.addWidget(self.convert_output_edit, 1, 0)
        
        self.convert_output_btn = QPushButton("瀏覽")
        self.convert_output_btn.clicked.connect(self.browse_convert_output)
        output_layout.addWidget(self.convert_output_btn, 1, 1)
        
        layout.addWidget(output_group)
        
        # 资料夹数量选择
        folder_count_group = QGroupBox("資料夾數量選擇")
        folder_count_layout = QGridLayout(folder_count_group)
        
        folder_count_layout.addWidget(QLabel("處理資料夾數量:"), 0, 0)
        self.folder_count_spin = QSpinBox()
        self.folder_count_spin.setRange(1, 1000)
        self.folder_count_spin.setValue(1)
        self.folder_count_spin.setSuffix(" 個資料夾")
        self.folder_count_spin.setToolTip("設定要處理的資料夾數量，將按順序處理前N個資料夾")
        # 连接信号以在数量变化时更新图片统计
        self.folder_count_spin.valueChanged.connect(self.update_image_count)
        folder_count_layout.addWidget(self.folder_count_spin, 0, 1)
        
        self.auto_detect_folders_btn = QPushButton("🔍 偵測資料夾")
        self.auto_detect_folders_btn.clicked.connect(self.auto_detect_folders)
        folder_count_layout.addWidget(self.auto_detect_folders_btn, 0, 2)
        
        # 资料夹状态标签
        self.folder_status_label = QLabel("")
        self.folder_status_label.setStyleSheet("color: #666666; font-size: 11px;")
        folder_count_layout.addWidget(self.folder_status_label, 1, 0, 1, 3)
        
        layout.addWidget(folder_count_group)
        
        # 数据集分割比例设置
        split_group = QGroupBox("數據集分割比例 Dataset Split Ratio")
        split_layout = QGridLayout(split_group)
        
        # 导入默认值
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from config.config import TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO
        except ImportError:
            TRAIN_SPLIT_RATIO = 0.80
            VAL_SPLIT_RATIO = 0.15
            TEST_SPLIT_RATIO = 0.05
        
        from PyQt5.QtWidgets import QDoubleSpinBox
        
        split_layout.addWidget(QLabel("訓練集 (Train):"), 0, 0)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setDecimals(2)
        self.train_ratio_spin.setValue(TRAIN_SPLIT_RATIO)
        self.train_ratio_spin.setToolTip("訓練集比例 (0.0 - 1.0)，當前值表示為小數")
        split_layout.addWidget(self.train_ratio_spin, 0, 1)
        
        # 显示百分比标签
        train_percent_label = QLabel(f"({TRAIN_SPLIT_RATIO*100:.0f}%)")
        train_percent_label.setStyleSheet("color: #666666; font-size: 10px;")
        split_layout.addWidget(train_percent_label, 0, 2)
        self.train_percent_label = train_percent_label
        
        split_layout.addWidget(QLabel("驗證集 (Val):"), 1, 0)
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 1.0)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setDecimals(2)
        self.val_ratio_spin.setValue(VAL_SPLIT_RATIO)
        self.val_ratio_spin.setToolTip("驗證集比例 (0.0 - 1.0)，當前值表示為小數")
        split_layout.addWidget(self.val_ratio_spin, 1, 1)
        
        val_percent_label = QLabel(f"({VAL_SPLIT_RATIO*100:.0f}%)")
        val_percent_label.setStyleSheet("color: #666666; font-size: 10px;")
        split_layout.addWidget(val_percent_label, 1, 2)
        self.val_percent_label = val_percent_label
        
        split_layout.addWidget(QLabel("測試集 (Test):"), 2, 0)
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setDecimals(2)
        self.test_ratio_spin.setValue(TEST_SPLIT_RATIO)
        self.test_ratio_spin.setToolTip("測試集比例 (0.0 - 1.0)，當前值表示為小數")
        split_layout.addWidget(self.test_ratio_spin, 2, 1)
        
        test_percent_label = QLabel(f"({TEST_SPLIT_RATIO*100:.0f}%)")
        test_percent_label.setStyleSheet("color: #666666; font-size: 10px;")
        split_layout.addWidget(test_percent_label, 2, 2)
        self.test_percent_label = test_percent_label
        
        # 比例总和显示
        self.split_sum_label = QLabel(f"總和: {TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO + TEST_SPLIT_RATIO:.2f}")
        self.split_sum_label.setStyleSheet("color: #666666; font-size: 11px; font-weight: bold;")
        split_layout.addWidget(self.split_sum_label, 3, 0, 1, 2)
        
        # 连接信号以更新总和和百分比标签
        self.train_ratio_spin.valueChanged.connect(self._update_split_sum)
        self.val_ratio_spin.valueChanged.connect(self._update_split_sum)
        self.test_ratio_spin.valueChanged.connect(self._update_split_sum)
        
        layout.addWidget(split_group)
        
        # 深度图选项（水平排列）
        depth_group = QGroupBox("深度圖選項")
        depth_layout = QHBoxLayout(depth_group)
        
        # 深度图选项按钮组
        self.depth_button_group = QButtonGroup()
        
        self.use_depth_radio = QRadioButton("使用深度圖 (4通道數據)")
        self.use_depth_radio.setChecked(True)
        self.use_depth_radio.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.depth_button_group.addButton(self.use_depth_radio, 0)
        depth_layout.addWidget(self.use_depth_radio)
        
        self.no_depth_radio = QRadioButton("不使用深度圖 (3通道RGB數據)")
        self.no_depth_radio.setStyleSheet("color: #28a745; font-weight: bold;")
        self.depth_button_group.addButton(self.no_depth_radio, 1)
        depth_layout.addWidget(self.no_depth_radio)
        
        self.stereo_radio = QRadioButton("立體視覺數據 (RGB左右視圖+視差圖)")
        self.stereo_radio.setStyleSheet("color: #ff6b35; font-weight: bold;")
        self.depth_button_group.addButton(self.stereo_radio, 2)
        depth_layout.addWidget(self.stereo_radio)
        
        layout.addWidget(depth_group)
        
        # 转换说明
        info_group = QGroupBox("轉換說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        數據轉換功能說明：

        1. 支持兩種模式：
        • 4通道模式：合併RGB圖像和深度圖為4通道NumPy文件
        • 3通道模式：直接複製RGB圖像為標準3通道文件

        2. 自動分割為訓練集、驗證集、測試集（可在上方調整比例）
        3. 生成YOLO格式的標籤文件
        4. 根據predefined_classes.txt，創建data_config.yaml配置文件
        5. 支持自定義輸出路徑

        數據結構要求：
        - Forest_Video_*/Img/Img0_*.png (圖像文件)
        - Forest_Video_*/Img/DepthGT_*.pfm (深度圖文件，4通道模式需要)
        - Forest_Video_*/YOLO_Label/*.txt (標籤文件)
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 转换控制
        control_group = QGroupBox("轉換控制")
        control_layout = QHBoxLayout(control_group)
        
        self.convert_start_btn = QPushButton("🔄 開始轉換")
        self.convert_start_btn.clicked.connect(self.start_convert)
        control_layout.addWidget(self.convert_start_btn)
        
        self.convert_stop_btn = QPushButton("⏹️ 停止轉換")
        self.convert_stop_btn.clicked.connect(self.stop_convert)
        self.convert_stop_btn.setEnabled(False)
        control_layout.addWidget(self.convert_stop_btn)
        
        layout.addWidget(control_group)
        
        self.tab_widget = tab
        return tab
    
    def load_settings(self, settings_manager):
        """加载数据转换模块设置"""
        try:
            convert_settings = settings_manager.get_section('convert')
            if convert_settings:
                if hasattr(self, 'convert_source_edit') and 'source_path' in convert_settings:
                    self.convert_source_edit.setText(convert_settings['source_path'])
                if hasattr(self, 'convert_output_edit') and 'output_path' in convert_settings:
                    self.convert_output_edit.setText(convert_settings['output_path'])
                if hasattr(self, 'folder_count_spin') and 'folder_count' in convert_settings:
                    self.folder_count_spin.setValue(convert_settings['folder_count'])
                
                # 数据集分割比例
                if hasattr(self, 'train_ratio_spin') and 'train_ratio' in convert_settings:
                    self.train_ratio_spin.setValue(convert_settings['train_ratio'])
                if hasattr(self, 'val_ratio_spin') and 'val_ratio' in convert_settings:
                    self.val_ratio_spin.setValue(convert_settings['val_ratio'])
                if hasattr(self, 'test_ratio_spin') and 'test_ratio' in convert_settings:
                    self.test_ratio_spin.setValue(convert_settings['test_ratio'])
                
                # 转换模式
                if hasattr(self, 'use_depth_radio') and 'use_depth' in convert_settings:
                    if convert_settings.get('use_stereo', False):
                        self.stereo_radio.setChecked(True)
                    elif convert_settings['use_depth']:
                        self.use_depth_radio.setChecked(True)
                    else:
                        self.no_depth_radio.setChecked(True)
                
                self.log("✅ 数据转换设置加载完成")
        except Exception as e:
            self.log(f"[WARNING] 加载数据转换设置失败: {e}")
    
    def save_settings(self, settings_manager):
        """保存数据转换模块设置"""
        try:
            convert_settings = {}
            
            # 基本參數
            try:
                convert_settings['source_path'] = self.convert_source_edit.text()
                convert_settings['output_path'] = self.convert_output_edit.text()
                convert_settings['folder_count'] = self.folder_count_spin.value()
                
                # 保存数据集分割比例
                convert_settings['train_ratio'] = self.train_ratio_spin.value()
                convert_settings['val_ratio'] = self.val_ratio_spin.value()
                convert_settings['test_ratio'] = self.test_ratio_spin.value()
                
                self.log("✅ 數據轉換基本參數已保存:")
                self.log(f"   源路徑: {convert_settings['source_path']}")
                self.log(f"   輸出路徑: {convert_settings['output_path']}")
                self.log(f"   資料夾數量: {convert_settings['folder_count']}")
                self.log(f"   分割比例: 訓練={convert_settings['train_ratio']:.2f}, 驗證={convert_settings['val_ratio']:.2f}, 測試={convert_settings['test_ratio']:.2f}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存基本參數時發生錯誤: {e}")
                convert_settings['source_path'] = ""
                convert_settings['output_path'] = ""
                convert_settings['folder_count'] = 1
                # 使用默认值
                try:
                    from config.config import TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO, TEST_SPLIT_RATIO
                    convert_settings['train_ratio'] = TRAIN_SPLIT_RATIO
                    convert_settings['val_ratio'] = VAL_SPLIT_RATIO
                    convert_settings['test_ratio'] = TEST_SPLIT_RATIO
                except ImportError:
                    convert_settings['train_ratio'] = 0.80
                    convert_settings['val_ratio'] = 0.15
                    convert_settings['test_ratio'] = 0.05
            
            # 转换模式
            try:
                if self.stereo_radio.isChecked():
                    convert_settings['use_stereo'] = True
                    convert_settings['use_depth'] = False
                elif self.use_depth_radio.isChecked():
                    convert_settings['use_stereo'] = False
                    convert_settings['use_depth'] = True
                else:
                    convert_settings['use_stereo'] = False
                    convert_settings['use_depth'] = False
                
                self.log(f"✅ 轉換模式已保存: 立體={convert_settings['use_stereo']}, 深度={convert_settings['use_depth']}")
                
            except AttributeError as e:
                self.log(f"[ERROR] 保存轉換模式時發生錯誤: {e}")
                convert_settings['use_stereo'] = False
                convert_settings['use_depth'] = False
            
            settings_manager.set_section('convert', convert_settings)
            self.log("✅ 数据转换设置保存完成")
            
        except Exception as e:
            self.log(f"[WARNING] 保存数据转换设置失败: {e}")
        
    def browse_convert_source(self):
        """浏览转换源路径"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇Forest數據集根目錄"
        )
        if folder_path:
            self.convert_source_edit.setText(folder_path)
            # 文本改变时会自动触发update_image_count
            
    def browse_convert_output(self):
        """浏览转换输出路径"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇輸出路徑"
        )
        if folder_path:
            self.convert_output_edit.setText(folder_path)
            
    def _validate_source_path(self, path_text, show_warning=True):
        """验证源路径"""
        if not path_text:
            if show_warning:
                QMessageBox.warning(
                    self.parent, "警告 Warning",
                    "請選擇源數據路徑 Please select source data path"
                )
            return None
        
        source_path = Path(path_text)
        if not source_path.exists():
            if show_warning:
                QMessageBox.warning(
                    self.parent, "警告 Warning",
                    "源路徑不存在，請檢查路徑是否正確 Source path does not exist"
                )
            return None
        
        return source_path
        
    def update_image_count(self):
        """更新图片数量统计（根据label标签文件数量）"""
        source_text = self.convert_source_edit.text()
        
        if not source_text:
            self.image_count_label.setText("📊 有效樣本數量 Valid Samples: --")
            return
        
        source_path = Path(source_text)
        if not source_path.exists():
            self.image_count_label.setText("📊 有效樣本數量 Valid Samples: 路徑不存在 Path not found")
            self.image_count_label.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #fff5f5;
                    border: 1px solid #dc3545;
                    border-radius: 4px;
                }
            """)
            return
        
        try:
            total_samples = 0
            total_images = 0
            folder_info = ""
            
            # 检测Forest格式资料夹（需要排序以确保一致性）
            forest_folders = sorted([
                f for f in source_path.iterdir()
                if f.is_dir() and f.name.startswith('Forest_Video_')
            ])
            
            if forest_folders:
                # Forest格式 - 统计YOLO_Label中的txt文件数量
                folder_limit = self.folder_count_spin.value() if hasattr(self, 'folder_count_spin') else len(forest_folders)
                folders_to_process = forest_folders[:folder_limit]
                
                for folder in folders_to_process:
                    label_folder = folder / 'YOLO_Label'
                    img_folder = folder / 'Img'
                    
                    if label_folder.exists():
                        # 统计标签文件数量
                        label_files = list(label_folder.glob('*.txt'))
                        total_samples += len(label_files)
                    
                    if img_folder.exists():
                        # 同时统计图片文件数量用于对比
                        img_files = list(img_folder.glob('Img0_*.png')) + list(img_folder.glob('Img0_*.jpg'))
                        total_images += len(img_files)
                
                folder_info = f"{len(folders_to_process)} 個Forest資料夾"
            else:
                # 检查单一资料夹格式 - 统计YOLO_Label中的txt文件数量
                label_folder = source_path / 'YOLO_Label'
                img_folder = source_path / 'Img'
                
                if label_folder.exists():
                    # 统计标签文件数量
                    label_files = list(label_folder.glob('*.txt'))
                    total_samples = len(label_files)
                
                if img_folder.exists():
                    # 同时统计图片文件数量用于对比
                    img_files = set()
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
                    for ext in image_extensions:
                        img_files.update(img_folder.glob(f'*{ext}'))
                        img_files.update(img_folder.glob(f'*{ext.upper()}'))
                    total_images = len(img_files)
                
                folder_info = "單一資料夾模式"
            
            if total_samples > 0:
                # 显示有效样本数量（有标签的）
                if total_images > total_samples:
                    # 如果图片数量多于标签数量，显示警告
                    self.image_count_label.setText(
                        f"📊 有效樣本數量 Valid Samples: {total_samples} 個 samples | 總圖片 Total Images: {total_images} 張 ({folder_info})"
                    )
                    self.image_count_label.setStyleSheet("""
                        QLabel {
                            color: #ffc107;
                            font-size: 12px;
                            font-weight: bold;
                            padding: 8px;
                            background-color: #fffef0;
                            border: 1px solid #ffc107;
                            border-radius: 4px;
                        }
                    """)
                else:
                    # 标签和图片数量匹配
                    self.image_count_label.setText(
                        f"📊 有效樣本數量 Valid Samples: {total_samples} 個 samples ({folder_info})"
                    )
                    self.image_count_label.setStyleSheet("""
                        QLabel {
                            color: #28a745;
                            font-size: 12px;
                            font-weight: bold;
                            padding: 8px;
                            background-color: #f0fff0;
                            border: 1px solid #28a745;
                            border-radius: 4px;
                        }
                    """)
            else:
                self.image_count_label.setText(
                    f"📊 有效樣本數量 Valid Samples: 0 個 samples (未找到標籤文件 No label files found)"
                )
                self.image_count_label.setStyleSheet("""
                    QLabel {
                        color: #dc3545;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 8px;
                        background-color: #fff5f5;
                        border: 1px solid #dc3545;
                        border-radius: 4px;
                    }
                """)
                
        except Exception as e:
            self.image_count_label.setText(f"📊 有效樣本數量 Valid Samples: 統計失敗 Error: {str(e)}")
            self.image_count_label.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #fff5f5;
                    border: 1px solid #dc3545;
                    border-radius: 4px;
                }
            """)
    
    def auto_detect_folders(self):
        """自动侦测资料夹数量"""
        source_path = self._validate_source_path(
            self.convert_source_edit.text()
        )
        if not source_path:
            return
        
        try:
            # 侦测Forest格式资料夹（需要排序以确保一致性）
            forest_folders = sorted([
                f for f in source_path.iterdir()
                if f.is_dir() and f.name.startswith('Forest_Video_')
            ])
            
            if forest_folders:
                self.folder_status_label.setText(
                    f"[OK] 偵測到 {len(forest_folders)} 個Forest資料夾"
                )
                self.folder_status_label.setStyleSheet("color: #28a745; font-size: 11px;")
                self.folder_count_spin.setRange(1, len(forest_folders))
                self.folder_count_spin.setValue(len(forest_folders))
                self.log(f"[SEARCH] 偵測到 {len(forest_folders)} 個Forest資料夾，預設處理全部")
                # 更新图片数量
                self.update_image_count()
            else:
                # 检查是否为单一资料夹格式
                required_folders = ['Img', 'YOLO_Label']
                has_required = all(
                    (source_path / folder).exists()
                    for folder in required_folders
                )
                
                if has_required:
                    self.folder_status_label.setText("[OK] 偵測到單一資料夾格式")
                    self.folder_status_label.setStyleSheet("color: #28a745; font-size: 11px;")
                    self.folder_count_spin.setRange(1, 1)
                    self.folder_count_spin.setValue(1)
                    self.log("[SEARCH] 偵測到單一資料夾格式")
                    # 更新图片数量
                    self.update_image_count()
                else:
                    self.folder_status_label.setText("[ERROR] 未偵測到有效的資料夾格式")
                    self.folder_status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
                    self.folder_count_spin.setRange(1, 1)
                    self.folder_count_spin.setValue(1)
                    self.log("[ERROR] 未偵測到有效的資料夾格式")
                    
        except Exception as e:
            self.folder_status_label.setText(f"[ERROR] 偵測失敗: {str(e)}")
            self.folder_status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
            self.log(f"[ERROR] 偵測資料夾失敗: {e}")
            
    def _toggle_convert_buttons(self, is_running):
        """切换转换按钮状态"""
        self.convert_start_btn.setEnabled(not is_running)
        self.convert_stop_btn.setEnabled(is_running)
        self.show_progress(is_running)
        
    def _update_split_sum(self):
        """更新分割比例总和显示和百分比标签"""
        if hasattr(self, 'train_ratio_spin') and hasattr(self, 'val_ratio_spin') and hasattr(self, 'test_ratio_spin'):
            train_val = self.train_ratio_spin.value()
            val_val = self.val_ratio_spin.value()
            test_val = self.test_ratio_spin.value()
            total = train_val + val_val + test_val
            
            # 更新百分比标签
            if hasattr(self, 'train_percent_label'):
                self.train_percent_label.setText(f"({train_val*100:.0f}%)")
            if hasattr(self, 'val_percent_label'):
                self.val_percent_label.setText(f"({val_val*100:.0f}%)")
            if hasattr(self, 'test_percent_label'):
                self.test_percent_label.setText(f"({test_val*100:.0f}%)")
            
            # 更新总和标签
            color = "#28a745" if abs(total - 1.0) < 0.01 else "#dc3545"
            self.split_sum_label.setText(f"總和: {total:.2f} ({total*100:.0f}%)")
            self.split_sum_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
    
    def _get_conversion_mode_info(self, use_depth, use_stereo):
        """获取转换模式信息"""
        if use_stereo:
            return "立體視覺數據 Stereo Vision Data", "🔄 開始立體視覺數據轉換... Starting stereo data conversion..."
        elif use_depth:
            return "4通道RGBD數據 4-Channel RGBD Data", "🔄 開始4通道數據轉換... Starting 4-channel data conversion..."
        else:
            return "3通道RGB數據 3-Channel RGB Data", "🔄 開始3通道數據轉換... Starting 3-channel data conversion..."
            
    def start_convert(self):
        """开始数据转换"""
        # 验证源路径
        source_path = self._validate_source_path(self.convert_source_edit.text())
        if not source_path:
            return
        
        # 切换按钮状态
        self._toggle_convert_buttons(True)
        
        # 获取深度图选项
        use_depth = self.use_depth_radio.isChecked()
        use_stereo = self.stereo_radio.isChecked()
        
        # 获取资料夹数量限制
        folder_count_limit = self.folder_count_spin.value()
        
        # 获取数据集分割比例
        train_ratio = self.train_ratio_spin.value()
        val_ratio = self.val_ratio_spin.value()
        test_ratio = self.test_ratio_spin.value()
        
        # 验证分割比例总和
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:  # 允许0.01的误差
            QMessageBox.warning(
                self.parent, "警告 Warning",
                f"數據集分割比例總和不為1.0 ({total_ratio:.2f})，將自動調整為1.0\n"
                f"Dataset split ratios sum to {total_ratio:.2f} (not 1.0), will normalize to 1.0"
            )
            # 归一化
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
            self.train_ratio_spin.setValue(train_ratio)
            self.val_ratio_spin.setValue(val_ratio)
            self.test_ratio_spin.setValue(test_ratio)
        
        # 获取转换模式信息
        mode_name, start_msg = self._get_conversion_mode_info(use_depth, use_stereo)
        self.log(start_msg)
        self.log(f"📊 數據集分割比例: 訓練={train_ratio:.2%}, 驗證={val_ratio:.2%}, 測試={test_ratio:.2%}")
        
        # 导入WorkerThread - 需要从主GUI获取
        from yolo_launcher_gui_modular import WorkerThread
        
        # 创建工作线程
        self.worker_thread = WorkerThread(
            "convert",
            source_path=self.convert_source_edit.text(),
            output_path=self.convert_output_edit.text() if self.convert_output_edit.text() else None,
            use_depth=use_depth,
            use_stereo=use_stereo,
            folder_count_limit=folder_count_limit,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        self.worker_thread.progress.connect(lambda msg: self.update_status(msg))
        self.worker_thread.finished.connect(self.on_convert_finished)
        self.worker_thread.log_message.connect(lambda msg: self.log(msg))
        self.worker_thread.start()
        
    def stop_convert(self):
        """停止数据转换"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.log("[INFO] 正在停止轉換... Stopping conversion...")
            self.worker_thread.stop()
            self._toggle_convert_buttons(False)
            self.update_status("轉換已停止 Conversion stopped")
            
    def on_convert_finished(self, success, message):
        """转换完成回调"""
        self._toggle_convert_buttons(False)
        
        if success:
            self.log(f"[SUCCESS] 轉換完成: {message}")
            QMessageBox.information(
                self.parent, "成功 Success",
                f"數據轉換完成！Data conversion completed!\n\n{message}"
            )
        else:
            self.log(f"[ERROR] 轉換失敗: {message}")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"數據轉換失敗 Data conversion failed:\n{message}"
            )

