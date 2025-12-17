"""
模型分析模块
Model Analyzer Module
分析和检查YOLO模型的结构、参数信息
"""

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QComboBox,
                            QTextEdit, QFileDialog, QMessageBox)
from .base_module import BaseModule


class ModelAnalyzerModule(BaseModule):
    """模型分析功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def create_tab(self):
        """创建模型分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型选择
        model_group = QGroupBox("模型選擇")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("選擇模型:"), 0, 0)
        self.analyzer_model_combo = QComboBox()
        self.analyzer_model_combo.setMinimumWidth(300)
        self.analyzer_model_combo.setPlaceholderText("選擇要分析的模型")
        self.analyzer_model_combo.currentTextChanged.connect(self.update_analyzer_model_info)
        model_layout.addWidget(self.analyzer_model_combo, 1, 0)
        
        self.refresh_analyzer_btn = QPushButton("🔄 刷新列表")
        self.refresh_analyzer_btn.clicked.connect(self.refresh_analyzer_model_list)
        model_layout.addWidget(self.refresh_analyzer_btn, 1, 1)
        
        self.browse_analyzer_folder_btn = QPushButton("📁 自定義資料夾")
        self.browse_analyzer_folder_btn.clicked.connect(self.browse_analyzer_model_folder)
        model_layout.addWidget(self.browse_analyzer_folder_btn, 1, 2)
        
        # 文件类型筛选
        model_layout.addWidget(QLabel("文件類型:"), 2, 0)
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(['所有類型', '.pt 文件', '.pth 文件', '.yaml 文件'])
        self.file_type_combo.currentTextChanged.connect(self.apply_file_type_filter)
        model_layout.addWidget(self.file_type_combo, 2, 1, 1, 2)
        
        # 模型信息标签
        self.analyzer_model_info = QLabel("")
        self.analyzer_model_info.setStyleSheet("color: #666666; font-size: 11px;")
        self.analyzer_model_info.setWordWrap(True)
        model_layout.addWidget(self.analyzer_model_info, 3, 0, 1, 3)
        
        layout.addWidget(model_group)
        
        # 分析控制
        control_group = QGroupBox("分析控制")
        control_layout = QHBoxLayout(control_group)
        
        self.analyze_single_btn = QPushButton("🔍 分析選中模型")
        self.analyze_single_btn.clicked.connect(self.analyze_selected_model)
        self.analyze_single_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        control_layout.addWidget(self.analyze_single_btn)
        
        self.analyze_batch_btn = QPushButton("📊 批次分析")
        self.analyze_batch_btn.clicked.connect(self.batch_analyze_models)
        self.analyze_batch_btn.setStyleSheet("""
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
        control_layout.addWidget(self.analyze_batch_btn)
        
        layout.addWidget(control_group)
        
        # 分析结果
        result_group = QGroupBox("分析結果")
        result_layout = QVBoxLayout(result_group)
        
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_result_text.setMinimumHeight(400)
        result_layout.addWidget(self.analysis_result_text)
        
        # 结果操作按钮
        result_btn_layout = QHBoxLayout()
        
        self.save_analysis_btn = QPushButton("💾 保存結果")
        self.save_analysis_btn.clicked.connect(self.save_analysis_results)
        result_btn_layout.addWidget(self.save_analysis_btn)
        
        self.clear_analysis_btn = QPushButton("🗑️ 清空結果")
        self.clear_analysis_btn.clicked.connect(self.clear_analysis_results)
        result_btn_layout.addWidget(self.clear_analysis_btn)
        
        result_layout.addLayout(result_btn_layout)
        
        layout.addWidget(result_group)
        
        self.tab_widget = tab
        return tab
    
    def load_settings(self, settings_manager):
        """加载模型分析模块设置"""
        try:
            analyzer_settings = settings_manager.get_section('model_analyzer')
            if analyzer_settings:
                if hasattr(self, 'analyzer_model_combo') and 'selected_model' in analyzer_settings:
                    # 找到对应的模型
                    for i in range(self.analyzer_model_combo.count()):
                        if self.analyzer_model_combo.itemData(i) == analyzer_settings['selected_model']:
                            self.analyzer_model_combo.setCurrentIndex(i)
                            break
                
                if hasattr(self, 'file_type_combo') and 'file_type_filter' in analyzer_settings:
                    # 找到对应的文件类型筛选器选项
                    for i in range(self.file_type_combo.count()):
                        if self.file_type_combo.itemText(i) == analyzer_settings['file_type_filter']:
                            self.file_type_combo.setCurrentIndex(i)
                            break
                
                self.log("✅ 模型分析设置加载完成")
        except Exception as e:
            self.log(f"[WARNING] 加载模型分析设置失败: {e}")
    
    def save_settings(self, settings_manager):
        """保存模型分析模块设置"""
        try:
            analyzer_settings = {}
            
            if hasattr(self, 'analyzer_model_combo'):
                selected_model = self.analyzer_model_combo.currentData()
                if selected_model:
                    analyzer_settings['selected_model'] = selected_model
            
            if hasattr(self, 'file_type_combo'):
                analyzer_settings['file_type_filter'] = self.file_type_combo.currentText()
            
            settings_manager.set_section('model_analyzer', analyzer_settings)
            self.log("✅ 模型分析设置保存完成")
            
        except Exception as e:
            self.log(f"[WARNING] 保存模型分析设置失败: {e}")
        
    def refresh_analyzer_model_list(self):
        """刷新模型列表"""
        self.analyzer_model_combo.clear()
        
        try:
            # 搜索模型文件（.pt 和 .pth）
            model_files = []
            # Model_file/PT_File
            model_files.extend(Path("Model_file/PT_File").glob("*.pt"))
            model_files.extend(Path("Model_file/PT_File").glob("*.pth"))
            # Model_file/Stereo_Vision
            stereo_dir = Path("Model_file/Stereo_Vision")
            if stereo_dir.exists():
                model_files.extend(stereo_dir.glob("*.pt"))
                model_files.extend(stereo_dir.glob("*.pth"))
            # runs 目錄
            runs_dir = Path("runs")
            if runs_dir.exists():
                model_files.extend(runs_dir.rglob("*.pt"))
                model_files.extend(runs_dir.rglob("*.pth"))

            if model_files:
                # 按修改时间排序
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for model_file in model_files:
                    file_size = model_file.stat().st_size / (1024 * 1024)
                    display_name = f"{model_file.name} ({file_size:.1f} MB)"
                    self.analyzer_model_combo.addItem(display_name, str(model_file))
                
                self.log(f"[OK] 找到 {len(model_files)} 個模型文件")
            else:
                self.log("[WARNING] 未找到模型文件")
                
            # 添加YAML文件
            yaml_files = list(Path("Model_file/YAML").glob("*.yaml"))
            for yaml_file in yaml_files:
                display_name = f"[YAML] {yaml_file.name}"
                self.analyzer_model_combo.addItem(display_name, str(yaml_file))
                
        except Exception as e:
            self.log(f"[ERROR] 刷新模型列表失敗: {e}")
            
    def browse_analyzer_model_folder(self):
        """浏览自定义模型文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "選擇模型資料夾"
        )
        if folder_path:
            self.scan_custom_folder_for_models(folder_path)
            
    def scan_custom_folder_for_models(self, folder_path):
        """扫描自定义文件夹中的模型"""
        try:
            folder = Path(folder_path)
            model_files = list(folder.rglob("*.pt")) + list(folder.rglob("*.pth"))
            
            if model_files:
                for model_file in model_files:
                    file_size = model_file.stat().st_size / (1024 * 1024)
                    display_name = f"{model_file.name} ({file_size:.1f} MB)"
                    self.analyzer_model_combo.addItem(display_name, str(model_file))
                
                self.log(f"[OK] 從自定義資料夾找到 {len(model_files)} 個模型")
            else:
                self.log("[WARNING] 自定義資料夾中未找到模型文件")
                QMessageBox.information(
                    self.parent, "提示 Info",
                    "未在選擇的資料夾中找到模型文件"
                )
        except Exception as e:
            self.log(f"[ERROR] 掃描自定義資料夾失敗: {e}")
            
    def apply_file_type_filter(self):
        """应用文件类型筛选"""
        filter_type = self.file_type_combo.currentText()
        
        # 保存当前选择
        current_selection = self.analyzer_model_combo.currentData()
        
        # 清空并重新填充
        self.refresh_analyzer_model_list()
        
        # 应用筛选
        if filter_type == '.pt 文件':
            for i in range(self.analyzer_model_combo.count() - 1, -1, -1):
                data_path = self.analyzer_model_combo.itemData(i)
                if not data_path or Path(data_path).suffix.lower() != '.pt':
                    self.analyzer_model_combo.removeItem(i)
        elif filter_type == '.pth 文件':
            for i in range(self.analyzer_model_combo.count() - 1, -1, -1):
                data_path = self.analyzer_model_combo.itemData(i)
                if not data_path or Path(data_path).suffix.lower() != '.pth':
                    self.analyzer_model_combo.removeItem(i)
        elif filter_type == '.yaml 文件':
            # 只保留YAML文件
            for i in range(self.analyzer_model_combo.count() - 1, -1, -1):
                if '[YAML]' not in self.analyzer_model_combo.itemText(i):
                    self.analyzer_model_combo.removeItem(i)
        
        # 尝试恢复之前的选择
        if current_selection:
            for i in range(self.analyzer_model_combo.count()):
                if self.analyzer_model_combo.itemData(i) == current_selection:
                    self.analyzer_model_combo.setCurrentIndex(i)
                    break
                    
    def update_analyzer_model_info(self):
        """更新模型信息"""
        model_path = self.analyzer_model_combo.currentData()
        if not model_path:
            self.analyzer_model_info.setText("")
            return
        
        try:
            model_path = Path(model_path)
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)
                file_type = model_path.suffix
                
                info_text = (
                    f"文件: {model_path.name} | "
                    f"大小: {file_size:.2f} MB | "
                    f"類型: {file_type}"
                )
                self.analyzer_model_info.setText(info_text)
                self.analyzer_model_info.setStyleSheet("color: #28a745; font-size: 11px;")
            else:
                self.analyzer_model_info.setText("✗ 文件不存在")
                self.analyzer_model_info.setStyleSheet("color: #dc3545; font-size: 11px;")
        except Exception as e:
            self.analyzer_model_info.setText(f"✗ 讀取失敗: {str(e)}")
            self.analyzer_model_info.setStyleSheet("color: #dc3545; font-size: 11px;")
            
    def analyze_selected_model(self):
        """分析选中的模型"""
        model_path = self.analyzer_model_combo.currentData()
        if not model_path:
            self.log("[WARNING] 請選擇要分析的模型")
            return
        
        self.log(f"🔍 開始分析模型: {Path(model_path).name}")
        
        try:
            from Code.Read_Model import get_model_info
            
            model_info = get_model_info(model_path)
            
            # 格式化输出
            result_text = f"{'='*60}\n"
            result_text += f"模型分析結果 - {Path(model_path).name}\n"
            result_text += f"{'='*60}\n\n"
            
            for key, value in model_info.items():
                result_text += f"{key}: {value}\n"
            
            result_text += f"\n{'='*60}\n"
            
            # 追加到结果文本
            current_text = self.analysis_result_text.toPlainText()
            if current_text:
                result_text = current_text + "\n\n" + result_text
            
            self.analysis_result_text.setPlainText(result_text)
            self.log(f"[SUCCESS] 模型分析完成: {Path(model_path).name}")
            
        except Exception as e:
            error_msg = f"[ERROR] 分析失敗: {str(e)}"
            self.log(error_msg)
            self.analysis_result_text.append(f"\n{error_msg}\n")
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"模型分析失敗:\n{str(e)}"
            )
            
    def batch_analyze_models(self):
        """批次分析所有模型"""
        if self.analyzer_model_combo.count() == 0:
            self.log("[WARNING] 沒有可分析的模型")
            return
        
        self.log(f"📊 開始批次分析 {self.analyzer_model_combo.count()} 個模型...")
        
        self.analysis_result_text.clear()
        success_count = 0
        fail_count = 0
        
        for i in range(self.analyzer_model_combo.count()):
            model_path = self.analyzer_model_combo.itemData(i)
            model_name = Path(model_path).name
            
            try:
                from Code.Read_Model import get_model_info
                model_info = get_model_info(model_path)
                
                # 格式化输出
                result_text = f"{'='*60}\n"
                result_text += f"[{i+1}/{self.analyzer_model_combo.count()}] {model_name}\n"
                result_text += f"{'='*60}\n"
                
                for key, value in model_info.items():
                    result_text += f"  {key}: {value}\n"
                
                self.analysis_result_text.append(result_text)
                success_count += 1
                
            except Exception as e:
                error_msg = f"\n[ERROR] {model_name}: {str(e)}\n"
                self.analysis_result_text.append(error_msg)
                fail_count += 1
        
        summary = f"\n{'='*60}\n"
        summary += f"批次分析完成\n"
        summary += f"成功: {success_count} | 失敗: {fail_count}\n"
        summary += f"{'='*60}\n"
        
        self.analysis_result_text.append(summary)
        self.log(f"[SUCCESS] 批次分析完成 - 成功: {success_count}, 失敗: {fail_count}")
        
    def save_analysis_results(self):
        """保存分析结果"""
        content = self.analysis_result_text.toPlainText()
        if not content:
            self.log("[WARNING] 沒有分析結果可保存")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "保存分析結果", "analysis_results.txt",
            "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log(f"[SUCCESS] 分析結果已保存: {file_path}")
                QMessageBox.information(
                    self.parent, "成功 Success",
                    f"分析結果已保存\n{file_path}"
                )
            except Exception as e:
                self.log(f"[ERROR] 保存失敗: {e}")
                QMessageBox.critical(
                    self.parent, "錯誤 Error",
                    f"保存失敗:\n{str(e)}"
                )
                
    def clear_analysis_results(self):
        """清空分析结果"""
        self.analysis_result_text.clear()
        self.log("[INFO] 已清空分析結果")

