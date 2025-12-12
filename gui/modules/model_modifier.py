"""
模型修改模块
Model Modifier Module  
修改YOLO模型的输入通道数
"""

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QPushButton,
                            QSpinBox, QTextEdit, QFileDialog, QMessageBox)
from .base_module import BaseModule


class ModelModifierModule(BaseModule):
    """模型修改功能模块"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def create_tab(self):
        """创建模型修改标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 输入模型选择
        input_group = QGroupBox("輸入模型")
        input_layout = QGridLayout(input_group)
        
        input_layout.addWidget(QLabel("原始模型文件:"), 0, 0)
        self.modifier_input_model_edit = QLineEdit()
        self.modifier_input_model_edit.setPlaceholderText("選擇要修改的模型文件 (.pt)")
        input_layout.addWidget(self.modifier_input_model_edit, 1, 0)
        
        self.modifier_input_browse_btn = QPushButton("瀏覽")
        self.modifier_input_browse_btn.clicked.connect(self.browse_modifier_input_model)
        input_layout.addWidget(self.modifier_input_browse_btn, 1, 1)
        
        self.analyze_model_btn = QPushButton("🔍 分析模型")
        self.analyze_model_btn.clicked.connect(self.analyze_model_for_modification)
        self.analyze_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
            }
        """)
        input_layout.addWidget(self.analyze_model_btn, 1, 2)
        
        layout.addWidget(input_group)
        
        # 模型信息显示
        info_group = QGroupBox("當前模型信息")
        info_layout = QVBoxLayout(info_group)
        
        self.modifier_model_info_text = QTextEdit()
        self.modifier_model_info_text.setReadOnly(True)
        self.modifier_model_info_text.setMaximumHeight(150)
        self.modifier_model_info_text.setPlaceholderText("模型信息將在分析後顯示...")
        info_layout.addWidget(self.modifier_model_info_text)
        
        layout.addWidget(info_group)
        
        # 修改参数
        params_group = QGroupBox("修改參數")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("目標輸入通道數:"), 0, 0)
        self.target_channels_spin = QSpinBox()
        self.target_channels_spin.setRange(1, 10)
        self.target_channels_spin.setValue(4)
        self.target_channels_spin.setToolTip("設置模型的目標輸入通道數")
        params_layout.addWidget(self.target_channels_spin, 0, 1)
        
        params_layout.addWidget(QLabel("當前通道數:"), 0, 2)
        self.current_channels_label = QLabel("未知")
        self.current_channels_label.setStyleSheet("color: #666666; font-weight: bold;")
        params_layout.addWidget(self.current_channels_label, 0, 3)
        
        layout.addWidget(params_group)
        
        # 输出模型设置
        output_group = QGroupBox("輸出模型")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出模型文件:"), 0, 0)
        self.modifier_output_model_edit = QLineEdit()
        self.modifier_output_model_edit.setPlaceholderText("保存修改後的模型文件")
        output_layout.addWidget(self.modifier_output_model_edit, 1, 0)
        
        self.modifier_output_browse_btn = QPushButton("瀏覽")
        self.modifier_output_browse_btn.clicked.connect(self.browse_modifier_output_model)
        output_layout.addWidget(self.modifier_output_browse_btn, 1, 1)
        
        layout.addWidget(output_group)
        
        # 说明信息
        info_group = QGroupBox("功能說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        模型修改功能說明：
        
        1. 用途：修改YOLO模型的輸入通道數
           - 將3通道RGB模型轉為4通道RGBD模型
           - 將4通道RGBD模型轉為3通道RGB模型
           - 支持其他通道數的轉換
        
        2. 使用步驟：
           - 選擇原始模型文件
           - 點擊「分析模型」查看當前通道數
           - 設置目標輸入通道數
           - 選擇輸出模型路徑
           - 點擊「開始修改」執行轉換
        
        3. 注意事項：
           - 修改後的模型需要重新訓練才能使用
           - 建議保留原始模型備份
           - 修改後模型的權重會被重置
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 操作按钮
        control_group = QGroupBox("操作控制")
        control_layout = QHBoxLayout(control_group)
        
        self.modify_model_btn = QPushButton("🔧 開始修改")
        self.modify_model_btn.clicked.connect(self.modify_model_channels)
        self.modify_model_btn.setStyleSheet("""
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
        control_layout.addWidget(self.modify_model_btn)
        
        self.clear_modifier_btn = QPushButton("🗑️ 清空設置")
        self.clear_modifier_btn.clicked.connect(self.clear_modifier_fields)
        control_layout.addWidget(self.clear_modifier_btn)
        
        layout.addWidget(control_group)
        
        self.tab_widget = tab
        return tab
    
    def load_settings(self, settings_manager):
        """加载模型修改模块设置"""
        try:
            modifier_settings = settings_manager.get_section('model_modifier')
            if modifier_settings:
                if hasattr(self, 'modifier_input_model_edit') and 'input_model' in modifier_settings:
                    self.modifier_input_model_edit.setText(modifier_settings['input_model'])
                if hasattr(self, 'modifier_output_model_edit') and 'output_model' in modifier_settings:
                    self.modifier_output_model_edit.setText(modifier_settings['output_model'])
                if hasattr(self, 'current_channels_label') and 'original_channels' in modifier_settings:
                    self.current_channels_label.setText(str(modifier_settings['original_channels']))
                if hasattr(self, 'target_channels_spin') and 'target_channels' in modifier_settings:
                    self.target_channels_spin.setValue(modifier_settings['target_channels'])
                if hasattr(self, 'weight_method_combo') and 'weight_method' in modifier_settings:
                    # 找到对应的方法
                    for i in range(self.weight_method_combo.count()):
                        if self.weight_method_combo.itemText(i) == modifier_settings['weight_method']:
                            self.weight_method_combo.setCurrentIndex(i)
                            break
                
                self.log("✅ 模型修改设置加载完成")
        except Exception as e:
            self.log(f"[WARNING] 加载模型修改设置失败: {e}")
    
    def save_settings(self, settings_manager):
        """保存模型修改模块设置"""
        try:
            modifier_settings = {}
            
            if hasattr(self, 'modifier_input_model_edit'):
                modifier_settings['input_model'] = self.modifier_input_model_edit.text()
            if hasattr(self, 'modifier_output_model_edit'):
                modifier_settings['output_model'] = self.modifier_output_model_edit.text()
            if hasattr(self, 'current_channels_label'):
                try:
                    modifier_settings['original_channels'] = int(self.current_channels_label.text())
                except ValueError:
                    pass
            if hasattr(self, 'target_channels_spin'):
                modifier_settings['target_channels'] = self.target_channels_spin.value()
            if hasattr(self, 'weight_method_combo'):
                modifier_settings['weight_method'] = self.weight_method_combo.currentText()
            
            settings_manager.set_section('model_modifier', modifier_settings)
            self.log("✅ 模型修改设置保存完成")
            
        except Exception as e:
            self.log(f"[WARNING] 保存模型修改设置失败: {e}")
        
    def browse_modifier_input_model(self):
        """浏览输入模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "選擇要修改的模型文件", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.modifier_input_model_edit.setText(file_path)
            # 自动生成输出文件名
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_modified{input_path.suffix}"
            self.modifier_output_model_edit.setText(str(output_path))
            # 自动分析模型
            self.analyze_model_for_modification()
            
    def browse_modifier_output_model(self):
        """浏览输出模型文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "保存修改後的模型", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.modifier_output_model_edit.setText(file_path)
            
    def analyze_model_for_modification(self):
        """分析模型以进行修改"""
        input_model = self.modifier_input_model_edit.text()
        if not input_model:
            self.log("[WARNING] 請先選擇輸入模型")
            return
        
        if not Path(input_model).exists():
            self.log("[ERROR] 輸入模型文件不存在")
            QMessageBox.warning(
                self.parent, "警告 Warning",
                "輸入模型文件不存在"
            )
            return
        
        self.log(f"🔍 分析模型: {Path(input_model).name}")
        
        try:
            from Code.Read_Model import get_model_info
            model_info = get_model_info(input_model)
            
            # 提取通道信息
            current_channels = None
            if 'input_channels' in model_info and model_info['input_channels'] is not None:
                try:
                    current_channels = int(model_info['input_channels'])
                except:
                    current_channels = None
            
            # 如果无法获取通道数，尝试从模型架构中推断
            if current_channels is None:
                try:
                    # 尝试从模型架构字符串中提取通道数信息
                    if 'architecture' in model_info and model_info['architecture']:
                        arch_str = str(model_info['architecture'])
                        # 查找常见的通道数模式
                        import re
                        # 查找 Conv2d(3, ...) 或 Conv2d(4, ...) 等模式
                        conv_pattern = r'Conv2d\((\d+),'
                        matches = re.findall(conv_pattern, arch_str)
                        if matches:
                            # 取第一个卷积层的输入通道数
                            current_channels = int(matches[0])
                except:
                    pass
            
            # 如果仍然无法确定，使用默认值3
            if current_channels is None:
                current_channels = 3
                self.log("[WARNING] 無法確定模型通道數，使用默認值3")
            
            self.current_channels_label.setText(str(current_channels))
            self.current_channels_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
            # 显示模型信息
            info_text = "模型分析結果:\n\n"
            for key, value in model_info.items():
                info_text += f"{key}: {value}\n"
            
            self.modifier_model_info_text.setPlainText(info_text)
            self.log(f"[SUCCESS] 模型分析完成 - 當前通道數: {current_channels}")
            
        except Exception as e:
            error_msg = f"[ERROR] 模型分析失敗: {str(e)}"
            self.log(error_msg)
            self.modifier_model_info_text.setPlainText(error_msg)
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"模型分析失敗:\n{str(e)}"
            )
            
    def modify_model_channels(self):
        """修改模型通道数"""
        input_model = self.modifier_input_model_edit.text()
        output_model = self.modifier_output_model_edit.text()
        target_channels = self.target_channels_spin.value()
        
        # 验证输入
        if not input_model:
            self.log("[WARNING] 請選擇輸入模型")
            QMessageBox.warning(
                self.parent, "警告 Warning",
                "請選擇輸入模型文件"
            )
            return
        
        if not output_model:
            self.log("[WARNING] 請指定輸出模型路徑")
            QMessageBox.warning(
                self.parent, "警告 Warning",
                "請指定輸出模型路徑"
            )
            return
        
        if not Path(input_model).exists():
            self.log("[ERROR] 輸入模型文件不存在")
            QMessageBox.warning(
                self.parent, "警告 Warning",
                "輸入模型文件不存在"
            )
            return
        
        # 确认操作
        reply = QMessageBox.question(
            self.parent, "確認 Confirm",
            f"確定要將模型通道數修改為 {target_channels} 嗎？\n\n"
            f"輸入模型: {Path(input_model).name}\n"
            f"輸出模型: {Path(output_model).name}\n\n"
            f"注意：修改後的模型需要重新訓練！",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            self.log("[INFO] 用戶取消操作")
            return
        
        self.log(f"🔧 開始修改模型通道數 -> {target_channels}")
        
        try:
            # 导入模型修改器
            from Code.model_modifier import modify_model_channels
            
            # 执行修改
            modify_model_channels(
                input_model_path=input_model,
                output_model_path=output_model,
                target_channels=target_channels
            )
            
            self.log(f"[SUCCESS] 模型修改完成！")
            self.log(f"   輸出文件: {output_model}")
            
            QMessageBox.information(
                self.parent, "成功 Success",
                f"模型修改完成！\n\n"
                f"輸出文件: {output_model}\n\n"
                f"通道數已修改為: {target_channels}\n"
                f"請重新訓練模型以使用新的通道配置。"
            )
            
        except Exception as e:
            error_msg = f"[ERROR] 模型修改失敗: {str(e)}"
            self.log(error_msg)
            QMessageBox.critical(
                self.parent, "錯誤 Error",
                f"模型修改失敗:\n{str(e)}"
            )
            
    def clear_modifier_fields(self):
        """清空修改器字段"""
        self.modifier_input_model_edit.clear()
        self.modifier_output_model_edit.clear()
        self.modifier_model_info_text.clear()
        self.current_channels_label.setText("未知")
        self.current_channels_label.setStyleSheet("color: #666666; font-weight: bold;")
        self.target_channels_spin.setValue(4)
        self.log("[INFO] 已清空設置")

