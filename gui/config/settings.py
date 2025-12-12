"""
设置管理模块
Settings Manager Module
处理应用配置的保存和加载
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class SettingsManager:
    """应用设置管理器"""
    
    def __init__(self, config_file: str = "config/gui_settings.yaml"):
        """
        初始化设置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file)
        self.settings = {}
        self._ensure_config_dir()
        self.load()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> Dict[str, Any]:
        """
        加载配置
        
        Returns:
            配置字典
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.settings = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"加载配置失败: {e}")
                self.settings = {}
        else:
            self.settings = self._get_default_settings()
            self.save()
        
        return self.settings
    
    def save(self) -> bool:
        """
        保存配置
        
        Returns:
            是否保存成功
        """
        try:
            # 添加保存时间戳
            self.settings['last_saved'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.settings, f, allow_unicode=True, default_flow_style=False)
            
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键（支持点号分隔的嵌套键，如 'training.epochs'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键（支持点号分隔的嵌套键）
            value: 配置值
        """
        keys = key.split('.')
        settings = self.settings
        
        # 导航到最后一级
        for k in keys[:-1]:
            if k not in settings:
                settings[k] = {}
            settings = settings[k]
        
        # 设置值
        settings[keys[-1]] = value
    
    def get_section(self, section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 节名称
            default: 默认值（如果节不存在）
            
        Returns:
            配置节字典
        """
        if default is None:
            default = {}
        return self.settings.get(section, default)
    
    def set_section(self, section: str, data: Dict[str, Any]) -> None:
        """
        设置配置节
        
        Args:
            section: 节名称
            data: 配置数据
        """
        self.settings[section] = data
    
    def reset(self) -> None:
        """重置为默认设置"""
        self.settings = self._get_default_settings()
        self.save()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        获取默认设置
        
        从 config.config 引用默认值，避免重复定义
        
        Returns:
            默认设置字典
        """
        # 尝试从 config.config 导入默认配置
        try:
            import sys
            from pathlib import Path
            
            # 添加项目根目录到 Python 路径
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from config.config import (
                STANDARD_TRAIN_DEFAULTS,
                INFERENCE_DEFAULTS,
                STEREO_TRAIN_GUI_DEFAULTS,
                DATA_CONFIG
            )
            
            # 从 config.py 引用标准训练默认值
            training_defaults = STANDARD_TRAIN_DEFAULTS.copy()
            training_defaults.update({
                'last_dataset': '',
                'last_model': ''
            })
            
            # 从 config.py 引用推理默认值
            inference_defaults = INFERENCE_DEFAULTS.copy()
            inference_defaults.update({
                'last_model': ''
            })
            
            # 从 config.py 引用立体视觉训练默认值
            stereo_defaults = STEREO_TRAIN_GUI_DEFAULTS.copy()
            stereo_defaults.update({
                'dataset_path': ''
            })
            
            # 从 config.py 引用数据配置
            data_source_path = DATA_CONFIG.get('source_path', 'D:\\DMD\\Forest')
            
        except ImportError as e:
            # 如果无法导入，使用本地默认值（向后兼容）
            print(f"[WARNING] 无法从 config.config 导入默认配置，使用本地默认值: {e}")
            training_defaults = {
                'last_dataset': '',
                'last_model': '',
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 10,
                'image_size': 640,
                'training_mode': 'pretrained'
            }
            inference_defaults = {
                'last_model': '',
                'confidence': 0.25,
                'iou_threshold': 0.45,
                'max_det': 300,
                'mode': 'single'
            }
            stereo_defaults = {
                'dataset_path': '',
                'model_name': 'raftstereo-sceneflow.pth',
                'epochs': 100,
                'batch_size': 6,
                'output_path': '',
                'advanced_params': {
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
                    'image_size': '320,720',
                    'spatial_scale_min': -0.2,
                    'spatial_scale_max': 0.4,
                    'saturation_min': 0.0,
                    'saturation_max': 1.4,
                    'gamma_min': 0.8,
                    'gamma_max': 1.2,
                    'do_flip': 'none',
                    'noyjitter': False
                }
            }
            data_source_path = 'D:\\DMD\\Forest'
        
        return {
            'version': '2.0.0',
            'window': {
                'width': 1400,
                'height': 900,
                'x': 100,
                'y': 100
            },
            'data_conversion': {
                'source_path': data_source_path,
                'output_path': '',
                'use_depth': True,
                'use_stereo': False,
                'folder_count': 1
            },
            'training': training_defaults,
            'inference': inference_defaults,
            'model_analyzer': {
                'last_folder': '',
                'file_type_filter': '所有類型'
            },
            'model_modifier': {
                'target_channels': 4
            },
            'stereo_training': stereo_defaults,
            'ui': {
                'theme': 'default',
                'font_size': 10,
                'language': 'zh_TW'
            }
        }
    
    def export_to_file(self, file_path: str) -> bool:
        """
        导出配置到文件
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.settings, f, allow_unicode=True, default_flow_style=False)
            
            return True
        except Exception as e:
            print(f"导出配置失败: {e}")
            return False
    
    def import_from_file(self, file_path: str) -> bool:
        """
        从文件导入配置
        
        Args:
            file_path: 导入文件路径
            
        Returns:
            是否导入成功
        """
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_settings = yaml.safe_load(f)
            
            if imported_settings:
                self.settings = imported_settings
                self.save()
                return True
            
            return False
        except Exception as e:
            print(f"导入配置失败: {e}")
            return False

