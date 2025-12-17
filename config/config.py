"""
Configuration module for YOLO project
Contains all configuration settings for data processing and training
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

# 數據配置
DATA_CONFIG = {
    'source_path': r'D:\DMD\Forest',
    'output_prefix': r'./dataset_RGB',
    'train_ratio': 0.8,
    'val_ratio': 0.15,
    'test_ratio': 0.05,
    'image_pattern': 'Img0_*',
    'image_width': 640,
    'image_height': 480,
    'aspect_ratio': 4/3,
    'channels': 3,
    'description': 'RGB图像数据，分辨率640x480'
}

# 數據集分割比例（用於 GUI 和數據處理）
TRAIN_SPLIT_RATIO = 0.80
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.05

# 訓練配置
TRAIN_CONFIG = {
    'model_size': 'n',
    'epochs': 20,
    'batch_size': 16,
    'img_size': 640,
    'original_width': 640,
    'original_height': 480,
    'channels': 3,
    'patience': 30,
    'description': '针对640x480 RGB数据优化的训练配置'
}

# GUI 標準訓練默認參數（用於 settings.py）
STANDARD_TRAIN_DEFAULTS = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 10,  # x0.001，實際為 0.01
    'image_size': 640,
    'training_mode': 'pretrained'
}

# GUI 推理默認參數（用於 settings.py）
INFERENCE_DEFAULTS = {
    'confidence': 0.25,
    'iou_threshold': 0.45,
    'max_det': 300,
    'mode': 'single'
}

# 預期數據配置
EXPECTED_DATA = {
    'videos': 10,
    'frames_per_video': 600,
    'total_images': 6000,
    'description': '预期处理约6000张RGB图像'
}

# 預定義類別配置
PREDEFINED_CLASSES = {
    'file': 'predefined_classes.txt',
    'classes': ['drone', 'fixed wing', 'tree', 'ground'],
    'description': '使用predefined_classes.txt中定义的类别'
}

# RAFT-Stereo 立體視覺訓練配置
STEREO_PRETRAINED_DIR = 'Model_file/Stereo_Vision'
RAFT_STEREO_CONFIG = {
    'name': 'raft-stereo',
    'batch_size': 2,
    'lr': 0.0002,
    'num_steps': 200000,
    'image_size': (320, 720),
    'train_iters': 16,
    'valid_iters': 32,
    'wdecay': 0.00001,
    'corr_implementation': 'reg',
    'corr_levels': 4,
    'corr_radius': 4,
    'n_downsample': 2,
    'context_norm': 'batch',
    'n_gru_layers': 3,
    'hidden_dims': [128, 128, 128],
    'spatial_scale': (0.0, 0.0),
    'noyjitter': False,
    'mixed_precision': False,
    'shared_backbone': False,
    'slow_fast_gru': False,
    'pretrained_dir': STEREO_PRETRAINED_DIR,
    'description': 'RAFT-Stereo 立體視覺訓練默認配置'
}

# GUI 立體視覺訓練默認參數（基於 RAFT_STEREO_CONFIG 自動生成，避免重複定義）
# 從 RAFT_STEREO_CONFIG 轉換為 GUI 格式
def _build_stereo_gui_defaults():
    """基於 RAFT_STEREO_CONFIG 構建 GUI 默認參數"""
    # 轉換 image_size 從元組到字符串格式
    img_size = RAFT_STEREO_CONFIG.get('image_size', (320, 720))
    img_size_str = f"{img_size[0]}x{img_size[1]}" if isinstance(img_size, (tuple, list)) else str(img_size)
    
    # 轉換 hidden_dims 從列表到字符串格式
    hidden_dims = RAFT_STEREO_CONFIG.get('hidden_dims', [128, 128, 128])
    hidden_dims_str = f"{hidden_dims[0]}x{hidden_dims[1]}x{hidden_dims[2]} (默認)" if isinstance(hidden_dims, list) else str(hidden_dims)
    
    return {
        # 基本參數：直接從 RAFT_STEREO_CONFIG 獲取
        'batch_size': RAFT_STEREO_CONFIG.get('batch_size', 6),
        'num_steps': RAFT_STEREO_CONFIG.get('num_steps', 100000),
        # GUI 特有參數
        'model_name': 'raftstereo-sceneflow.pth',
        'output_path': '',
        # 高級參數：從 RAFT_STEREO_CONFIG 轉換
        'advanced_params': {
            'corr_implementation': RAFT_STEREO_CONFIG.get('corr_implementation', 'reg'),
            'n_downsample': RAFT_STEREO_CONFIG.get('n_downsample', 2),
            'corr_levels': RAFT_STEREO_CONFIG.get('corr_levels', 4),
            'corr_radius': RAFT_STEREO_CONFIG.get('corr_radius', 4),
            'n_gru_layers': RAFT_STEREO_CONFIG.get('n_gru_layers', 3),
            'shared_backbone': RAFT_STEREO_CONFIG.get('shared_backbone', False),
            'context_norm': RAFT_STEREO_CONFIG.get('context_norm', 'batch'),
            'slow_fast_gru': RAFT_STEREO_CONFIG.get('slow_fast_gru', False),
            'hidden_dims': hidden_dims_str,
            'mixed_precision': RAFT_STEREO_CONFIG.get('mixed_precision', False),
            'weight_decay': RAFT_STEREO_CONFIG.get('wdecay', 0.00001),
            'train_iters': RAFT_STEREO_CONFIG.get('train_iters', 16),
            'valid_iters': RAFT_STEREO_CONFIG.get('valid_iters', 32),
            'learning_rate': RAFT_STEREO_CONFIG.get('lr', 0.0002),
            'image_size': img_size_str,
            # 增廣參數默認值（RAFT_STEREO_CONFIG 中沒有，使用 GUI 默認值）
            'spatial_scale_min': -0.2,
            'spatial_scale_max': 0.4,
            'saturation_min': 0.0,
            'saturation_max': 1.4,
            'gamma_min': 0.8,
            'gamma_max': 1.2,
            'do_flip': '無 None',
            'noyjitter': RAFT_STEREO_CONFIG.get('noyjitter', False)
        }
    }

STEREO_TRAIN_GUI_DEFAULTS = _build_stereo_gui_defaults()


@dataclass
class TrainingConfig:
    """RAFT-Stereo 訓練配置類"""
    
    # 基本參數
    name: str = 'raft-stereo'
    restore_ckpt: Optional[str] = None
    mixed_precision: bool = False
    
    # 訓練參數
    batch_size: int = 6
    train_datasets: List[str] = None
    lr: float = 0.0002
    num_steps: int = 100000
    image_size: Tuple[int, int] = (320, 720)
    train_iters: int = 16
    wdecay: float = 0.00001
    valid_iters: int = 32
    
    # 模型架構參數
    corr_implementation: str = "reg"
    shared_backbone: bool = False
    corr_levels: int = 4
    corr_radius: int = 4
    n_downsample: int = 2
    context_norm: str = "batch"
    slow_fast_gru: bool = False
    n_gru_layers: int = 3
    hidden_dims: List[int] = None
    
    # 增廣參數
    img_gamma: Optional[List[float]] = None
    saturation_range: Optional[List[float]] = None
    do_flip: Optional[str] = None
    spatial_scale: Tuple[float, float] = (0.0, 0.0)
    noyjitter: bool = False
    
    # 路徑參數
    output_dir: str = '.'
    dataset_root: str = 'Dataset'
    
    def __post_init__(self):
        """初始化後處理"""
        if self.train_datasets is None:
            self.train_datasets = ['drone']
        
        if self.hidden_dims is None:
            self.hidden_dims = [128] * 3
        
        # 確保 image_size 是元組格式
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)
        
        # 確保 spatial_scale 是元組格式
        if isinstance(self.spatial_scale, list):
            self.spatial_scale = tuple(self.spatial_scale)
    
    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            'name': self.name,
            'restore_ckpt': self.restore_ckpt,
            'mixed_precision': self.mixed_precision,
            'batch_size': self.batch_size,
            'train_datasets': self.train_datasets,
            'lr': self.lr,
            'num_steps': self.num_steps,
            'image_size': self.image_size,
            'train_iters': self.train_iters,
            'wdecay': self.wdecay,
            'valid_iters': self.valid_iters,
            'corr_implementation': self.corr_implementation,
            'shared_backbone': self.shared_backbone,
            'corr_levels': self.corr_levels,
            'corr_radius': self.corr_radius,
            'n_downsample': self.n_downsample,
            'context_norm': self.context_norm,
            'slow_fast_gru': self.slow_fast_gru,
            'n_gru_layers': self.n_gru_layers,
            'hidden_dims': self.hidden_dims,
            'img_gamma': self.img_gamma,
            'saturation_range': self.saturation_range,
            'do_flip': self.do_flip,
            'spatial_scale': self.spatial_scale,
            'noyjitter': self.noyjitter,
            'output_dir': self.output_dir,
            'dataset_root': self.dataset_root
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """從字典創建配置對象"""
        return cls(**data)
    
    @classmethod
    def from_default_config(cls, **overrides) -> 'TrainingConfig':
        """從默認配置創建配置對象"""
        config_data = RAFT_STEREO_CONFIG.copy()
        config_data.update(overrides)
        return cls(**config_data)
    
    def update(self, **kwargs):
        """更新配置參數"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """驗證配置參數的有效性"""
        try:
            # 檢查必要的參數
            assert self.batch_size > 0, "batch_size must be positive"
            assert self.lr > 0, "learning rate must be positive"
            assert self.num_steps > 0, "num_steps must be positive"
            assert len(self.image_size) == 2, "image_size must be (width, height)"
            assert all(s > 0 for s in self.image_size), "image_size dimensions must be positive"
            assert self.train_iters > 0, "train_iters must be positive"
            assert self.valid_iters > 0, "valid_iters must be positive"
            assert self.corr_levels > 0, "corr_levels must be positive"
            assert self.corr_radius > 0, "corr_radius must be positive"
            assert self.n_downsample >= 0, "n_downsample must be non-negative"
            assert self.n_gru_layers > 0, "n_gru_layers must be positive"
            assert len(self.hidden_dims) == self.n_gru_layers, "hidden_dims length must match n_gru_layers"
            
            # 檢查路徑
            assert Path(self.output_dir).parent.exists(), f"output_dir parent directory does not exist: {self.output_dir}"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
