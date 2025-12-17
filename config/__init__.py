"""
Config package for YOLO project
"""

# 導入主要配置
from .config import (
    DATA_CONFIG,
    TRAIN_CONFIG, 
    EXPECTED_DATA,
    PREDEFINED_CLASSES,
    RAFT_STEREO_CONFIG,
    STANDARD_TRAIN_DEFAULTS,
    INFERENCE_DEFAULTS,
    STEREO_TRAIN_GUI_DEFAULTS,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
    TrainingConfig
)

# 導入預定義類別
try:
    from .predefined_classes import load_predefined_classes
except ImportError:
    def load_predefined_classes():
        return ['drone', 'fixed wing', 'tree', 'ground']

__all__ = [
    'DATA_CONFIG',
    'TRAIN_CONFIG', 
    'EXPECTED_DATA',
    'PREDEFINED_CLASSES',
    'RAFT_STEREO_CONFIG',
    'STANDARD_TRAIN_DEFAULTS',
    'INFERENCE_DEFAULTS',
    'STEREO_TRAIN_GUI_DEFAULTS',
    'TRAIN_SPLIT_RATIO',
    'VAL_SPLIT_RATIO',
    'TEST_SPLIT_RATIO',
    'TrainingConfig',
    'load_predefined_classes'
]
