"""
常量定义模块
Constants Module
定义应用中使用的常量
"""

# 应用信息
APP_NAME = "YOLO统一启动器"
APP_VERSION = "2.0.0"
APP_AUTHOR = "YOLO Team"

# 文件类型
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
SUPPORTED_MODEL_FORMATS = ['.pt', '.pth']
SUPPORTED_CONFIG_FORMATS = ['.yaml', '.yml']

# 默认路径
DEFAULT_DATASET_DIR = "Dataset"
DEFAULT_MODEL_DIR = "Model_file"
DEFAULT_PT_DIR = "Model_file/PT_File"
DEFAULT_STEREO_DIR = "Model_file/Stereo_Vision"
DEFAULT_YAML_DIR = "Model_file/YAML"
DEFAULT_OUTPUT_DIR = "runs"
DEFAULT_PREDICT_DIR = "Predict"


# UI相关常量
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800
LOG_MAX_LINES = 10000

# 颜色定义
COLOR_SUCCESS = "#28a745"
COLOR_ERROR = "#dc3545"
COLOR_WARNING = "#ffc107"
COLOR_INFO = "#17a2b8"
COLOR_PRIMARY = "#007bff"

# 日志级别
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# 模型类型映射
MODEL_TYPES = {
    'yolo11': ['n', 's', 'm', 'l', 'x'],
    'yolo12': ['n', 's', 'm', 'l', 'x'],
    'yolo13': ['n', 's', 'm', 'l', 'x'],
    'yolov8': ['n', 's', 'm', 'l', 'x']
}

# 优化器选项
OPTIMIZER_OPTIONS = ['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp']

# 自动增强策略
AUTO_AUGMENT_STRATEGIES = ['randaugment', 'autoaugment', 'augmix', 'None']

# RAFT-Stereo预训练模型
STEREO_PRETRAINED_MODELS = [
    'raftstereo-sceneflow.pth',
    'raftstereo-middlebury.pth',
    'raftstereo-eth3d.pth',
    'iraftstereo_rvc.pth',
    'raftstereo-realtime.pth'
]

# 文件大小单位
FILE_SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

# 消息类型
MESSAGE_TYPES = {
    'INFO': '信息',
    'WARNING': '警告',
    'ERROR': '错误',
    'SUCCESS': '成功'
}

