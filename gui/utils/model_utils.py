"""
模型工具模块
Model Utility Module
提供模型相关的辅助函数
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        包含模型信息的字典
    """
    try:
        # 使用Code目录下的Read_Model模块
        from Code.Read_Model import get_model_info as _get_model_info
        return _get_model_info(model_path)
    except Exception as e:
        return {
            'error': str(e),
            'file': Path(model_path).name,
            'size': f"{Path(model_path).stat().st_size / (1024**2):.2f} MB"
        }


def validate_model(model_path: str,
                  expected_channels: Optional[int] = None) -> Dict[str, Any]:
    """
    验证模型文件
    
    Args:
        model_path: 模型文件路径
        expected_channels: 期望的输入通道数（可选）
        
    Returns:
        验证结果字典
    """
    result = {
        'valid': False,
        'exists': False,
        'readable': False,
        'channels': None,
        'message': ''
    }
    
    path = Path(model_path)
    
    # 检查文件是否存在
    if not path.exists():
        result['message'] = f"文件不存在: {model_path}"
        return result
    
    result['exists'] = True
    
    # 检查文件是否可读
    try:
        with open(path, 'rb') as f:
            f.read(1)
        result['readable'] = True
    except Exception as e:
        result['message'] = f"文件不可读: {e}"
        return result
    
    # 尝试加载模型获取信息
    try:
        if path.suffix == '.pt':
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 尝试获取通道信息
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    if hasattr(model, 'yaml') and 'ch' in model.yaml:
                        result['channels'] = model.yaml['ch']
                
            result['valid'] = True
            result['message'] = "模型验证通过"
            
            # 验证通道数（如果指定）
            if expected_channels is not None and result['channels'] is not None:
                if result['channels'] != expected_channels:
                    result['valid'] = False
                    result['message'] = (
                        f"通道数不匹配: 期望 {expected_channels}, "
                        f"实际 {result['channels']}"
                    )
                    
        elif path.suffix == '.yaml':
            result['valid'] = True
            result['message'] = "YAML配置文件"
            
    except Exception as e:
        result['message'] = f"模型加载失败: {e}"
    
    return result


def get_model_channels(model_path: str) -> int:
    """
    获取模型的输入通道数
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        输入通道数（默认3）
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
                if hasattr(model, 'yaml') and 'ch' in model.yaml:
                    return model.yaml['ch']
    except Exception:
        pass
    
    return 3  # 默认返回3通道


def is_model_compatible(model_path: str,
                       dataset_channels: int) -> bool:
    """
    检查模型是否与数据集兼容
    
    Args:
        model_path: 模型文件路径
        dataset_channels: 数据集通道数
        
    Returns:
        是否兼容
    """
    model_channels = get_model_channels(model_path)
    return model_channels == dataset_channels


def get_model_size_category(file_size_mb: float) -> str:
    """
    根据文件大小判断模型规模
    
    Args:
        file_size_mb: 文件大小（MB）
        
    Returns:
        模型规模分类 (nano, small, medium, large, xlarge)
    """
    if file_size_mb < 10:
        return 'nano'
    elif file_size_mb < 50:
        return 'small'
    elif file_size_mb < 100:
        return 'medium'
    elif file_size_mb < 200:
        return 'large'
    else:
        return 'xlarge'


def format_model_name(model_path: str) -> str:
    """
    格式化模型名称以便显示
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        格式化的显示名称
    """
    path = Path(model_path)
    size_mb = path.stat().st_size / (1024 ** 2)
    size_category = get_model_size_category(size_mb)
    
    return f"{path.stem} ({size_mb:.1f}MB - {size_category})"

