"""
GPU工具函数
GPU Utility Functions
处理GPU检测和相关信息
"""

import torch


def get_gpu_name():
    """获取GPU名称"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            return f"{gpu_name} (共 {gpu_count} 個GPU)"
        else:
            return "未檢測到CUDA設備"
    except Exception as e:
        return f"獲取GPU信息失敗: {e}"


def get_device_info():
    """获取设备信息"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return {
                'device': 'cuda',
                'name': device_name,
                'count': device_count,
                'memory_gb': round(device_memory, 2)
            }
        else:
            return {
                'device': 'cpu',
                'name': 'CPU',
                'count': 1,
                'memory_gb': 0
            }
    except Exception as e:
        return {
            'device': 'unknown',
            'name': f'Error: {e}',
            'count': 0,
            'memory_gb': 0
        }
