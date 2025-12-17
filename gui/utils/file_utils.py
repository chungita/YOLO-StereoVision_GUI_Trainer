"""
文件工具模块
File Utility Module
提供文件和目录操作的辅助函数
"""

import os
from pathlib import Path
from typing import List, Optional, Union


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        Path对象
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: Union[str, Path], 
                 unit: str = 'MB') -> float:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        unit: 单位 ('B', 'KB', 'MB', 'GB')
        
    Returns:
        文件大小（指定单位）
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0
    
    size_bytes = path.stat().st_size
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    divisor = units.get(unit.upper(), 1024 ** 2)
    return size_bytes / divisor


def find_files(directory: Union[str, Path],
              pattern: str = '*',
              recursive: bool = True,
              max_depth: Optional[int] = None) -> List[Path]:
    """
    查找文件
    
    Args:
        directory: 搜索目录
        pattern: 文件匹配模式（如 '*.pt', '*.yaml'）
        recursive: 是否递归搜索
        max_depth: 最大搜索深度（None表示无限制）
        
    Returns:
        找到的文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    if recursive:
        if max_depth is None:
            return list(dir_path.rglob(pattern))
        else:
            # 限制搜索深度
            files = []
            for depth in range(max_depth + 1):
                pattern_with_depth = '/'.join(['*'] * depth) + f'/{pattern}'
                files.extend(dir_path.glob(pattern_with_depth))
            return files
    else:
        return list(dir_path.glob(pattern))


def get_relative_path(file_path: Union[str, Path],
                     base_path: Union[str, Path]) -> str:
    """
    获取相对路径
    
    Args:
        file_path: 文件路径
        base_path: 基础路径
        
    Returns:
        相对路径字符串
    """
    try:
        return str(Path(file_path).relative_to(Path(base_path)))
    except ValueError:
        return str(file_path)


def safe_filename(filename: str) -> str:
    """
    生成安全的文件名（移除非法字符）
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 替换Windows不允许的字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename


def get_unique_path(base_path: Union[str, Path],
                   suffix: str = '') -> Path:
    """
    获取唯一的文件路径（如果存在则添加序号）
    
    Args:
        base_path: 基础路径
        suffix: 文件后缀（可选）
        
    Returns:
        唯一的路径
    """
    path = Path(base_path)
    
    if suffix:
        path = path.with_suffix(suffix)
    
    if not path.exists():
        return path
    
    # 添加序号
    counter = 1
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def count_files_in_dir(directory: Union[str, Path],
                      pattern: str = '*',
                      recursive: bool = True) -> int:
    """
    统计目录中的文件数量
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归统计
        
    Returns:
        文件数量
    """
    return len(find_files(directory, pattern, recursive))


def dir_size(directory: Union[str, Path],
            unit: str = 'MB') -> float:
    """
    计算目录总大小
    
    Args:
        directory: 目录路径
        unit: 单位 ('B', 'KB', 'MB', 'GB')
        
    Returns:
        目录总大小（指定单位）
    """
    total_size = 0
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return 0.0
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    divisor = units.get(unit.upper(), 1024 ** 2)
    return total_size / divisor

