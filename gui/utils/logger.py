"""
日志工具模块
Logger Utility Module
提供统一的日志记录功能
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(name: str = 'YOLO_GUI', 
                log_file: Optional[str] = None,
                level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_message(message: str, 
               level: str = 'INFO',
               logger_name: str = 'YOLO_GUI') -> str:
    """
    记录日志消息并返回格式化的消息
    
    Args:
        message: 日志消息
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: 日志记录器名称
        
    Returns:
        格式化的日志消息
    """
    logger = logging.getLogger(logger_name)
    
    # 根据级别记录日志
    level_map = {
        'DEBUG': logger.debug,
        'INFO': logger.info,
        'WARNING': logger.warning,
        'ERROR': logger.error,
        'CRITICAL': logger.critical
    }
    
    log_func = level_map.get(level.upper(), logger.info)
    log_func(message)
    
    # 返回格式化的消息（用于GUI显示）
    timestamp = datetime.now().strftime('%H:%M:%S')
    return f"[{timestamp}] {message}"


class LoggerMixin:
    """
    日志混入类
    为类提供日志功能
    """
    
    def __init__(self, logger_name: str = 'YOLO_GUI'):
        self._logger = logging.getLogger(logger_name)
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self._logger.debug(message)
    
    def log_info(self, message: str):
        """记录一般信息"""
        self._logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告信息"""
        self._logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误信息"""
        self._logger.error(message)
    
    def log_critical(self, message: str):
        """记录严重错误信息"""
        self._logger.critical(message)

