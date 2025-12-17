"""
訓練核心模組
Training Core Module
處理所有訓練相關的核心邏輯，完全獨立於GUI
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# 添加Code目录到Python路径
code_dir = Path(__file__).parent.parent.parent / "Code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from Code.YOLO_standard_trainer import YOLOStandardTrainer


class TrainingCore:
    """訓練核心類 - 完全獨立的訓練邏輯"""
    
    def __init__(self):
        self.current_trainer = None
        self._stop_requested = False
        
    def train_with_pretrained(self, 
                            config_path: str,
                            model_path: str,
                            epochs: int = 50,
                            learning_rate: float = 0.01,
                            batch_size: int = 16,
                            imgsz: int = 640,
                            save_period: int = -1,
                            # 数据增强参数
                            scale: float = 0.5,
                            mosaic: float = 1.0,
                            mixup: float = 0.0,
                            copy_paste: float = 0.0,
                            hsv_h: float = 0.015,
                            hsv_s: float = 0.7,
                            hsv_v: float = 0.4,
                            bgr: float = 0.0,
                            auto_augment: Optional[str] = None,
                            # 几何变换参数
                            degrees: float = 0.0,
                            translate: float = 0.1,
                            shear: float = 0.0,
                            perspective: float = 0.0,
                            # 翻转和裁剪参数
                            flipud: float = 0.0,
                            fliplr: float = 0.5,
                            erasing: float = 0.0,
                            crop_fraction: float = 1.0,
                            # 优化器参数
                            weight_decay: float = 0.0005,
                            momentum: float = 0.937,
                            beta1: float = 0.9,
                            beta2: float = 0.999,
                            # 学习率调度参数
                            lr_scheduler: str = 'auto',
                            lr_decay: float = 0.1,
                            warmup_epochs: int = 3,
                            warmup_momentum: float = 0.8,
                            # 验证参数
                            val_frequency: int = 1,
                            val_iters: int = 32,
                            early_stopping_patience: int = 50,
                            early_stopping_min_delta: float = 0.001,
                            # 设备参数
                            device: str = 'auto',
                            multi_gpu: bool = False,
                            gpu_memory_optimization: bool = True,
                            data_loading_optimization: bool = True,
                            # 其他高级参数
                            close_mosaic: int = 10,
                            single_cls: bool = False,
                            cache: bool = False,
                            resume: bool = False,
                            workers: int = 8,
                            optimizer: str = 'auto',
                            amp: bool = True,
                            progress_callback: Optional[Callable] = None,
                            log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        使用預訓練模型進行訓練
        
        Args:
            config_path: 配置文件路徑
            model_path: 模型文件路徑
            epochs: 訓練輪數
            learning_rate: 學習率
            batch_size: 批次大小
            imgsz: 圖像大小
            save_period: 保存週期
            # 数据增强参数
            scale: 縮放比例
            mosaic: Mosaic數據增強
            mixup: Mixup數據增強
            copy_paste: Copy-paste數據增強
            hsv_h: HSV色相增強參數
            hsv_s: HSV飽和度增強參數
            hsv_v: HSV明度增強參數
            bgr: BGR通道增強參數
            auto_augment: 自動增強策略
            # 几何变换参数
            degrees: 旋轉角度
            translate: 平移距離
            shear: 剪切角度
            perspective: 透視變換
            # 翻转和裁剪参数
            flipud: 上下翻轉概率
            fliplr: 左右翻轉概率
            erasing: 隨機擦除概率
            crop_fraction: 裁剪比例
            # 优化器参数
            weight_decay: 權重衰減
            momentum: 動量
            beta1: Adam優化器β1參數
            beta2: Adam優化器β2參數
            # 学习率调度参数
            lr_scheduler: 學習率調度器
            lr_decay: 學習率衰減
            warmup_epochs: 預熱輪數
            warmup_momentum: 預熱動量
            # 验证参数
            val_frequency: 驗證頻率
            val_iters: 驗證迭代次數
            early_stopping_patience: 早停耐心值
            early_stopping_min_delta: 早停最小改善
            # 设备参数
            device: 設備選擇
            multi_gpu: 多GPU訓練
            gpu_memory_optimization: GPU內存優化
            data_loading_optimization: 數據加載優化
            # 其他高级参数
            close_mosaic: 關閉Mosaic的epoch數
            single_cls: 單類別訓練
            cache: 數據緩存
            resume: 恢復訓練
            workers: 工作進程數
            optimizer: 優化器類型
            amp: 是否使用混合精度
            progress_callback: 進度回調函數
            log_callback: 日誌回調函數
            
        Returns:
            訓練結果字典
        """
        try:
            if log_callback:
                log_callback("📦 調用標準訓練器模組 Calling standard trainer module...")
                log_callback(f"📋 預訓練模式 Pretrained mode - PT: {model_path}")
            
            # 創建訓練器
            self.current_trainer = YOLOStandardTrainer(
                config_path=config_path,
                model_path=model_path,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                imgsz=imgsz,
                save_period=save_period,
                # 数据增强参数
                scale=scale,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                bgr=bgr,
                auto_augment=auto_augment,
                # 几何变换参数
                degrees=degrees,
                translate=translate,
                shear=shear,
                perspective=perspective,
                # 翻转和裁剪参数
                flipud=flipud,
                fliplr=fliplr,
                erasing=erasing,
                crop_fraction=crop_fraction,
                # 优化器参数
                weight_decay=weight_decay,
                momentum=momentum,
                beta1=beta1,
                beta2=beta2,
                # 学习率调度参数
                lr_scheduler=lr_scheduler,
                lr_decay=lr_decay,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                # 验证参数
                val_frequency=val_frequency,
                val_iters=val_iters,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                # 设备参数
                device=device,
                multi_gpu=multi_gpu,
                gpu_memory_optimization=gpu_memory_optimization,
                data_loading_optimization=data_loading_optimization,
                # 其他高级参数
                close_mosaic=close_mosaic,
                single_cls=single_cls,
                cache=cache,
                resume=resume,
                workers=workers,
                optimizer=optimizer,
                amp=amp
            )
            
            if log_callback:
                log_callback("🚀 開始訓練（預訓練模式）... Starting training (Pretrained mode)...")
                log_callback(f"   輪數 Epochs: {epochs}, 批次 Batch: {batch_size}, 學習率 LR: {learning_rate}")
            
            # 執行訓練
            results = self.current_trainer.train(
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            return {
                'success': True,
                'results': results,
                'message': '訓練完成 Training completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'訓練失敗 Training failed: {str(e)}'
            }
    
    def train_with_yaml(self,
                       config_path: str,
                       yaml_path: str,
                       model_size: str = 'n',
                       epochs: int = 50,
                       learning_rate: float = 0.01,
                       batch_size: int = 16,
                       imgsz: int = 640,
                       save_period: int = -1,
                       scale: float = 0.5,
                       mosaic: float = 1.0,
                       mixup: float = 0.0,
                       copy_paste: float = 0.0,
                       hsv_h: float = 0.015,
                       hsv_s: float = 0.7,
                       hsv_v: float = 0.4,
                       bgr: float = 0.0,
                       auto_augment: Optional[str] = None,
                       degrees: float = 0.0,
                       translate: float = 0.1,
                       shear: float = 0.0,
                       perspective: float = 0.0,
                       flipud: float = 0.0,
                       fliplr: float = 0.5,
                       erasing: float = 0.0,
                       crop_fraction: float = 1.0,
                       close_mosaic: int = 10,
                       workers: int = 8,
                       optimizer: str = 'auto',
                       amp: bool = True,
                       progress_callback: Optional[Callable] = None,
                       log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        使用YAML配置從頭訓練
        
        Args:
            config_path: 配置文件路徑
            yaml_path: YAML配置文件路徑
            model_size: 模型大小 (n, s, m, l, x)
            epochs: 訓練輪數
            learning_rate: 學習率
            batch_size: 批次大小
            imgsz: 圖像大小
            save_period: 保存週期
            scale: 縮放比例
            mosaic: Mosaic數據增強
            mixup: Mixup數據增強
            copy_paste: Copy-paste數據增強
            hsv_h: HSV色相增強參數
            hsv_s: HSV飽和度增強參數
            hsv_v: HSV明度增強參數
            bgr: BGR通道增強參數
            auto_augment: 自動增強策略
            degrees: 旋轉角度
            translate: 平移距離
            shear: 剪切角度
            perspective: 透視變換
            flipud: 上下翻轉概率
            fliplr: 左右翻轉概率
            erasing: 隨機擦除概率
            crop_fraction: 裁剪比例
            close_mosaic: 關閉Mosaic的epoch數
            workers: 工作進程數
            optimizer: 優化器類型
            amp: 是否使用混合精度
            progress_callback: 進度回調函數
            log_callback: 日誌回調函數
            
        Returns:
            訓練結果字典
        """
        try:
            import warnings
            warnings.filterwarnings('ignore')
            from ultralytics import YOLO
            
            if log_callback:
                log_callback(f"📋 重新訓練模式 Retrain mode - YAML: {yaml_path}")
                log_callback(f"📋 模型大小 Model size: {model_size}")
            
            # 構建帶有模型大小的YAML路徑
            base_name = Path(yaml_path).stem
            sized_yaml = f"{base_name}{model_size}.yaml"
            
            # 檢查帶有模型大小的YAML文件是否存在
            sized_yaml_path = Path(sized_yaml)
            if sized_yaml_path.exists():
                if log_callback:
                    log_callback(f"📋 使用 Using: {sized_yaml}")
                model = YOLO(model=sized_yaml)
            else:
                if log_callback:
                    log_callback(f"📋 使用基礎文件 Using base file: {yaml_path}")
                model = YOLO(model=yaml_path)
            
            # 存儲訓練器引用以支持停止功能
            self.current_trainer = model
            
            if log_callback:
                log_callback("🚀 開始訓練（YAML模式）... Starting training (YAML mode)...")
                log_callback(f"   輪數 Epochs: {epochs}, 批次 Batch: {batch_size}, 學習率 LR: {learning_rate}")
            
            # 直接使用 ultralytics 訓練
            results = model.train(
                data=config_path,
                imgsz=imgsz,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                amp=amp,
                workers=workers,
                device='',
                optimizer=optimizer,
                close_mosaic=close_mosaic,
                resume=False,
                project='runs',
                name=self._generate_custom_model_name(config_path, yaml_path, model_size, epochs, 'retrain'),
                single_cls=False,
                cache=False,
                save_period=save_period,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                bgr=bgr,
                auto_augment=auto_augment,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                erasing=erasing,
                crop_fraction=crop_fraction
            )
            
            return {
                'success': True,
                'results': results,
                'message': '訓練完成 Training completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'訓練失敗 Training failed: {str(e)}'
            }
    
    def stop_training(self):
        """停止訓練"""
        self._stop_requested = True
        
        if self.current_trainer:
            try:
                if hasattr(self.current_trainer, 'request_stop'):
                    self.current_trainer.request_stop()
            except Exception:
                pass  # 靜默處理
    
    def _generate_custom_model_name(self, config_path: str, model_file: str, model_size: str, epochs: int, training_mode: str) -> str:
        """生成自定義模型名稱
        
        格式: {model_name}_{channel_type}_{epochs}epochs_{timestamp}
        例如: yolo12n_RGB_50epochs_20251212_1430
        """
        try:
            import yaml
            import re
            with open(config_path, 'r', encoding='utf-8') as f:
                dataset_config = yaml.safe_load(f)
            
            channels = dataset_config.get('channels', 3)
            channel_type = 'RGBD' if channels == 4 else 'RGB'
            
            # 獲取模型名稱
            if training_mode == 'retrain':
                model_name = Path(model_file).stem
                full_model_name = f"{model_name}{model_size}"  # 例如: yolo12n
            else:
                model_name = Path(model_file).stem
                full_model_name = model_name
            
            # 移除模型名稱中已有的通道類型後綴 (避免重複)
            # 例如: yolo12n_RGBD -> yolo12n, yolo12n_RGB -> yolo12n
            full_model_name = re.sub(r'_(RGBD|RGB|4ch|3ch)$', '', full_model_name, flags=re.IGNORECASE)
            
            # 生成時間戳 (格式: YYYYMMDD_HHMM)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # 生成基礎模型名稱 (格式: {模型名}_{通道類型}_{輪數}epochs_{時間戳})
            base_custom_name = f"{full_model_name}_{channel_type}_{epochs}epochs_{timestamp}"
            
            # 檢查文件夾是否已存在，如果存在則添加序號
            custom_name = self._get_unique_training_folder_name(base_custom_name)
            
            return custom_name
            
        except Exception as e:
            return 'exp'
    
    def _get_unique_training_folder_name(self, base_name: str) -> str:
        """生成唯一的訓練文件夾名稱，如果重複則添加序號"""
        # 檢查 runs/train 目錄是否存在
        runs_train_dir = Path('runs/train')
        if not runs_train_dir.exists():
            return base_name
        
        # 檢查基礎名稱是否已存在
        if not (runs_train_dir / base_name).exists():
            return base_name
        
        # 如果存在，添加序號
        counter = 1
        while True:
            unique_name = f"{base_name}({counter})"
            if not (runs_train_dir / unique_name).exists():
                return unique_name
            counter += 1


# 全局訓練核心實例
training_core = TrainingCore()
