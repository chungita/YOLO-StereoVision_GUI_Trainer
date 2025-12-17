"""
通用工作线程
General Worker Thread
处理各种后台任务的工作线程类，包括数据转换、推理和训练
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

# 导入数据转换模块
from Code.data_converter import RGBPreprocessor, StereoPreprocessor

# 添加Code目录到Python路径
code_dir = Path(__file__).parent.parent.parent / "Code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

# 导入训练核心模块
try:
    from gui.modules.training_engine import training_core
except ImportError:
    training_core = None


class WorkerThread(QThread):
    """通用工作线程类 - General Worker Thread Class
    支持数据转换、推理和训练任务
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int, str)  # current, total, text
    
    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.mutex = QMutex()
        self._stop_requested = False
        self._current_trainer = None
        
    def run(self):
        try:
            if self._stop_requested:
                return
                
            if self.task_type == "convert":
                self._convert_data()
            elif self.task_type == "inference":
                self._inference()
            elif self.task_type == "stereo_inference":
                self._stereo_inference()
            elif self.task_type == "train_pretrained":
                self._train_with_pretrained()
            elif self.task_type == "train_yaml":
                self._train_with_yaml()
            elif self.task_type == "train_stereo":
                self._train_stereo()
            else:
                raise ValueError(f"未知的任務類型 Unknown task type: {self.task_type}")
            
            if not self._stop_requested:
                self.finished.emit(True, "任務完成 Task completed")
        except Exception as e:
            if not self._stop_requested:
                self.finished.emit(False, str(e))
    
    def stop(self):
        """安全停止线程 - Stop thread safely"""
        self._stop_requested = True
        
        # 如果正在训练，请求训练器停止
        if hasattr(self, '_current_trainer') and self._current_trainer:
            try:
                if hasattr(self._current_trainer, 'stop'):
                    self._current_trainer.stop()
                elif hasattr(self._current_trainer, 'request_stop'):
                    self._current_trainer.request_stop()
            except Exception:
                pass  # 静默处理
        
        # 停止训练核心
        if training_core:
            try:
                training_core.stop_training()
            except Exception:
                pass
        
        # 释放PyTorch和CUDA资源
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            # 静默处理CUDA资源释放错误
            pass
        
        self.quit()
        self.wait(3000)  # 等待3秒
        if self.isRunning():
            self.terminate()
            self.wait(1000)  # 再等待1秒
    
    def _convert_data(self):
        """数据转换 - Data Conversion"""
        try:
            self.progress.emit("正在開始數據轉換... Starting data conversion...")
            
            # 提取参数
            source_path = self.kwargs['source_path']
            output_path = self.kwargs.get('output_path')
            use_depth = self.kwargs.get('use_depth', True)
            use_stereo = self.kwargs.get('use_stereo', False)
            folder_count_limit = self.kwargs.get('folder_count_limit')
            train_ratio = self.kwargs.get('train_ratio')
            val_ratio = self.kwargs.get('val_ratio')
            test_ratio = self.kwargs.get('test_ratio')
            
            # 验证源路径
            if not Path(source_path).exists():
                raise FileNotFoundError(f"源路徑不存在 Source path does not exist: {source_path}")
            
            # 输出转换模式信息
            if use_stereo:
                mode_desc = "立體視覺數據 Stereo Vision Data"
                self.log_message.emit("🔄 開始立體視覺數據轉換... Starting stereo data conversion...")
            elif use_depth:
                mode_desc = "4通道RGBD數據 4-Channel RGBD Data"
                self.log_message.emit("🔄 開始4通道數據轉換... Starting 4-channel data conversion...")
            else:
                mode_desc = "3通道RGB數據 3-Channel RGB Data"
                self.log_message.emit("🔄 開始3通道數據轉換... Starting 3-channel data conversion...")
            
            self.log_message.emit(f"源路徑 Source: {source_path}")
            if output_path:
                self.log_message.emit(f"輸出路徑 Output: {output_path}")
            self.log_message.emit(f"數據模式 Mode: {mode_desc}")
            
            # 根据选项创建对应的预处理器
            preprocessor_kwargs = {
                'source_path': source_path,
                'output_path': output_path,
                'folder_count_limit': folder_count_limit
            }
            
            # 如果提供了自定义分割比例，添加到参数中
            if train_ratio is not None:
                preprocessor_kwargs['train_ratio'] = train_ratio
            if val_ratio is not None:
                preprocessor_kwargs['val_ratio'] = val_ratio
            if test_ratio is not None:
                preprocessor_kwargs['test_ratio'] = test_ratio
            
            if use_stereo:
                preprocessor = StereoPreprocessor(**preprocessor_kwargs)
            else:
                preprocessor_kwargs['use_depth'] = use_depth
                preprocessor = RGBPreprocessor(**preprocessor_kwargs)
            
            # 处理数据
            preprocessor.process_all_data()
            
            self.log_message.emit("[SUCCESS] 數據轉換完成! Data conversion completed!")
            self.log_message.emit(f"[FOLDER] 數據集保存在 Dataset saved at: {preprocessor.output_path}")
            
            self.progress.emit("數據轉換完成 Data conversion completed")
            
        except Exception as e:
            error_msg = f"[ERROR] 數據轉換失敗 Data conversion failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.progress.emit("數據轉換失敗 Data conversion failed")
            raise e
    
    
    def _inference(self):
        """推理处理 - Inference Processing"""
        try:
            self.progress.emit("正在開始推理... Starting inference...")
            self.log_message.emit("🎯 開始推理處理... Starting inference processing...")
            
            # 获取推理参数
            model_path = self.kwargs.get('model_path')
            data_path = self.kwargs.get('data_path')
            output_path = self.kwargs.get('output_path')
            confidence = self.kwargs.get('confidence', 0.25)
            iou_threshold = self.kwargs.get('iou_threshold', 0.45)
            max_det = self.kwargs.get('max_det', 300)
            inference_mode = self.kwargs.get('inference_mode', 'single')
            
            # 验证参数
            if not model_path:
                raise ValueError("模型路径不能为空 Model path cannot be empty")
            
            if not data_path:
                raise ValueError("数据路径不能为空 Data path cannot be empty")
            
            # 检测设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log_message.emit(f"使用设备 Using device: {device}")
            
            # 导入推理模块
            try:
                from Code.yolo_inference import enhanced_inference
                self.log_message.emit("✅ 成功載入推理模組 Successfully loaded inference module")
            except ImportError as e:
                self.log_message.emit(f"❌ 無法導入推理模組 Failed to import inference module: {e}")
                raise e
            
            # 设置输出路径
            if not output_path:
                output_path = "Predict/Result"
            
            # 记录推理参数
            self.log_message.emit(f"模型 Model: {model_path}")
            self.log_message.emit(f"数据 Data: {data_path}")
            self.log_message.emit(f"输出 Output: {output_path}")
            self.log_message.emit(f"置信度 Confidence: {confidence}")
            self.log_message.emit(f"IoU阈值 IoU threshold: {iou_threshold}")
            self.log_message.emit(f"最大检测 Max detections: {max_det}")
            self.log_message.emit(f"推理模式 Inference mode: {inference_mode}")
            
            # 执行推理
            self.log_message.emit("🚀 開始執行推理... Starting inference execution...")
            
            results = enhanced_inference(
                model_path=model_path,
                confidence_threshold=confidence,
                device=device,
                predict_data_dir=data_path,
                iou_threshold=iou_threshold,
                max_det=max_det,
                line_width=3,
                show_labels=True,
                show_conf=True,
                show_boxes=True,
                save_txt=True,
                save_conf=True,
                save_crop=False,
                visualize=True,
                augment=False,
                agnostic_nms=False,
                retina_masks=False,
                output_format='torch',
                verbose=False,
                show=False
            )
            
            # 处理推理结果
            if results:
                self.log_message.emit(f"✅ 推理完成，處理了 {len(results)} 個結果")
                self.log_message.emit(f"✅ Inference completed, processed {len(results)} results")
            else:
                self.log_message.emit("⚠️ 推理完成，但未檢測到任何目標")
                self.log_message.emit("⚠️ Inference completed but no targets detected")
            
            self.log_message.emit(f"[FOLDER] 結果保存在 Results saved to: {output_path}")
            self.progress.emit("推理完成 Inference completed")
            
        except Exception as e:
            error_msg = f"[ERROR] 推理失敗 Inference failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.progress.emit("推理失敗 Inference failed")
            raise e
    
    def _stereo_inference(self):
        """立体视觉推理处理 - Stereo Inference Processing"""
        try:
            self.progress.emit("正在開始立體視覺推理... Starting stereo inference...")
            self.log_message.emit("🔍 開始立體視覺推理處理... Starting stereo inference processing...")
            
            # 获取推理参数
            model_path = self.kwargs.get('model_path')
            left_imgs = self.kwargs.get('left_imgs')
            right_imgs = self.kwargs.get('right_imgs')
            output_dir = self.kwargs.get('output_dir', 'demo_output')
            valid_iters = self.kwargs.get('valid_iters', 32)
            mixed_precision = self.kwargs.get('mixed_precision', True)
            save_numpy = self.kwargs.get('save_numpy', False)
            output_format = self.kwargs.get('output_format', 'png')  # 輸出格式
            flip_non_pfm = self.kwargs.get('flip_non_pfm', False)  # 非PFM格式翻轉
            
            # 验证参数
            if not model_path:
                raise ValueError("模型路径不能为空 Model path cannot be empty")
            
            if not left_imgs:
                raise ValueError("左图像路径不能为空 Left images path cannot be empty")
            
            if not right_imgs:
                raise ValueError("右图像路径不能为空 Right images path cannot be empty")
            
            # 检测设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log_message.emit(f"使用设备 Using device: {device}")
            
            # 导入 Code/raft_stereo_inference.py 中的函数和类
            # 由于该模块包含完整的模型实现，我们需要导入它
            project_root = Path(__file__).parent.parent.parent
            inference_module_path = project_root / "Code" / "raft_stereo_inference.py"
            
            if not inference_module_path.exists():
                raise FileNotFoundError(f"找不到立體視覺推理模組文件: {inference_module_path}")
            
            # 动态导入推理模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("raft_stereo_inference", inference_module_path)
            inference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_module)
            
            self.log_message.emit("✅ 成功載入立體視覺推理模組 Successfully loaded stereo inference module")
            
            # 创建参数对象
            from argparse import Namespace
            args = Namespace(
                restore_ckpt=model_path,
                save_numpy=save_numpy,
                left_imgs=left_imgs,
                right_imgs=right_imgs,
                output_directory=output_dir,
                mixed_precision=mixed_precision,
                valid_iters=valid_iters,
                hidden_dims=[128]*3,
                corr_implementation="alt",
                shared_backbone=False,
                corr_levels=4,
                corr_radius=4,
                n_downsample=2,
                context_norm="batch",
                slow_fast_gru=False,
                n_gru_layers=3,
                output_format=output_format,  # 輸出格式選項
                flip_non_pfm=flip_non_pfm  # 非PFM格式翻轉選項
            )
            
            # 记录推理参数
            self.log_message.emit(f"模型 Model: {Path(model_path).name}")
            self.log_message.emit(f"左圖像 Left images: {left_imgs}")
            self.log_message.emit(f"右圖像 Right images: {right_imgs}")
            self.log_message.emit(f"輸出目錄 Output directory: {output_dir}")
            self.log_message.emit(f"迭代次數 Iterations: {valid_iters}")
            self.log_message.emit(f"混合精度 Mixed precision: {'啟用' if mixed_precision else '禁用'}")
            
            # 检查图像文件
            import glob
            left_files = sorted(glob.glob(left_imgs, recursive=True))
            right_files = sorted(glob.glob(right_imgs, recursive=True))
            
            if not left_files:
                raise FileNotFoundError(f"未找到左圖像 No left images found: {left_imgs}")
            
            if not right_files:
                raise FileNotFoundError(f"未找到右圖像 No right images found: {right_imgs}")
            
            num_pairs = min(len(left_files), len(right_files))
            self.log_message.emit(f"找到 {len(left_files)} 張左圖像和 {len(right_files)} 張右圖像")
            self.log_message.emit(f"Found {len(left_files)} left images and {len(right_files)} right images")
            self.log_message.emit(f"將處理 {num_pairs} 對圖像 Will process {num_pairs} image pairs")
            
            # 执行推理
            self.log_message.emit("🚀 開始執行立體視覺推理... Starting stereo inference execution...")
            
            # 调用 raft_stereo_inference.py 中的 demo 函数
            inference_module.demo(args)
            
            self.log_message.emit(f"✅ 立體視覺推理完成！處理了 {num_pairs} 對圖像")
            self.log_message.emit(f"✅ Stereo inference completed! Processed {num_pairs} image pairs")
            self.log_message.emit(f"[FOLDER] 結果保存在 Results saved to: {output_dir}")
            self.progress.emit(f"立體視覺推理完成 Stereo inference completed - {num_pairs} pairs")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = f"[ERROR] 立體視覺推理失敗 Stereo inference failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.log_message.emit(f"詳細錯誤信息 Detailed error:\n{error_detail}")
            self.progress.emit("立體視覺推理失敗 Stereo inference failed")
            raise e
    
    def _train_with_pretrained(self):
        """使用預訓練模型訓練 - Train with pretrained model"""
        if not training_core:
            raise ImportError("訓練核心模組未找到 Training core module not found")
            
        try:
            self.progress.emit("正在開始模型訓練... Starting model training...")
            self.log_message.emit("🎯 開始模型訓練... Starting model training...")
            
            # 提取參數
            config_path = self.kwargs['config_path']
            model_file = self.kwargs.get('model_file')
            epochs = self.kwargs.get('epochs', 50)
            learning_rate = self.kwargs.get('learning_rate', 0.001)
            batch_size = self.kwargs.get('batch_size', 16)
            imgsz = self.kwargs.get('imgsz', 640)
            save_period = self.kwargs.get('save_period', 10)
            
            # 数据增强参数
            scale = self.kwargs.get('scale', 0.5)
            mosaic = self.kwargs.get('mosaic', 1.0)
            mixup = self.kwargs.get('mixup', 0.0)
            copy_paste = self.kwargs.get('copy_paste', 0.0)
            hsv_h = self.kwargs.get('hsv_h', 0.015)
            hsv_s = self.kwargs.get('hsv_s', 0.7)
            hsv_v = self.kwargs.get('hsv_v', 0.4)
            bgr = self.kwargs.get('bgr', 0.0)
            auto_augment = self.kwargs.get('auto_augment', None)
            
            # 几何变换参数
            degrees = self.kwargs.get('degrees', 0.0)
            translate = self.kwargs.get('translate', 0.1)
            shear = self.kwargs.get('shear', 0.0)
            perspective = self.kwargs.get('perspective', 0.0)
            
            # 翻转和裁剪参数
            flipud = self.kwargs.get('flipud', 0.0)
            fliplr = self.kwargs.get('fliplr', 0.5)
            erasing = self.kwargs.get('erasing', 0.0)
            crop_fraction = self.kwargs.get('crop_fraction', 1.0)
            
            # 优化器参数
            weight_decay = self.kwargs.get('weight_decay', 0.0005)
            momentum = self.kwargs.get('momentum', 0.937)
            beta1 = self.kwargs.get('beta1', 0.9)
            beta2 = self.kwargs.get('beta2', 0.999)
            
            # 学习率调度参数
            lr_scheduler = self.kwargs.get('lr_scheduler', 'auto')
            lr_decay = self.kwargs.get('lr_decay', 0.1)
            warmup_epochs = self.kwargs.get('warmup_epochs', 3)
            warmup_momentum = self.kwargs.get('warmup_momentum', 0.8)
            
            # 验证参数
            val_frequency = self.kwargs.get('val_frequency', 1)
            val_iters = self.kwargs.get('val_iters', 32)
            early_stopping_patience = self.kwargs.get('early_stopping_patience', 50)
            early_stopping_min_delta = self.kwargs.get('early_stopping_min_delta', 0.001)
            
            # 设备参数
            device = self.kwargs.get('device', 'auto')
            multi_gpu = self.kwargs.get('multi_gpu', False)
            gpu_memory_optimization = self.kwargs.get('gpu_memory_optimization', True)
            data_loading_optimization = self.kwargs.get('data_loading_optimization', True)
            
            # 其他高级参数
            close_mosaic = self.kwargs.get('close_mosaic', 10)
            single_cls = self.kwargs.get('single_cls', False)
            cache = self.kwargs.get('cache', False)
            resume = self.kwargs.get('resume', False)
            workers = self.kwargs.get('workers', 8)
            optimizer = self.kwargs.get('optimizer', 'auto')
            amp = self.kwargs.get('amp', True)
            
            # 定義回調函數
            def progress_callback(message):
                self.progress.emit(message)
                self.log_message.emit(message)
            
            def log_callback(message):
                self.log_message.emit(message)
            
            # 調用訓練核心
            result = training_core.train_with_pretrained(
                config_path=config_path,
                model_path=model_file,
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
                amp=amp,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if result['success']:
                self.log_message.emit("[SUCCESS] 訓練完成! Training completed!")
                self.progress.emit("訓練完成 Training completed")
            else:
                raise Exception(result['message'])
                
        except Exception as e:
            error_msg = f"[ERROR] 訓練失敗 Training failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.progress.emit("訓練失敗 Training failed")
            raise e
    
    def _train_with_yaml(self):
        """使用YAML配置從頭訓練 - Train from scratch with YAML config"""
        if not training_core:
            raise ImportError("訓練核心模組未找到 Training core module not found")
            
        try:
            self.progress.emit("正在開始模型訓練... Starting model training...")
            self.log_message.emit("🎯 開始模型訓練... Starting model training...")
            
            # 提取參數
            config_path = self.kwargs['config_path']
            model_file = self.kwargs.get('model_file')
            model_size = self.kwargs.get('model_size', 'n')
            epochs = self.kwargs.get('epochs', 50)
            learning_rate = self.kwargs.get('learning_rate', 0.001)
            batch_size = self.kwargs.get('batch_size', 16)
            imgsz = self.kwargs.get('imgsz', 640)
            save_period = self.kwargs.get('save_period', -1)
            scale = self.kwargs.get('scale', 0.5)
            mosaic = self.kwargs.get('mosaic', 1.0)
            mixup = self.kwargs.get('mixup', 0.0)
            copy_paste = self.kwargs.get('copy_paste', 0.0)
            hsv_h = self.kwargs.get('hsv_h', 0.015)
            hsv_s = self.kwargs.get('hsv_s', 0.7)
            hsv_v = self.kwargs.get('hsv_v', 0.4)
            bgr = self.kwargs.get('bgr', 0.0)
            auto_augment = self.kwargs.get('auto_augment', None)
            degrees = self.kwargs.get('degrees', 0.0)
            translate = self.kwargs.get('translate', 0.1)
            shear = self.kwargs.get('shear', 0.0)
            perspective = self.kwargs.get('perspective', 0.0)
            flipud = self.kwargs.get('flipud', 0.0)
            fliplr = self.kwargs.get('fliplr', 0.5)
            erasing = self.kwargs.get('erasing', 0.0)
            crop_fraction = self.kwargs.get('crop_fraction', 1.0)
            close_mosaic = self.kwargs.get('close_mosaic', 10)
            workers = self.kwargs.get('workers', 8)
            optimizer = self.kwargs.get('optimizer', 'auto')
            amp = self.kwargs.get('amp', True)
            
            # 定義回調函數
            def progress_callback(message):
                self.progress.emit(message)
                self.log_message.emit(message)
            
            def log_callback(message):
                self.log_message.emit(message)
            
            # 調用訓練核心
            result = training_core.train_with_yaml(
                config_path=config_path,
                yaml_path=model_file,
                model_size=model_size,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                imgsz=imgsz,
                save_period=save_period,
                scale=scale,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                bgr=bgr,
                auto_augment=auto_augment,
                degrees=degrees,
                translate=translate,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                erasing=erasing,
                crop_fraction=crop_fraction,
                close_mosaic=close_mosaic,
                workers=workers,
                optimizer=optimizer,
                amp=amp,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if result['success']:
                self.log_message.emit("[SUCCESS] 訓練完成! Training completed!")
                self.progress.emit("訓練完成 Training completed")
            else:
                raise Exception(result['message'])
                
        except Exception as e:
            error_msg = f"[ERROR] 訓練失敗 Training failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.progress.emit("訓練失敗 Training failed")
            raise e
    
    def _train_stereo(self):
        """立體視覺訓練 - Stereo Vision Training"""
        try:
            self.progress.emit("正在開始立體視覺訓練... Starting stereo vision training...")
            self.log_message.emit("🎯 開始立體視覺訓練... Starting stereo vision training...")
            
            # 提取基本參數
            dataset_path = self.kwargs['dataset_path']
            model_name = self.kwargs.get('model_name', 'raftstereo-sceneflow.pth')
            batch_size = self.kwargs.get('batch_size', 6)
            # 支持num_steps和epochs（向后兼容）
            num_steps = self.kwargs.get('num_steps')
            if num_steps is None:
                # 向后兼容：如果提供了epochs，转换为num_steps
                epochs = self.kwargs.get('epochs', 100)
                num_steps = epochs * 1000
            output_dir = self.kwargs.get('output_dir', 'checkpoints')
            
            # 提取高級參數
            train_iters = self.kwargs.get('train_iters', 16)
            valid_iters = self.kwargs.get('valid_iters', 32)
            corr_implementation = self.kwargs.get('corr_implementation', 'reg')
            mixed_precision = self.kwargs.get('mixed_precision', False)
            n_downsample = self.kwargs.get('n_downsample', 2)
            corr_levels = self.kwargs.get('corr_levels', 4)
            corr_radius = self.kwargs.get('corr_radius', 4)
            n_gru_layers = self.kwargs.get('n_gru_layers', 3)
            learning_rate = self.kwargs.get('learning_rate', 0.0002)
            weight_decay = self.kwargs.get('weight_decay', 0.00001)
            
            # 獲取 image_size 參數
            # 注意：image_size 用於記錄目的，實際訓練中不使用裁剪（保持原始圖像尺寸）
            # GUI 傳遞的是 [width, height]，需要轉換為 TrainingConfig 期望的 (height, width)
            image_size_raw = self.kwargs.get('image_size', [320, 720])
            if isinstance(image_size_raw, list) and len(image_size_raw) == 2:
                # GUI 格式是 [width, height]，轉換為 (height, width)
                image_size = [image_size_raw[1], image_size_raw[0]]  # [height, width]
            else:
                image_size = [320, 720]  # 默認值 (height, width)
            
            # 注意：crop_size 設置為 None，以保持原始圖像尺寸
            # 這避免了當數據集圖像小於裁剪尺寸時的錯誤
            
            # 提取 hidden_dims 參數
            hidden_dims_raw = self.kwargs.get('hidden_dims')
            if hidden_dims_raw:
                if isinstance(hidden_dims_raw, str):
                    # 如果是字符串格式 "128x128x128 (默認)"，解析它
                    try:
                        dims_part = hidden_dims_raw.split('(')[0].strip()
                        hidden_dims = [int(d) for d in dims_part.split('x')]
                    except:
                        hidden_dims = [128, 128, 128]
                elif isinstance(hidden_dims_raw, list):
                    hidden_dims = hidden_dims_raw
                else:
                    hidden_dims = [128, 128, 128]
            else:
                # 根據 n_gru_layers 生成默認值
                hidden_dims = [128] * n_gru_layers
            
            # 增廣參數
            spatial_scale_min = self.kwargs.get('spatial_scale_min', -0.2)
            spatial_scale_max = self.kwargs.get('spatial_scale_max', 0.4)
            saturation_min = self.kwargs.get('saturation_min', 0.0)
            saturation_max = self.kwargs.get('saturation_max', 1.4)
            gamma_min = self.kwargs.get('gamma_min', 0.8)
            gamma_max = self.kwargs.get('gamma_max', 1.2)
            
            # 處理 do_flip 參數：將 "無 None" 轉換為 None
            do_flip_raw = self.kwargs.get('do_flip', '無 None')
            if do_flip_raw in ['無 None', 'None', '無', None, False]:
                do_flip = None
            else:
                do_flip = do_flip_raw
            
            noyjitter = self.kwargs.get('noyjitter', False)
            
            # 記錄訓練參數
            self.log_message.emit(f"🚀 立體視覺訓練參數:")
            self.log_message.emit(f"   數據集: {dataset_path}")
            self.log_message.emit(f"   預訓練模型: {model_name}")
            self.log_message.emit(f"   訓練參數: 步數={num_steps}, 批次={batch_size}")
            self.log_message.emit(f"   迭代參數: 訓練={train_iters}, 驗證={valid_iters}")
            self.log_message.emit(f"   圖像尺寸: {image_size[0]}x{image_size[1]} (height x width)")
            self.log_message.emit(f"   相關實現: {corr_implementation}")
            self.log_message.emit(f"   模型架構: n_downsample={n_downsample}, corr_levels={corr_levels}, corr_radius={corr_radius}")
            self.log_message.emit(f"   GRU層數: {n_gru_layers}, Hidden Dims: {hidden_dims}")
            self.log_message.emit(f"   優化選項: 混合精度={mixed_precision}, 學習率={learning_rate}, 權重衰減={weight_decay}")
            
            # 導入必要的模組
            try:
                # 確保 Code 目錄在 sys.path 中
                if str(code_dir) not in sys.path:
                    sys.path.insert(0, str(code_dir))
                
                # 直接導入，因為 Code 目錄已在 sys.path 中
                import importlib.util
                trainer_path = code_dir / "raft_stereo_trainer.py"
                if not trainer_path.exists():
                    raise FileNotFoundError(f"找不到訓練器文件: {trainer_path}")
                
                spec = importlib.util.spec_from_file_location("raft_stereo_trainer", trainer_path)
                raft_stereo_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(raft_stereo_module)
                RAFTStereoTrainer = raft_stereo_module.RAFTStereoTrainer
                
                from config.config import TrainingConfig
                self.log_message.emit("✅ 成功載入立體視覺訓練模組 Successfully loaded stereo training modules")
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                self.log_message.emit(f"❌ 無法導入立體視覺訓練模組 Failed to import stereo training modules: {e}")
                self.log_message.emit(f"詳細錯誤信息 Detailed error: {error_detail}")
                raise e
            
            # 創建帶時間戳的輸出資料夾
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%dT%H%M")
            output_folder = f"runs/raft_stereo_{timestamp}"
            import os
            os.makedirs(output_folder, exist_ok=True)
            
            self.log_message.emit(f"創建輸出資料夾: {output_folder}")
            self.log_message.emit(f"Created output folder: {output_folder}")
            
            # 構建預訓練模型路徑
            restore_ckpt = None
            if model_name:
                # 檢查模型文件是否存在
                model_paths = [
                    Path("Model_file/Stereo_Vision") / model_name,
                    Path("Model_file/PTH_File") / model_name,  # 向後兼容舊目錄
                    Path("Model_file") / model_name,
                    Path(model_name),  # 如果提供的是完整路徑
                ]
                
                for mp in model_paths:
                    if mp.exists():
                        restore_ckpt = str(mp.absolute())
                        self.log_message.emit(f"✅ 找到預訓練模型: {restore_ckpt}")
                        break
                
                if restore_ckpt is None:
                    self.log_message.emit(f"⚠️ 警告: 未找到預訓練模型 {model_name}，將從頭開始訓練")
                    self.log_message.emit(f"⚠️ Warning: Pretrained model {model_name} not found, training from scratch")
            
            # 創建訓練配置
            # 注意：image_size 僅用於記錄，實際訓練使用原始圖像尺寸（不裁剪）
            config = TrainingConfig(
                name=f"raft-stereo-{timestamp}",
                train_datasets=['drone'],
                dataset_root=dataset_path,
                batch_size=batch_size,
                num_steps=num_steps,
                train_iters=train_iters,
                valid_iters=valid_iters,
                lr=learning_rate,
                wdecay=weight_decay,
                image_size=tuple(image_size),  # 確保是元組格式 (height, width)，但不用於裁剪
                corr_implementation=corr_implementation,
                mixed_precision=mixed_precision,
                n_downsample=n_downsample,
                corr_levels=corr_levels,
                corr_radius=corr_radius,
                n_gru_layers=n_gru_layers,
                hidden_dims=hidden_dims,  # 添加 hidden_dims 參數
                spatial_scale=(spatial_scale_min, spatial_scale_max),
                saturation_range=[saturation_min, saturation_max] if (saturation_min != 0.0 or saturation_max != 1.4) else None,
                img_gamma=[gamma_min, gamma_max] if (gamma_min != 0.8 or gamma_max != 1.2) else None,
                do_flip=do_flip,  # 已在前面處理過，直接使用
                noyjitter=noyjitter,
                output_dir=output_folder,
                restore_ckpt=restore_ckpt
            )
            
            self.log_message.emit(f"📋 配置詳情:")
            self.log_message.emit(f"   image_size: {config.image_size} (記錄用途，不裁剪)")
            self.log_message.emit(f"   spatial_scale: {config.spatial_scale}")
            self.log_message.emit(f"   實際訓練將使用原始圖像尺寸（無裁剪）")
            
            # 驗證配置
            if not config.validate():
                self.log_message.emit("配置驗證失敗，請檢查參數設置")
                self.log_message.emit("Configuration validation failed, please check parameters")
                raise ValueError("配置驗證失敗 Configuration validation failed")
            
            self.log_message.emit("準備開始訓練...")
            self.log_message.emit("Prepare to start training...")
            self.log_message.emit(f"使用配置: {config.name}")
            self.log_message.emit(f"Using configuration: {config.name}")
            if restore_ckpt:
                self.log_message.emit(f"預訓練模型: {restore_ckpt}")
                self.log_message.emit(f"Pretrained model: {restore_ckpt}")
            self.log_message.emit("-" * 50)
            
            # 設置日誌
            import logging
            logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
            
            # 檢查停止請求
            if self._stop_requested:
                self.log_message.emit("訓練已取消 Training cancelled")
                return
            
            # 創建訓練器並執行訓練
            self.log_message.emit("正在初始化訓練器... Initializing trainer...")
            trainer = RAFTStereoTrainer(config)
            self._current_trainer = trainer  # 保存訓練器引用以便停止
            self.log_message.emit("✅ 訓練器初始化完成 Trainer initialized")
            
            # 檢查停止請求
            if self._stop_requested:
                self.log_message.emit("訓練已取消 Training cancelled")
                return
            
            self.log_message.emit("🚀 開始執行訓練... Starting training...")
            
            # 創建進度回調函數
            def progress_callback(current_step, total_steps, message):
                """進度回調函數"""
                if self._stop_requested:
                    return  # 如果請求停止，不再更新進度
                # 發送進度消息（包含步數信息，用於解析）
                progress_msg = f"Step {current_step}/{total_steps}: {message}"
                self.progress.emit(progress_msg)
                self.epoch_progress.emit(current_step, total_steps, message)
            
            # 執行訓練，傳遞進度回調
            result_path = trainer.train(progress_callback=progress_callback)
            
            self.log_message.emit("-" * 50)
            self.log_message.emit("訓練完成！")
            self.log_message.emit("Training completed!")
            self.log_message.emit(f"模型保存路徑: {result_path}")
            self.log_message.emit(f"Model saved to: {result_path}")
            self.log_message.emit(f"完整的訓練輸出位於 Complete training output located at: {output_folder}")
            
            self.log_message.emit("[SUCCESS] 立體視覺訓練完成! Stereo vision training completed!")
            self.progress.emit("立體視覺訓練完成 Stereo vision training completed")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = f"[ERROR] 立體視覺訓練失敗 Stereo vision training failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.log_message.emit(f"詳細錯誤信息 Detailed error traceback:")
            self.log_message.emit(error_detail)
            self.progress.emit("立體視覺訓練失敗 Stereo vision training failed")
            raise e