"""
YOLO 統一啟動器 - 圖形化界面版本
基於PyQt5的現代化圖形界面
整合4通道訓練、數據轉換功能
"""

import sys
import os
import torch
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QTabWidget, QLabel, 
                            QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                            QFileDialog, QMessageBox, QProgressBar,
                            QStatusBar, QGroupBox, QTextEdit,
                            QFrame, QRadioButton, QButtonGroup, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QFont

# 添加Code目錄到Python路徑
code_dir = os.path.join(os.path.dirname(__file__), "Code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# 導入現有模組
from Code.data_converter import RGBPreprocessor, StereoPreprocessor

# 導入yolo_inference模組
import yolo_inference  # type: ignore

# 導入Read_Model模組
from Code.Read_Model import get_model_info, find_pt_files

class WorkerThread(QThread):
    """工作線程類"""
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
        
    def run(self):
        try:
            if self._stop_requested:
                return
                
            if self.task_type == "train":
                self._train_model()
            elif self.task_type == "convert":
                self._convert_data()
            elif self.task_type == "inference":
                self._inference()
            elif self.task_type == "inference_test":
                self._inference_test()
            elif self.task_type == "stereo_training":
                self._stereo_training()
            
            if not self._stop_requested:
                self.finished.emit(True, "任務完成")
        except Exception as e:
            if not self._stop_requested:
                self.finished.emit(False, str(e))
    
    def stop(self):
        """安全停止線程"""
        self._stop_requested = True
        
        # 如果正在训练，请求训练器停止
        if hasattr(self, '_current_trainer') and self._current_trainer:
            try:
                self._current_trainer.request_stop()
            except Exception as e:
                pass  # 静默处理
        
        # 釋放PyTorch和CUDA資源
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            # 靜默處理CUDA資源釋放錯誤
            pass
        
        self.quit()
        self.wait(3000)  # 等待3秒
        if self.isRunning():
            self.terminate()
            self.wait(1000)  # 再等待1秒
    
    def _get_conversion_mode_msg(self, use_depth, use_stereo):
        """獲取轉換模式信息 - WorkerThread版本 (Get conversion mode info)"""
        if use_stereo:
            return "立體視覺數據 Stereo Vision Data", "🔄 開始立體視覺數據轉換... Starting stereo data conversion..."
        elif use_depth:
            return "4通道RGBD數據 4-Channel RGBD Data", "🔄 開始4通道數據轉換... Starting 4-channel data conversion..."
        else:
            return "3通道RGB數據 3-Channel RGB Data", "🔄 開始3通道數據轉換... Starting 3-channel data conversion..."
    
    def _handle_error(self, task_name, exception):
        """統一的錯誤處理 - 避免重复代码 (Unified error handling)"""
        error_msg = f"[ERROR] {task_name}失敗 failed: {str(exception)}"
        self.log_message.emit(error_msg)
        self.log_message.emit(f"錯誤類型 Error type: {type(exception).__name__}")
        
        # 打印詳細錯誤信息
        import traceback
        error_details = traceback.format_exc()
        self.log_message.emit("詳細錯誤信息 Detailed error:")
        for line in error_details.split('\n'):
            if line.strip():
                self.log_message.emit(f"  {line}")
        
        self.progress.emit(f"{task_name}失敗 {task_name} failed")
            
    def _train_model(self):
        """標準模型訓練"""
        self.progress.emit("正在開始模型訓練...")
        self.log_message.emit("🎯 開始模型訓練...")
        
        config_path = self.kwargs['config_path']
        model_file = self.kwargs.get('model_file')
        epochs = self.kwargs.get('epochs', 50)
        learning_rate = self.kwargs.get('learning_rate', 0.001)
        batch_size = self.kwargs.get('batch_size', 16)
        
        # 新增的高級訓練參數
        imgsz = self.kwargs.get('imgsz', 640)
        save_period = self.kwargs.get('save_period', 10)
        scale = self.kwargs.get('scale', 0.5)
        mosaic = self.kwargs.get('mosaic', 1.0)
        mixup = self.kwargs.get('mixup', 0.0)
        copy_paste = self.kwargs.get('copy_paste', 0.1)
        
        # 新增的HSV和BGR增強參數
        hsv_h = self.kwargs.get('hsv_h', 0)
        hsv_s = self.kwargs.get('hsv_s', 0)
        hsv_v = self.kwargs.get('hsv_v', 0)
        bgr = self.kwargs.get('bgr', 0)
        auto_augment = self.kwargs.get('auto_augment', None)
        
        # 新增的幾何變換參數
        degrees = self.kwargs.get('degrees', 0)
        translate = self.kwargs.get('translate', 0)
        shear = self.kwargs.get('shear', 0)
        perspective = self.kwargs.get('perspective', 0)
        
        # 新增的翻轉和裁剪參數
        flipud = self.kwargs.get('flipud', 0)
        fliplr = self.kwargs.get('fliplr', 0)
        erasing = self.kwargs.get('erasing', 0)
        crop_fraction = self.kwargs.get('crop_fraction', 0)
        
        # 新增的訓練控制參數
        close_mosaic = self.kwargs.get('close_mosaic', 10)
        workers = self.kwargs.get('workers', 0)
        optimizer = self.kwargs.get('optimizer', 'SGD')
        amp = self.kwargs.get('amp', True)
        
        try:
            # 使用與train.py相同的邏輯 - 直接使用ultralytics YOLO
            import warnings
            warnings.filterwarnings('ignore')
            from ultralytics import YOLO
            
            # 根據訓練模式選擇不同的處理方式
            training_mode = self.kwargs.get('training_mode', 'pretrained')
            
            if training_mode == 'retrain' and model_file and str(model_file).endswith('.yaml'):
                # 重新訓練模式 - 使用YAML配置文件
                model_size = self.kwargs.get('model_size', 'n')
                self.log_message.emit(f"📋 重新訓練模式 - 使用YAML配置: {model_file}")
                self.log_message.emit(f"📋 重新訓練模式 - 模型大小: {model_size}")
                
                # 構建帶有模型大小的YAML路徑
                base_name = Path(model_file).stem
                sized_yaml = f"{base_name}{model_size}.yaml"
                self.log_message.emit(f"📋 重新訓練模式 - 將使用: {sized_yaml}")
                
                # 檢查帶有模型大小的YAML文件是否存在
                sized_yaml_path = Path(sized_yaml)
                if sized_yaml_path.exists():
                    self.log_message.emit(f"📋 使用帶有模型大小的YAML文件: {sized_yaml}")
                    model = YOLO(model=sized_yaml)
                else:
                    self.log_message.emit(f"📋 帶有模型大小的YAML文件不存在，使用基礎文件: {model_file}")
                    model = YOLO(model=model_file)
            else:
                # 預訓練模式 - 使用PT模型文件
                self.log_message.emit(f"📋 預訓練模式 - 使用PT模型: {model_file}")
                # 使用標準訓練器模組
                from Code.YOLO_standard_trainer import YOLOStandardTrainer
                
                # 創建訓練器
                trainer = YOLOStandardTrainer(
                    config_path=config_path,
                    model_path=model_file,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    scale=scale,
                    mosaic=mosaic,
                    mixup=mixup,
                    copy_paste=copy_paste,
                    hsv_h=hsv_h,
                    hsv_s=hsv_s,
                    hsv_v=hsv_v,
                    bgr=bgr,
                    auto_augment=auto_augment,
                    # 新增的幾何變換參數
                    degrees=degrees,
                    translate=translate,
                    shear=shear,
                    perspective=perspective,
                    # 新增的翻轉和裁剪參數
                    flipud=flipud,
                    fliplr=fliplr,
                    erasing=erasing,
                    crop_fraction=crop_fraction,
                    # 新增的訓練控制參數
                    close_mosaic=close_mosaic,
                    workers=workers,
                    optimizer=optimizer,
                    amp=amp
                )
            
            # 定義回調函數 - 改進版
            def progress_callback(message):
                self.progress.emit(message)
                
                # 解析epoch進度信息
                if "Epoch" in message and "/" in message:
                    try:
                        parts = message.split()
                        for i, part in enumerate(parts):
                            if part == "Epoch" and i + 1 < len(parts):
                                epoch_info = parts[i + 1]
                                if "/" in epoch_info:
                                    current, total = epoch_info.split("/")
                                    current = int(current)
                                    total = int(total)
                                    self.epoch_progress.emit(current, total, message)
                                    break
                    except:
                        pass
            
            def log_callback(message):
                self.log_message.emit(message)
            
            # 生成自定義模型名稱：{RGBD or RGB}_{model_name}_{epoch}_{時間戳}
            try:
                # 讀取數據集配置以確定通道類型
                import yaml
                from datetime import datetime
                with open(config_path, 'r', encoding='utf-8') as f:
                    dataset_config = yaml.safe_load(f)
                
                channels = dataset_config.get('channels', 3)
                channel_type = 'RGBD' if channels == 4 else 'RGB'
                
                # 獲取模型名稱
                if training_mode == 'retrain':
                    model_name = Path(model_file).stem  # 例如: yolo12
                    model_size = self.kwargs.get('model_size', 'n')
                    full_model_name = f"{model_name}{model_size}"  # 例如: yolo12n
                else:
                    model_name = Path(model_file).stem  # 例如: yolov12n
                    full_model_name = model_name
                
                # 生成時間戳
                timestamp = datetime.now().strftime("%Y%m%d")
                
                # 生成基礎模型名稱
                base_custom_name = f"{channel_type}_{full_model_name}_{epochs}epoch_{timestamp}"
                
                # 檢查資料夾是否已存在，如果存在則添加序號
                custom_name = self._get_unique_training_folder_name(base_custom_name)
                self.log_message.emit(f"📋 自定義模型名稱: {custom_name}")
                
            except Exception as e:
                self.log_message.emit(f"⚠️ 生成自定義模型名稱失敗，使用默認名稱: {e}")
                custom_name = 'exp'
            
            # 執行訓練 - 根據訓練模式選擇不同的訓練方式
            if training_mode == 'retrain' and model_file and str(model_file).endswith('.yaml'):
                # 重新訓練模式 - 使用YAML配置文件從頭開始訓練
                self.log_message.emit("🚀 重新訓練模式 - 從頭開始訓練...")
                
                # 使用與train.py相同的參數
                results = model.train(
                    data=config_path,
                    imgsz=imgsz,
                    epochs=epochs,
                    batch=batch_size,
                    amp=amp,
                    workers=workers,
                    device='',
                    optimizer=optimizer,
                    close_mosaic=close_mosaic,
                    resume=False,
                    project='runs/train',
                    name=custom_name,  # 使用自定義名稱
                    single_cls=False,
                    cache=False,
                    save_period=save_period,  # 檢查點保存週期
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
            else:
                # 預訓練模式 - 使用PT模型進行微調訓練
                self.log_message.emit("🚀 預訓練模式 - 使用預訓練權重進行微調...")
                results = trainer.train(
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
            
            self.progress.emit("訓練完成")
            return results
            
        except Exception as e:
            self.log_message.emit(f"[ERROR] 訓練出錯: {e}")
            self.progress.emit("訓練出錯")
            raise e
                
    def _convert_data(self):
        """數據轉換 - 优化后的版本 (Optimized version)"""
        try:
            self.progress.emit("正在開始數據轉換... Starting data conversion...")
            
            # 提取參數
            source_path = self.kwargs['source_path']
            output_path = self.kwargs.get('output_path')
            use_depth = self.kwargs.get('use_depth', True)
            use_stereo = self.kwargs.get('use_stereo', False)
            folder_count_limit = self.kwargs.get('folder_count_limit')
            
            # 驗證源路徑
            if not Path(source_path).exists():
                raise FileNotFoundError(f"源路徑不存在 Source path does not exist: {source_path}")
            
            # 統一的轉換模式信息輸出
            mode_desc, mode_log = self._get_conversion_mode_msg(use_depth, use_stereo)
            self.log_message.emit(mode_log)
            self.log_message.emit(f"源路徑 Source: {source_path}")
            if output_path:
                self.log_message.emit(f"輸出路徑 Output: {output_path}")
            self.log_message.emit(f"數據模式 Mode: {mode_desc}")
            
            # 根據選項創建對應的預處理器
            if use_stereo:
                preprocessor = StereoPreprocessor(
                    source_path=source_path,
                    output_path=output_path,
                    folder_count_limit=folder_count_limit
                )
            else:
                preprocessor = RGBPreprocessor(
                    source_path=source_path,
                    output_path=output_path,
                    folder_count_limit=folder_count_limit,
                    use_depth=use_depth
                )
            
            # 處理數據
            preprocessor.process_all_data()
            
            self.log_message.emit("[OK] 數據轉換完成!")
            self.log_message.emit(f"[FOLDER] 數據集保存在: {preprocessor.output_path}")
            
            
            self.progress.emit("數據轉換完成 Data conversion completed")
            
        except Exception as e:
            self._handle_error("數據轉換 Data conversion", e)
            raise e
    
    def _inference(self):
        """推理處理"""
        try:
            self.progress.emit("正在開始推理...")
            self.log_message.emit("🎯 開始推理處理...")
            
            # 獲取推理參數
            model_path = self.kwargs.get('model_path', 'Model_file/PT_File/yolo12n_RGBD.pt')
            confidence_threshold = self.kwargs.get('confidence_threshold', 0.25)
            num_classes = self.kwargs.get('num_classes', 1)
            inference_mode = self.kwargs.get('inference_mode', 'Data目錄處理模式')
            dataset_path = self.kwargs.get('dataset_path', None)
            
            # 檢測設備
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.log_message.emit(f"模型: {model_path}")
            self.log_message.emit(f"置信度閾值: {confidence_threshold:.2f}")
            self.log_message.emit(f"類別數量: {num_classes}")
            self.log_message.emit(f"推理模式: {inference_mode}")
            
            # 使用yolo_inference.py進行推理
            import sys
            import os
            
            # 檢查yolo_inference模組是否可用
            if yolo_inference is None:
                self.log_message.emit("❌ yolo_inference模組未載入")
                return
            
            try:
                from yolo_inference import enhanced_inference  # type: ignore
                self.log_message.emit("✅ 成功載入增強版推理模組")
            except ImportError as e:
                self.log_message.emit(f"❌ 無法導入yolo_inference.enhanced_inference: {e}")
                return
            
            # 設置預測數據目錄
            predict_data_dir = "Predict/Data"
            if not os.path.exists(predict_data_dir):
                os.makedirs(predict_data_dir, exist_ok=True)
                self.log_message.emit(f"📁 創建預測數據目錄: {predict_data_dir}")
            
            # 檢查預測數據目錄中是否有文件
            image_files = []
            for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
                if os.path.exists(predict_data_dir):
                    image_files.extend([f for f in os.listdir(predict_data_dir) if f.lower().endswith(ext)])
            
            if not image_files:
                self.log_message.emit(f"⚠️ 預測數據目錄 {predict_data_dir} 中未找到圖片文件")
                self.log_message.emit("💡 請將圖片文件（.npy, .jpg, .png等）放入 Predict/Data 目錄")
                return
            
            self.log_message.emit(f"📊 找到 {len(image_files)} 個圖片文件")
            
            # 獲取高級推理參數
            iou_threshold = self.kwargs.get('iou_threshold', 0.45)
            max_det = self.kwargs.get('max_det', 300)
            line_width = self.kwargs.get('line_width', 3)
            show_labels = self.kwargs.get('show_labels', True)
            show_conf = self.kwargs.get('show_conf', True)
            show_boxes = self.kwargs.get('show_boxes', True)
            save_txt = self.kwargs.get('save_txt', True)
            save_conf = self.kwargs.get('save_conf', True)
            save_crop = self.kwargs.get('save_crop', False)
            visualize = self.kwargs.get('visualize', True)
            augment = self.kwargs.get('augment', False)
            agnostic_nms = self.kwargs.get('agnostic_nms', False)
            retina_masks = self.kwargs.get('retina_masks', False)
            output_format = self.kwargs.get('output_format', 'torch')
            verbose = self.kwargs.get('verbose', False)
            show = self.kwargs.get('show', False)
            
            # 記錄高級參數
            self.log_message.emit(f"🎯 高級參數: IoU={iou_threshold:.2f}, 最大檢測={max_det}, 線寬={line_width}")
            self.log_message.emit(f"📊 顯示選項: 標籤={show_labels}, 置信度={show_conf}, 邊框={show_boxes}")
            self.log_message.emit(f"💾 保存選項: 文本={save_txt}, 置信度={save_conf}, 裁剪={save_crop}")
            self.log_message.emit(f"🔧 高級選項: 可視化={visualize}, 增強={augment}, 無關NMS={agnostic_nms}")
            self.log_message.emit(f"📋 輸出格式: {output_format}, 詳細={verbose}, 顯示={show}")
            
            # 根據模式執行推理
            if inference_mode == "Data目錄處理模式":
                self.log_message.emit("🔍 使用Data目錄處理模式...")
                # 執行增強版推理
                results = enhanced_inference(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    predict_data_dir=predict_data_dir,
                    iou_threshold=iou_threshold,
                    max_det=max_det,
                    line_width=line_width,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    show_boxes=show_boxes,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    save_crop=save_crop,
                    visualize=visualize,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    retina_masks=retina_masks,
                    output_format=output_format,
                    verbose=verbose,
                    show=show
                )
                
            elif inference_mode == "數據集測試模式":
                self.log_message.emit("📊 使用數據集測試模式...")
                # 執行增強版推理
                results = enhanced_inference(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    predict_data_dir=predict_data_dir,
                    iou_threshold=iou_threshold,
                    max_det=max_det,
                    line_width=line_width,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    show_boxes=show_boxes,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    save_crop=save_crop,
                    visualize=visualize,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    retina_masks=retina_masks,
                    output_format=output_format,
                    verbose=verbose,
                    show=show
                )
                
            elif inference_mode == "單個文件處理模式":
                self.log_message.emit("📁 使用單個文件處理模式...")
                # 執行增強版推理
                results = enhanced_inference(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    predict_data_dir=predict_data_dir,
                    iou_threshold=iou_threshold,
                    max_det=max_det,
                    line_width=line_width,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    show_boxes=show_boxes,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    save_crop=save_crop,
                    visualize=visualize,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    retina_masks=retina_masks,
                    output_format=output_format,
                    verbose=verbose,
                    show=show
                )
            
            # 處理推理結果
            if 'results' in locals() and results:
                self.log_message.emit(f"✅ 推理完成，處理了 {len(results)} 個結果")
            else:
                self.log_message.emit("⚠️ 推理完成，但未檢測到任何目標")
            
            self.log_message.emit("✅ 推理完成!")
            self.log_message.emit(f"[FOLDER] 結果保存在: Predict/Result/")
            self.progress.emit("推理完成")
            
        except Exception as e:
            self.log_message.emit(f"[ERROR] 推理失敗: {e}")
            self.progress.emit("推理失敗")
            raise e
    
   
            
        except Exception as e:
            self.log_message.emit(f"[ERROR] 推理測試失敗: {e}")
            self.progress.emit("推理測試失敗")
            raise e

    def _run_custom_inference(self, model_path, confidence_threshold):
        """運行自定義推理 - 基於修正後的yolo_inference.py"""
        try:
            self.log_message.emit("🎯 開始自定義推理處理...")
            
            # 使用修正後的推理模組
            try:
                from Code.yolo_inference import main as inference_main
                self.log_message.emit("✅ 成功載入修正後的推理模組")
            except ImportError as e:
                self.log_message.emit(f"❌ 無法導入yolo_inference模組: {e}")
                return
            
            # 設置推理參數
            predict_data_dir = "Predict/Data"
            if not os.path.exists(predict_data_dir):
                os.makedirs(predict_data_dir, exist_ok=True)
                self.log_message.emit(f"📁 創建預測數據目錄: {predict_data_dir}")
            
            # 檢查預測數據目錄中是否有文件
            import os
            image_files = []
            for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
                if os.path.exists(predict_data_dir):
                    image_files.extend([f for f in os.listdir(predict_data_dir) if f.lower().endswith(ext)])
            
            if not image_files:
                self.log_message.emit(f"⚠️ 預測數據目錄 {predict_data_dir} 中未找到圖片文件")
                self.log_message.emit("💡 請將圖片文件（.npy, .jpg, .png等）放入 Predict/Data 目錄")
                return
            
            self.log_message.emit(f"📊 找到 {len(image_files)} 個圖片文件")
            
            # 執行推理
            self.log_message.emit("🚀 開始執行推理...")
            results = inference_main(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=None,  # 自動檢測設備
                predict_data_dir=predict_data_dir
            )
            
            if results:
                self.log_message.emit(f"✅ 自定義推理完成，處理了 {len(results)} 個結果")
                self.log_message.emit("📁 結果保存在: Predict/Result 目錄")
            else:
                self.log_message.emit("⚠️ 自定義推理完成，但未檢測到任何目標")
                
        except Exception as e:
            self.log_message.emit(f"❌ 自定義推理失敗: {e}")
            raise e
    
    def _get_unique_training_folder_name(self, base_name):
        """生成唯一的訓練資料夾名稱，如果重複則添加序號"""
        from pathlib import Path
        
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
    
    def _stereo_training(self):
        """立體視覺深度估計訓練"""
        try:
            self.progress.emit("正在準備立體視覺訓練...")
            self.log_message.emit("🚀 開始立體視覺深度估計訓練...")
            self.log_message.emit("🚀 Starting stereo vision depth estimation training...")
            
            # 獲取訓練參數
            dataset_path = self.kwargs.get('dataset_path', '')
            model_name = self.kwargs.get('model_name', 'raftstereo-sceneflow.pth')
            batch_size = self.kwargs.get('batch_size', 6)
            lr = self.kwargs.get('lr', 0.0002)
            num_steps = self.kwargs.get('num_steps', 100000)
            image_size = self.kwargs.get('image_size', [320, 720])
            corr_implementation = self.kwargs.get('corr_implementation', 'reg')
            corr_levels = self.kwargs.get('corr_levels', 4)
            train_iters = self.kwargs.get('train_iters', 16)
            valid_iters = self.kwargs.get('valid_iters', 32)
            mixed_precision = self.kwargs.get('mixed_precision', True)
            shared_backbone = self.kwargs.get('shared_backbone', False)
            train_datasets = self.kwargs.get('train_datasets', ['sceneflow'])
            wdecay = self.kwargs.get('wdecay', 0.00001)
            name = self.kwargs.get('name', 'raft-stereo-custom')
            
            self.log_message.emit(f"📊 訓練參數:")
            self.log_message.emit(f"📊 Training parameters:")
            self.log_message.emit(f"   數據集路徑: {dataset_path}")
            self.log_message.emit(f"   Dataset path: {dataset_path}")
            self.log_message.emit(f"   預訓練模型: {model_name}")
            self.log_message.emit(f"   Pretrained model: {model_name}")
            self.log_message.emit(f"   批次大小: {batch_size}")
            self.log_message.emit(f"   Batch size: {batch_size}")
            self.log_message.emit(f"   學習率: {lr}")
            self.log_message.emit(f"   Learning rate: {lr}")
            self.log_message.emit(f"   訓練步數: {num_steps}")
            self.log_message.emit(f"   Training steps: {num_steps}")
            self.log_message.emit(f"   圖像尺寸: {image_size}")
            self.log_message.emit(f"   Image size: {image_size}")
            
            # 檢查數據集路徑
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"數據集路徑不存在: {dataset_path}")
            
            # 導入並運行立體視覺訓練器
            import sys
            import os
            
            # 添加 Code 目錄到路徑
            code_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Code')
            if code_dir not in sys.path:
                sys.path.append(code_dir)
            
            # 導入訓練器
            import importlib.util
            trainer_path = os.path.join(code_dir, 'raft-stereo_trainer.py')
            spec = importlib.util.spec_from_file_location("raft_stereo_trainer", trainer_path)
            raft_stereo_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(raft_stereo_trainer)
            RAFTStereoTrainer = raft_stereo_trainer.RAFTStereoTrainer
            create_args = raft_stereo_trainer.create_args
            
            # 創建參數對象
            import argparse
            args = argparse.Namespace()
            args.name = name
            args.restore_ckpt = None  # 可以根據需要設置預訓練模型路徑
            args.mixed_precision = mixed_precision
            args.batch_size = batch_size
            args.train_datasets = train_datasets
            args.lr = lr
            args.num_steps = num_steps
            args.image_size = image_size
            args.train_iters = train_iters
            args.wdecay = wdecay
            args.valid_iters = valid_iters
            args.corr_implementation = corr_implementation
            args.shared_backbone = shared_backbone
            args.corr_levels = corr_levels
            args.corr_radius = 4
            args.n_downsample = 2
            args.context_norm = 'batch'
            args.slow_fast_gru = False
            args.n_gru_layers = 3
            args.hidden_dims = [128]*3
            args.img_gamma = None
            args.saturation_range = None
            args.do_flip = False
            args.spatial_scale = [0, 0]
            args.noyjitter = False
            args.validation_frequency = 10000
            
            # 創建訓練器
            trainer = RAFTStereoTrainer(args)
            
            # 設置停止標誌檢查
            self._current_trainer = trainer
            
            # 開始訓練
            self.progress.emit("正在進行立體視覺深度估計訓練...")
            self.log_message.emit("🔄 開始模型訓練...")
            self.log_message.emit("🔄 Starting model training...")
            
            model_path = trainer.train()
            
            if self._stop_requested:
                self.log_message.emit("⏹️ 訓練被用戶停止")
                self.log_message.emit("⏹️ Training stopped by user")
                return
            
            self.log_message.emit("✅ 立體視覺訓練完成!")
            self.log_message.emit("✅ Stereo vision training completed!")
            self.log_message.emit(f"💾 模型保存至: {model_path}")
            self.log_message.emit(f"💾 Model saved to: {model_path}")
            self.progress.emit("立體視覺訓練完成")
            
        except Exception as e:
            self.log_message.emit(f"❌ 立體視覺訓練失敗: {e}")
            self.log_message.emit(f"❌ Stereo vision training failed: {e}")
            self.progress.emit("立體視覺訓練失敗")
            raise e


class YOLOLauncherGUI(QMainWindow):
    """YOLO 統一啟動器圖形界面"""
    
    def _initialize_trainers(self):
        """初始化训练器（占位方法）"""
        # 功能检查已经在其他地方完成，这里保持为空以避免重复执行
        pass
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 統一啟動器 - 整合版")
        
        # 初始化變量
        self.worker_thread = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_name = self._get_gpu_name()
        
        # 配置文件路徑
        self.settings_file = Path(__file__).parent / 'config' / 'gui_settings.yaml'
        
        # 執行一次功能檢查（避免重複執行）
        self._initialize_trainers()
        
        # 設置樣式
        self.setup_style()
        
        # 設置用戶界面
        self.setup_ui()
        
        # 先嘗試加載保存的設置（包括窗口位置）
        self.load_settings()
        
        # 如果沒有保存的窗口位置，才設置默認窗口大小
        if not hasattr(self, '_window_geometry_loaded') or not self._window_geometry_loaded:
            # 只設置窗口大小，不設置位置，讓系統決定位置
            self.log_message("⚠️ 沒有保存的窗口位置，使用默認大小")
            self.resize(1200, 800)  # 設置一個合理的默認大小
        else:
            self.log_message("✅ 使用保存的窗口位置")
            # 確保窗口位置正確設置
            current_geometry = self.geometry()
            self.log_message(f"📍 當前窗口位置: ({current_geometry.x()}, {current_geometry.y()}) 大小: {current_geometry.width()}x{current_geometry.height()}")
        
        # 自動載入資料集和模型列表（啟動時自動執行）
        self.log_message("🔄 啟動時自動載入資料集和模型...")
        
        # 先載入資料集列表
        self.auto_find_train_dataset()
        
        # 再載入模型列表
        self.refresh_model_list()
        
        # 最後重新恢復上次使用的選擇（因為列表已更新）
        self.log_message("🔄 恢復上次使用的選擇...")
        try:
            import yaml
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = yaml.safe_load(f)
                if settings and 'standard_training' in settings:
                    self._restore_last_used_selections(settings['standard_training'])
        except Exception as e:
            self.log_message(f"[WARNING] 恢復上次選擇失敗: {e}")
        
        self.log_message("[OK] 啟動時自動載入完成")
    
    def _setup_window_geometry(self):
        """设置默认窗口大小和位置（仅在首次启动时使用）"""
        try:
            # 获取主屏幕的可用区域（排除任务栏）
            screen = QApplication.primaryScreen()
            available_geometry = screen.availableGeometry()
            
            # 获取屏幕尺寸
            screen_width = available_geometry.width()
            screen_height = available_geometry.height()
            
            # 设置窗口大小：宽度600，高度为屏幕高度减去200
            window_width = 600
            window_height = max(400, screen_height - 200)  # 最小高度400，最大为屏幕高度-200
            
            # 添加日誌信息
            print(f"螢幕高度: {screen_height}px, 窗口高度: {window_height}px")
            
            # 计算居中位置
            x = available_geometry.x() + (screen_width - window_width) // 2
            y = available_geometry.y() + 20  # 从顶部往下20像素，更接近保存的位置
            
            # 最终边界检查
            x = max(available_geometry.x() + 10, x)
            y = max(available_geometry.y() + 10, y)
            x = min(x, available_geometry.x() + screen_width - window_width - 10)
            y = min(y, available_geometry.y() + screen_height - window_height - 10)
            
            # 设置窗口几何形状
            self.setGeometry(x, y, window_width, window_height)
            
        except Exception as e:
            # 如果出错，使用默认设置：宽度600，高度为屏幕高度减去200
            window_width = 600
            try:
                screen_height = QApplication.primaryScreen().availableGeometry().height()
                window_height = max(400, screen_height - 200)
            except:
                window_height = 400  # 如果无法获取屏幕高度，使用默认400
            
            self.resize(window_width, window_height)
            
            # 使用Qt的居中方法
            frame_geometry = self.frameGeometry()
            screen_center = QApplication.primaryScreen().availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            self.move(frame_geometry.topLeft())
    
    def _get_gpu_name(self):
        """獲取GPU名稱"""
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                # 簡化GPU名稱，只保留主要型號
                if 'RTX' in gpu_name:
                    # 提取RTX型號，如 "NVIDIA GeForce RTX 5070 Ti" -> "RTX 5070 Ti"
                    import re
                    match = re.search(r'RTX\s+\d+\w*(?:\s+\w+)?', gpu_name)
                    if match:
                        return match.group()
                elif 'GTX' in gpu_name:
                    # 提取GTX型號
                    import re
                    match = re.search(r'GTX\s+\d+\w*(?:\s+\w+)?', gpu_name)
                    if match:
                        return match.group()
                elif 'Tesla' in gpu_name:
                    # 提取Tesla型號
                    import re
                    match = re.search(r'Tesla\s+\w+', gpu_name)
                    if match:
                        return match.group()
                else:
                    # 其他情況，返回完整名稱
                    return gpu_name
            else:
                return "CPU"
        except Exception as e:
            # 靜默處理GPU名稱獲取錯誤
            return "Unknown"
    
    def setup_ui(self):
        """設置用戶界面"""
        # 創建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 創建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 創建標籤頁
        self.create_tabs(main_layout)
        
        # 創建狀態欄
        self.create_status_bar()
        
        # 初始化架構描述
        self.update_arch_description()
        
        # 自動載入配置（在UI完全創建後）
        self.auto_load_configs()
        
        
        # 自動偵測資料夾數量（如果源路徑已設定）
        if self.convert_source_edit.text():
            self.auto_detect_folders()
        
        # 設置關閉事件處理
        self.setAttribute(Qt.WA_QuitOnClose, True)
    
    def closeEvent(self, event):
        """應用程序關閉事件處理"""
        try:
            # 保存當前設置
            self.save_settings()
            
            # 停止所有正在運行的線程
            if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
                self.log_message("🛑 正在停止工作線程...")
                self.worker_thread.stop()
            
            # 等待線程結束
            if hasattr(self, 'worker_thread') and self.worker_thread:
                self.worker_thread.wait(2000)  # 等待2秒
            
            event.accept()
        except Exception as e:
            # 靜默處理關閉應用程序錯誤
            event.accept()
        
    def setup_style(self):
        """設置界面樣式 - 簡化留白設置"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                margin: 0px;
                padding: 0px;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
                margin: 0px;
                padding: 0px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin: 0px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #0078d4;
            }
            QLineEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 6px;
                font-size: 12px;
                margin: 0px;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
            }
            QComboBox {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 6px;
                font-size: 12px;
                margin: 0px;
            }
            QComboBox:focus {
                border: 2px solid #0078d4;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                margin: 0px;
                padding: 5px;
            }
        """)
        
        
    def create_tabs(self, parent_layout):
        """創建標籤頁"""
        self.tab_widget = QTabWidget()
        
        # 數據轉換標籤頁
        self.convert_tab = self.create_convert_tab()
        self.tab_widget.addTab(self.convert_tab, "🔄 數據轉換")
        
        # 標準訓練標籤頁
        self.train_tab = self.create_train_tab()
        self.tab_widget.addTab(self.train_tab, "🎯 標準訓練")
        
        # 推理標籤頁
        self.inference_tab = self.create_inference_tab()
        self.tab_widget.addTab(self.inference_tab, "🔍 推理處理")
        
        # 模型分析標籤頁
        self.model_analyzer_tab = self.create_model_analyzer_tab()
        self.tab_widget.addTab(self.model_analyzer_tab, "🔬 模型分析")
        
        # 模型修改器標籤頁
        self.model_modifier_tab = self.create_model_modifier_tab()
        self.tab_widget.addTab(self.model_modifier_tab, "🔧 模型修改器")
        
        # 立體視覺標籤頁
        self.stereo_tab = self.create_stereo_tab()
        self.tab_widget.addTab(self.stereo_tab, "👁️ 立體視覺")
        
        # 日誌標籤頁
        self.log_tab = self.create_log_tab()
        self.tab_widget.addTab(self.log_tab, "📋 運行日誌")
        
        parent_layout.addWidget(self.tab_widget)
        
        # 初始化模型分析列表（在所有標籤頁創建完成後）
        self.refresh_analyzer_model_list()
        
    def create_convert_tab(self):
        """創建數據轉換標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 源數據路徑選擇
        source_group = QGroupBox("源數據設置")
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel("Forest數據集路徑:"), 0, 0)
        self.convert_source_edit = QLineEdit()
        self.convert_source_edit.setPlaceholderText("選擇Forest數據集根目錄")
        self.convert_source_edit.setText("D:\\DMD\\Forest")  # 默認路徑
        source_layout.addWidget(self.convert_source_edit, 1, 0)
        
        self.convert_source_btn = QPushButton("瀏覽")
        self.convert_source_btn.clicked.connect(self.browse_convert_source)
        source_layout.addWidget(self.convert_source_btn, 1, 1)
        
        layout.addWidget(source_group)
        
        # 輸出設置
        output_group = QGroupBox("輸出設置")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("輸出路徑 (可選):"), 0, 0)
        self.convert_output_edit = QLineEdit()
        self.convert_output_edit.setPlaceholderText("留空則使用默認路徑 (dataset_時間戳)")
        output_layout.addWidget(self.convert_output_edit, 1, 0)
        
        self.convert_output_btn = QPushButton("瀏覽")
        self.convert_output_btn.clicked.connect(self.browse_convert_output)
        output_layout.addWidget(self.convert_output_btn, 1, 1)
        
        layout.addWidget(output_group)
        
        # 資料夾數量選擇
        folder_count_group = QGroupBox("資料夾數量選擇")
        folder_count_layout = QGridLayout(folder_count_group)
        
        folder_count_layout.addWidget(QLabel("處理資料夾數量:"), 0, 0)
        self.folder_count_spin = QSpinBox()
        self.folder_count_spin.setRange(1, 1000)
        self.folder_count_spin.setValue(1)  # 預設為1，避免0的混淆
        self.folder_count_spin.setSuffix(" 個資料夾")
        self.folder_count_spin.setToolTip("設定要處理的資料夾數量，將按順序處理前N個資料夾")
        folder_count_layout.addWidget(self.folder_count_spin, 0, 1)
        
        self.auto_detect_folders_btn = QPushButton("🔍 偵測資料夾")
        self.auto_detect_folders_btn.clicked.connect(self.auto_detect_folders)
        folder_count_layout.addWidget(self.auto_detect_folders_btn, 0, 2)
        
        # 資料夾狀態標籤
        self.folder_status_label = QLabel("")
        self.folder_status_label.setStyleSheet("color: #666666; font-size: 11px;")
        folder_count_layout.addWidget(self.folder_status_label, 1, 0, 1, 3)
        
        layout.addWidget(folder_count_group)
        
        # 深度圖選項
        depth_group = QGroupBox("深度圖選項")
        depth_layout = QVBoxLayout(depth_group)
        
        # 深度圖選項按鈕組
        self.depth_button_group = QButtonGroup()
        
        self.use_depth_radio = QRadioButton("使用深度圖 (4通道數據)")
        self.use_depth_radio.setChecked(True)  # 默認選中
        self.use_depth_radio.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.depth_button_group.addButton(self.use_depth_radio, 0)
        depth_layout.addWidget(self.use_depth_radio)
        
        self.no_depth_radio = QRadioButton("不使用深度圖 (3通道RGB數據)")
        self.no_depth_radio.setStyleSheet("color: #28a745; font-weight: bold;")
        self.depth_button_group.addButton(self.no_depth_radio, 1)
        depth_layout.addWidget(self.no_depth_radio)
        
        self.stereo_radio = QRadioButton("立體視覺數據 (RGB左右視圖+視差圖)") 
        self.stereo_radio.setStyleSheet("color: #ff6b35; font-weight: bold;")
        self.depth_button_group.addButton(self.stereo_radio, 2)
        depth_layout.addWidget(self.stereo_radio)
        
        # 深度圖說明
        depth_info = QLabel("""
        • 使用深度圖：合併RGB圖像和深度圖為4通道NumPy文件
        • 不使用深度圖：直接複製RGB圖像為標準3通道文件
        • 立體視覺數據：RGB左右視圖+視差圖
        • 選擇會影響生成的數據集類型和後續訓練方式
        """)
        depth_info.setStyleSheet("color: #666666; font-size: 11px; margin: 10px;")
        depth_info.setWordWrap(True)
        depth_layout.addWidget(depth_info)
        
        layout.addWidget(depth_group)
        
        # 轉換說明
        info_group = QGroupBox("轉換說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        數據轉換功能說明：

        1. 支持兩種模式：
        • 4通道模式：合併RGB圖像和深度圖為4通道NumPy文件
        • 3通道模式：直接複製RGB圖像為標準3通道文件

        2. 自動分割為訓練集(80%)、驗證集(15%)、測試集(5%)
        3. 生成YOLO格式的標籤文件
        4. 根據predefined_classes.txt，創建data_config.yaml配置文件
        5. 支持自定義輸出路徑

        數據結構要求：
        - Forest_Video_*/Img/Img0_*.png (圖像文件)
        - Forest_Video_*/Img/DepthGT_*.pfm (深度圖文件，4通道模式需要)
        - Forest_Video_*/YOLO_Label/*.txt (標籤文件)
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 轉換控制
        control_group = QGroupBox("轉換控制")
        control_layout = QHBoxLayout(control_group)
        
        self.convert_start_btn = QPushButton("🔄 開始轉換")
        self.convert_start_btn.clicked.connect(self.start_convert)
        control_layout.addWidget(self.convert_start_btn)
        
        self.convert_stop_btn = QPushButton("⏹️ 停止轉換")
        self.convert_stop_btn.clicked.connect(self.stop_convert)
        self.convert_stop_btn.setEnabled(False)
        control_layout.addWidget(self.convert_stop_btn)
        
        layout.addWidget(control_group)
        
        return tab
    
    def create_train_tab(self):
        """創建標準訓練標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 資料集選擇
        dataset_group = QGroupBox("資料集設置")
        dataset_layout = QGridLayout(dataset_group)
        
        dataset_layout.addWidget(QLabel("選擇資料集:"), 0, 0)
        self.train_dataset_combo = QComboBox()
        self.train_dataset_combo.setPlaceholderText("點擊「自動尋找」或「瀏覽」選擇資料集")
        self.train_dataset_combo.setEditable(True)  # 允許手動輸入
        self.train_dataset_combo.setMinimumWidth(300)
        self.train_dataset_combo.currentTextChanged.connect(self.update_train_dataset_info)
        dataset_layout.addWidget(self.train_dataset_combo, 1, 0)
        
        self.train_dataset_btn = QPushButton("瀏覽")
        self.train_dataset_btn.clicked.connect(self.browse_train_dataset)
        dataset_layout.addWidget(self.train_dataset_btn, 1, 1)
        
        self.auto_find_train_dataset_btn = QPushButton("🔍 自動尋找")
        self.auto_find_train_dataset_btn.clicked.connect(self.auto_find_train_dataset)
        self.auto_find_train_dataset_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        dataset_layout.addWidget(self.auto_find_train_dataset_btn, 1, 2)
        
        # 資料集狀態標籤
        self.train_dataset_status = QLabel("")
        self.train_dataset_status.setStyleSheet("color: #666666; font-size: 11px;")
        dataset_layout.addWidget(self.train_dataset_status, 2, 0, 1, 3)
        
        # 上次使用信息
        self.last_used_info = QLabel("")
        self.last_used_info.setStyleSheet("color: #007bff; font-size: 10px; font-style: italic; padding: 2px;")
        dataset_layout.addWidget(self.last_used_info, 3, 0, 1, 3)
        
        layout.addWidget(dataset_group)
        
        # 訓練參數 - 優化三列布局
        params_group = QGroupBox("訓練參數")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        params_layout = QGridLayout(params_group)
        params_layout.setSpacing(8)
        
        # 第一列 - 基本訓練參數
        params_layout.addWidget(QLabel("訓練輪數:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.epochs_spin, 0, 1)
        
        params_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.batch_size_spin, 1, 1)
        
        params_layout.addWidget(QLabel("學習率:"), 2, 0)
        self.learning_rate_spin = QSpinBox()
        self.learning_rate_spin.setRange(1, 1000)
        self.learning_rate_spin.setValue(1)
        self.learning_rate_spin.setSuffix(" (×0.001)")
        self.learning_rate_spin.setToolTip("學習率 = 設定值 × 0.001\n例如: 1 = 0.001, 10 = 0.01")
        self.learning_rate_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        params_layout.addWidget(QLabel("圖像大小:"), 3, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setToolTip("訓練時的圖像大小")
        self.imgsz_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.imgsz_spin, 3, 1)
        
        # 檢查點保存週期設置
        params_layout.addWidget(QLabel("檢查點週期:"), 4, 0)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(1, 100)
        self.save_period_spin.setValue(10)
        self.save_period_spin.setSuffix(" epochs")
        self.save_period_spin.setToolTip("每N個epoch保存一次檢查點，-1表示不保存")
        self.save_period_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.save_period_spin, 4, 1)
        
        # 第二列 - 數據增強參數
        params_layout.addWidget(QLabel("縮放比例:"), 0, 2)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(0, 100)
        self.scale_spin.setValue(0)
        self.scale_spin.setSuffix(" (×0.01)")
        self.scale_spin.setToolTip("模型縮放比例，0 = 0.0 (無縮放)")
        self.scale_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.scale_spin, 0, 3)
        
        params_layout.addWidget(QLabel("Mosaic:"), 1, 2)
        self.mosaic_spin = QSpinBox()
        self.mosaic_spin.setRange(0, 100)
        self.mosaic_spin.setValue(0)
        self.mosaic_spin.setSuffix(" (×0.01)")
        self.mosaic_spin.setToolTip("Mosaic數據增強強度，100 = 1.0")
        self.mosaic_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.mosaic_spin, 1, 3)
        
        params_layout.addWidget(QLabel("Mixup:"), 2, 2)
        self.mixup_spin = QSpinBox()
        self.mixup_spin.setRange(0, 100)
        self.mixup_spin.setValue(0)
        self.mixup_spin.setSuffix(" (×0.01)")
        self.mixup_spin.setToolTip("Mixup數據增強強度，0 = 0.0")
        self.mixup_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.mixup_spin, 2, 3)
        
        params_layout.addWidget(QLabel("Copy-paste:"), 3, 2)
        self.copy_paste_spin = QSpinBox()
        self.copy_paste_spin.setRange(0, 100)
        self.copy_paste_spin.setValue(0)
        self.copy_paste_spin.setSuffix(" (×0.01)")
        self.copy_paste_spin.setToolTip("Copy-paste數據增強強度，10 = 0.1")
        self.copy_paste_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.copy_paste_spin, 3, 3)
        
        # 第三列 - 圖像處理參數
        params_layout.addWidget(QLabel("圖像尺寸:"), 0, 4)
        self.image_size_label = QLabel("未檢測到")
        self.image_size_label.setStyleSheet("color: #666666; font-size: 11px; padding: 4px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 3px;")
        params_layout.addWidget(self.image_size_label, 0, 5)
        
        params_layout.addWidget(QLabel("HSV色相:"), 1, 4)
        self.hsv_h_spin = QSpinBox()
        self.hsv_h_spin.setRange(0, 100)
        self.hsv_h_spin.setValue(0)
        self.hsv_h_spin.setSuffix(" (×0.01)")
        self.hsv_h_spin.setToolTip("HSV色相增強參數，0 = 0.0")
        self.hsv_h_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.hsv_h_spin, 1, 5)
        
        params_layout.addWidget(QLabel("HSV飽和度:"), 2, 4)
        self.hsv_s_spin = QSpinBox()
        self.hsv_s_spin.setRange(0, 100)
        self.hsv_s_spin.setValue(0)
        self.hsv_s_spin.setSuffix(" (×0.01)")
        self.hsv_s_spin.setToolTip("HSV飽和度增強參數，0 = 0.0")
        self.hsv_s_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.hsv_s_spin, 2, 5)
        
        params_layout.addWidget(QLabel("HSV明度:"), 3, 4)
        self.hsv_v_spin = QSpinBox()
        self.hsv_v_spin.setRange(0, 100)
        self.hsv_v_spin.setValue(0)
        self.hsv_v_spin.setSuffix(" (×0.01)")
        self.hsv_v_spin.setToolTip("HSV明度增強參數，0 = 0.0")
        self.hsv_v_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.hsv_v_spin, 3, 5)
        
        params_layout.addWidget(QLabel("BGR通道:"), 4, 4)
        self.bgr_spin = QSpinBox()
        self.bgr_spin.setRange(0, 100)
        self.bgr_spin.setValue(0)
        self.bgr_spin.setSuffix(" (×0.01)")
        self.bgr_spin.setToolTip("BGR通道增強參數，0 = 0.0")
        params_layout.addWidget(self.bgr_spin, 4, 5)
        
        params_layout.addWidget(QLabel("自動增強:"), 5, 4)
        self.auto_augment_combo = QComboBox()
        self.auto_augment_combo.addItem("無", None)
        self.auto_augment_combo.addItem("RandAugment", "randaugment")
        self.auto_augment_combo.addItem("AutoAugment", "autoaugment")
        self.auto_augment_combo.setToolTip("自動增強策略")
        params_layout.addWidget(self.auto_augment_combo, 5, 5)
        
        # 第四列 - 幾何變換參數
        params_layout.addWidget(QLabel("旋轉角度:"), 6, 0)
        self.degrees_spin = QSpinBox()
        self.degrees_spin.setRange(0, 180)
        self.degrees_spin.setValue(0)
        self.degrees_spin.setSuffix("°")
        self.degrees_spin.setToolTip("圖像旋轉角度，0 = 不旋轉")
        self.degrees_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.degrees_spin, 6, 1)
        
        params_layout.addWidget(QLabel("平移距離:"), 7, 0)
        self.translate_spin = QSpinBox()
        self.translate_spin.setRange(0, 100)
        self.translate_spin.setValue(0)
        self.translate_spin.setSuffix(" (×0.01)")
        self.translate_spin.setToolTip("圖像平移距離，0 = 不平移")
        self.translate_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.translate_spin, 7, 1)
        
        params_layout.addWidget(QLabel("剪切角度:"), 8, 0)
        self.shear_spin = QSpinBox()
        self.shear_spin.setRange(0, 100)
        self.shear_spin.setValue(0)
        self.shear_spin.setSuffix(" (×0.01)")
        self.shear_spin.setToolTip("圖像剪切角度，0 = 不剪切")
        self.shear_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.shear_spin, 8, 1)
        
        params_layout.addWidget(QLabel("透視變換:"), 9, 0)
        self.perspective_spin = QSpinBox()
        self.perspective_spin.setRange(0, 100)
        self.perspective_spin.setValue(0)
        self.perspective_spin.setSuffix(" (×0.01)")
        self.perspective_spin.setToolTip("透視變換強度，0 = 不變換")
        self.perspective_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.perspective_spin, 9, 1)
        
        # 第五列 - 翻轉和裁剪參數
        params_layout.addWidget(QLabel("上下翻轉:"), 6, 2)
        self.flipud_spin = QSpinBox()
        self.flipud_spin.setRange(0, 100)
        self.flipud_spin.setValue(0)
        self.flipud_spin.setSuffix(" (×0.01)")
        self.flipud_spin.setToolTip("上下翻轉概率，0 = 不翻轉")
        self.flipud_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.flipud_spin, 6, 3)
        
        params_layout.addWidget(QLabel("左右翻轉:"), 7, 2)
        self.fliplr_spin = QSpinBox()
        self.fliplr_spin.setRange(0, 100)
        self.fliplr_spin.setValue(0)
        self.fliplr_spin.setSuffix(" (×0.01)")
        self.fliplr_spin.setToolTip("左右翻轉概率，0 = 不翻轉")
        self.fliplr_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.fliplr_spin, 7, 3)
        
        params_layout.addWidget(QLabel("隨機擦除:"), 8, 2)
        self.erasing_spin = QSpinBox()
        self.erasing_spin.setRange(0, 100)
        self.erasing_spin.setValue(0)
        self.erasing_spin.setSuffix(" (×0.01)")
        self.erasing_spin.setToolTip("隨機擦除概率，0 = 不擦除")
        self.erasing_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.erasing_spin, 8, 3)
        
        params_layout.addWidget(QLabel("裁剪比例:"), 9, 2)
        self.crop_fraction_spin = QSpinBox()
        self.crop_fraction_spin.setRange(0, 100)
        self.crop_fraction_spin.setValue(0)
        self.crop_fraction_spin.setSuffix(" (×0.01)")
        self.crop_fraction_spin.setToolTip("裁剪比例，0 = 不裁剪")
        self.crop_fraction_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.crop_fraction_spin, 9, 3)
        
        # 第六列 - 訓練控制參數
        params_layout.addWidget(QLabel("關閉Mosaic:"), 6, 4)
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 50)
        self.close_mosaic_spin.setValue(10)
        self.close_mosaic_spin.setToolTip("最後N個epoch關閉Mosaic增強")
        self.close_mosaic_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.close_mosaic_spin, 6, 5)
        
        params_layout.addWidget(QLabel("工作進程:"), 7, 4)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(0)
        self.workers_spin.setToolTip("數據加載工作進程數，0 = 自動")
        self.workers_spin.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.workers_spin, 7, 5)
        
        params_layout.addWidget(QLabel("優化器:"), 8, 4)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW", "RMSProp"])
        self.optimizer_combo.setCurrentText("SGD")
        self.optimizer_combo.setToolTip("優化器類型")
        self.optimizer_combo.setStyleSheet("padding: 4px; border: 1px solid #ced4da; border-radius: 4px;")
        params_layout.addWidget(self.optimizer_combo, 8, 5)
        
        params_layout.addWidget(QLabel("AMP混合精度:"), 9, 4)
        self.amp_checkbox = QCheckBox("啟用")
        self.amp_checkbox.setChecked(True)
        self.amp_checkbox.setToolTip("自動混合精度訓練")
        self.amp_checkbox.setStyleSheet("padding: 4px;")
        params_layout.addWidget(self.amp_checkbox, 9, 5)
        
        layout.addWidget(params_group)
        
        # 模型選擇與訓練模式 - 整合設計
        model_group = QGroupBox("模型選擇與訓練模式")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        model_layout = QVBoxLayout(model_group)
        
        # 訓練模式選擇區域
        training_mode_frame = QFrame()
        training_mode_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 10px;
            }
        """)
        training_mode_layout = QHBoxLayout(training_mode_frame)
        
        # 預訓練模型選項
        self.pretrained_radio = QRadioButton("使用預訓練模型 (PT)")
        self.pretrained_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6c757d;
                border-radius: 9px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #007bff;
                border-radius: 9px;
                background-color: #007bff;
            }
        """)
        self.pretrained_radio.setChecked(True)  # 默認選擇預訓練模型
        self.pretrained_radio.toggled.connect(self.on_training_mode_changed)
        training_mode_layout.addWidget(self.pretrained_radio)
        
        # 重新訓練選項
        self.retrain_radio = QRadioButton("重新訓練 (YAML)")
        self.retrain_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6c757d;
                border-radius: 9px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #28a745;
                border-radius: 9px;
                background-color: #28a745;
            }
        """)
        self.retrain_radio.toggled.connect(self.on_training_mode_changed)
        training_mode_layout.addWidget(self.retrain_radio)
        
        training_mode_layout.addStretch()
        model_layout.addWidget(training_mode_frame)
        
        # 當前模式狀態指示器
        self.current_mode_label = QLabel("當前模式：預訓練模型 (PT)")
        self.current_mode_label.setStyleSheet("""
            QLabel {
                color: #007bff;
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 4px;
                margin-bottom: 5px;
            }
        """)
        model_layout.addWidget(self.current_mode_label)
        
        # 模式說明
        mode_info_label = QLabel("💡 預訓練模型：使用已訓練的權重進行微調 | 重新訓練：從頭開始訓練新模型")
        mode_info_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 12px;
                font-style: italic;
                padding: 8px;
                background-color: #e9ecef;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        """)
        mode_info_label.setWordWrap(True)
        model_layout.addWidget(mode_info_label)
        
        # 合併的模型選擇區域 - 緊湊布局
        model_selection_frame = QFrame()
        model_selection_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        model_selection_layout = QGridLayout(model_selection_frame)
        model_selection_layout.setSpacing(8)
        
        # 第一行：模型文件選擇
        model_selection_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.model_file_combo = QComboBox()
        self.model_file_combo.setPlaceholderText("选择模型文件")
        self.model_file_combo.setMinimumWidth(250)
        self.model_file_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
        """)
        model_selection_layout.addWidget(self.model_file_combo, 0, 1)
        
        self.refresh_model_btn = QPushButton("🔄 刷新")
        self.refresh_model_btn.clicked.connect(self.smart_refresh_model_list)
        self.refresh_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        model_selection_layout.addWidget(self.refresh_model_btn, 0, 2)
        
        # 第二行：模型大小選擇（僅在YAML類型時顯示）
        model_selection_layout.addWidget(QLabel("模型大小:"), 1, 0)
        self.train_model_size_combo = QComboBox()
        self.train_model_size_combo.setPlaceholderText("选择模型大小")
        self.train_model_size_combo.addItems(["n (nano)", "s (small)", "m (medium)", "l (large)", "x (xlarge)"])
        self.train_model_size_combo.setVisible(False)  # 初始隐藏
        self.train_model_size_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                min-width: 200px;
            }
        """)
        self.train_model_size_combo.currentTextChanged.connect(self.on_train_model_size_changed)
        self.train_model_size_combo.currentIndexChanged.connect(self.on_train_model_size_changed)
        model_selection_layout.addWidget(self.train_model_size_combo, 1, 1)
        
        # 添加狀態標籤
        self.model_selection_status = QLabel("請選擇模型文件")
        self.model_selection_status.setStyleSheet("color: #666666; font-size: 11px; font-style: italic;")
        model_selection_layout.addWidget(self.model_selection_status, 1, 2)
        
        model_layout.addWidget(model_selection_frame)
        
        # 簡化的模式狀態顯示
        self.train_model_status = QLabel("預訓練模式：將使用PT模型文件進行微調訓練")
        self.train_model_status.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                margin: 5px 0;
            }
        """)
        self.train_model_status.setWordWrap(True)
        model_layout.addWidget(self.train_model_status)
        
        layout.addWidget(model_group)
        
        # 訓練控制
        control_group = QGroupBox("訓練控制")
        control_layout = QHBoxLayout(control_group)
        
        
        self.train_start_btn = QPushButton("🚀 開始訓練")
        self.train_start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_start_btn)
        
        self.train_stop_btn = QPushButton("⏹️ 停止訓練")
        self.train_stop_btn.clicked.connect(self.stop_training)
        self.train_stop_btn.setEnabled(False)
        control_layout.addWidget(self.train_stop_btn)
        
        layout.addWidget(control_group)
        
        return tab
        
    def create_inference_tab(self):
        """創建推理標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型選擇
        model_group = QGroupBox("模型設置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.inference_model_edit = QLineEdit()
        self.inference_model_edit.setPlaceholderText("選擇模型文件 (.pt)")
        self.inference_model_edit.setText("Model_file/PT_File/yolo12n_RGBD.pt")  # 默認模型
        model_layout.addWidget(self.inference_model_edit, 1, 0)
        
        self.inference_model_btn = QPushButton("瀏覽")
        self.inference_model_btn.clicked.connect(self.browse_inference_model)
        model_layout.addWidget(self.inference_model_btn, 1, 1)
        
        # 置信度閾值設定
        model_layout.addWidget(QLabel("置信度閾值:"), 2, 0)
        self.inference_confidence_spin = QSpinBox()
        self.inference_confidence_spin.setRange(1, 99)
        self.inference_confidence_spin.setValue(25)  # 默認0.25
        self.inference_confidence_spin.setSuffix(" (×0.01)")
        self.inference_confidence_spin.setToolTip("置信度閾值 = 設定值 × 0.01\n例如: 25 = 0.25, 50 = 0.5")
        model_layout.addWidget(self.inference_confidence_spin, 2, 1)
        
        # 注意：架構類型已移除，因為.pt文件已包含完整架構
        
        # 類別數量設定
        model_layout.addWidget(QLabel("類別數量:"), 3, 0)
        self.inference_num_classes_spin = QSpinBox()
        self.inference_num_classes_spin.setRange(1, 100)
        self.inference_num_classes_spin.setValue(1)
        self.inference_num_classes_spin.setToolTip("設定檢測的類別數量")
        model_layout.addWidget(self.inference_num_classes_spin, 3, 1)
        
        layout.addWidget(model_group)
        
        # 高級推理參數
        advanced_group = QGroupBox("高級推理參數")
        advanced_layout = QGridLayout(advanced_group)
        
        # IoU閾值
        advanced_layout.addWidget(QLabel("IoU閾值:"), 0, 0)
        self.inference_iou_spin = QSpinBox()
        self.inference_iou_spin.setRange(1, 99)
        self.inference_iou_spin.setValue(45)  # 默認0.45
        self.inference_iou_spin.setSuffix(" (×0.01)")
        self.inference_iou_spin.setToolTip("NMS IoU閾值 = 設定值 × 0.01\n例如: 45 = 0.45, 50 = 0.5")
        advanced_layout.addWidget(self.inference_iou_spin, 0, 1)
        
        # 最大檢測數量
        advanced_layout.addWidget(QLabel("最大檢測數量:"), 0, 2)
        self.inference_max_det_spin = QSpinBox()
        self.inference_max_det_spin.setRange(1, 1000)
        self.inference_max_det_spin.setValue(300)
        self.inference_max_det_spin.setToolTip("每張圖片最大檢測目標數量")
        advanced_layout.addWidget(self.inference_max_det_spin, 0, 3)
        
        # 邊框線寬
        advanced_layout.addWidget(QLabel("邊框線寬:"), 1, 0)
        self.inference_line_width_spin = QSpinBox()
        self.inference_line_width_spin.setRange(1, 10)
        self.inference_line_width_spin.setValue(3)
        self.inference_line_width_spin.setToolTip("檢測框邊框線寬度")
        advanced_layout.addWidget(self.inference_line_width_spin, 1, 1)
        
        # 顯示選項
        self.inference_show_labels_check = QCheckBox("顯示標籤")
        self.inference_show_labels_check.setChecked(True)
        self.inference_show_labels_check.setToolTip("在檢測框上顯示類別標籤")
        advanced_layout.addWidget(self.inference_show_labels_check, 1, 2)
        
        self.inference_show_conf_check = QCheckBox("顯示置信度")
        self.inference_show_conf_check.setChecked(True)
        self.inference_show_conf_check.setToolTip("在檢測框上顯示置信度數值")
        advanced_layout.addWidget(self.inference_show_conf_check, 1, 3)
        
        self.inference_show_boxes_check = QCheckBox("顯示邊框")
        self.inference_show_boxes_check.setChecked(True)
        self.inference_show_boxes_check.setToolTip("顯示檢測邊框")
        advanced_layout.addWidget(self.inference_show_boxes_check, 2, 0)
        
        # 保存選項
        self.inference_save_txt_check = QCheckBox("保存文本結果")
        self.inference_save_txt_check.setChecked(True)
        self.inference_save_txt_check.setToolTip("保存檢測結果為文本文件")
        advanced_layout.addWidget(self.inference_save_txt_check, 2, 1)
        
        self.inference_save_conf_check = QCheckBox("保存置信度")
        self.inference_save_conf_check.setChecked(True)
        self.inference_save_conf_check.setToolTip("在文本結果中保存置信度")
        advanced_layout.addWidget(self.inference_save_conf_check, 2, 2)
        
        self.inference_save_crop_check = QCheckBox("保存裁剪")
        self.inference_save_crop_check.setChecked(False)
        self.inference_save_crop_check.setToolTip("保存檢測到的目標裁剪圖片")
        advanced_layout.addWidget(self.inference_save_crop_check, 2, 3)
        
        # 高級選項
        self.inference_visualize_check = QCheckBox("啟用可視化")
        self.inference_visualize_check.setChecked(True)
        self.inference_visualize_check.setToolTip("啟用特徵可視化功能")
        advanced_layout.addWidget(self.inference_visualize_check, 3, 0)
        
        self.inference_augment_check = QCheckBox("數據增強")
        self.inference_augment_check.setChecked(False)
        self.inference_augment_check.setToolTip("推理時使用數據增強")
        advanced_layout.addWidget(self.inference_augment_check, 3, 1)
        
        self.inference_agnostic_nms_check = QCheckBox("類別無關NMS")
        self.inference_agnostic_nms_check.setChecked(False)
        self.inference_agnostic_nms_check.setToolTip("使用類別無關的NMS")
        advanced_layout.addWidget(self.inference_agnostic_nms_check, 3, 2)
        
        self.inference_retina_masks_check = QCheckBox("視網膜遮罩")
        self.inference_retina_masks_check.setChecked(False)
        self.inference_retina_masks_check.setToolTip("使用視網膜遮罩（僅用於分割任務）")
        advanced_layout.addWidget(self.inference_retina_masks_check, 3, 3)
        
        # 輸出格式
        advanced_layout.addWidget(QLabel("輸出格式:"), 4, 0)
        self.inference_format_combo = QComboBox()
        self.inference_format_combo.addItems(["torch", "numpy", "pandas"])
        self.inference_format_combo.setCurrentText("torch")
        self.inference_format_combo.setToolTip("推理結果的輸出格式")
        advanced_layout.addWidget(self.inference_format_combo, 4, 1)
        
        # 詳細輸出
        self.inference_verbose_check = QCheckBox("詳細輸出")
        self.inference_verbose_check.setChecked(False)
        self.inference_verbose_check.setToolTip("顯示詳細的推理過程信息")
        advanced_layout.addWidget(self.inference_verbose_check, 4, 2)
        
        self.inference_show_check = QCheckBox("顯示圖片")
        self.inference_show_check.setChecked(False)
        self.inference_show_check.setToolTip("推理時顯示圖片窗口")
        advanced_layout.addWidget(self.inference_show_check, 4, 3)
        
        layout.addWidget(advanced_group)
        
        # 數據目錄信息
        data_group = QGroupBox("數據目錄信息")
        data_layout = QVBoxLayout(data_group)
        
        self.data_info_label = QLabel("[FOLDER] 輸入目錄: Predict/Data/")
        self.data_info_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        data_layout.addWidget(self.data_info_label)
        
        self.result_info_label = QLabel("[FOLDER] 輸出目錄: Predict/Result/")
        self.result_info_label.setStyleSheet("color: #28a745; font-weight: bold;")
        data_layout.addWidget(self.result_info_label)
        
        # 檢查數據目錄按鈕
        self.check_data_btn = QPushButton("檢查Data目錄")
        self.check_data_btn.clicked.connect(self.check_data_directory)
        data_layout.addWidget(self.check_data_btn)
        
        layout.addWidget(data_group)
        
        # 推理模式選擇
        mode_group = QGroupBox("推理模式")
        mode_layout = QVBoxLayout(mode_group)
        
        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItems([
            "Data目錄處理模式",
            "數據集測試模式", 
            "單個文件處理模式"
        ])
        self.inference_mode_combo.setCurrentText("Data目錄處理模式")
        self.inference_mode_combo.setToolTip("選擇推理處理模式")
        mode_layout.addWidget(self.inference_mode_combo)
        
        # 數據集選擇（僅在數據集測試模式下顯示）
        self.dataset_group = QGroupBox("數據集設置")
        self.dataset_layout = QVBoxLayout(self.dataset_group)
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("選擇數據集目錄（可選，留空自動查找最新數據集）")
        self.dataset_layout.addWidget(self.dataset_path_edit)
        
        self.dataset_browse_btn = QPushButton("瀏覽數據集")
        self.dataset_browse_btn.clicked.connect(self.browse_inference_dataset)
        self.dataset_layout.addWidget(self.dataset_browse_btn)
        
        # 初始隱藏數據集設置
        self.dataset_group.setVisible(False)
        
        # 連接模式變化信號
        self.inference_mode_combo.currentTextChanged.connect(self.on_inference_mode_changed)
        
        layout.addWidget(mode_group)
        layout.addWidget(self.dataset_group)
        
        # 推理說明
        info_group = QGroupBox("推理說明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setPlainText("""
        YOLO推理器功能說明：

        1. 支持的文件類型：
        - 圖片：JPG, PNG, BMP, TIFF
        - NPY：4通道NumPy文件 (支持單個和批量)
        - GIF：動態圖片
        - 影片：MP4, AVI, MOV, MKV, WMV, FLV

        2. 推理模式：
        - Data目錄處理：處理Predict/Data/目錄中的所有文件
        - 數據集測試：按標準模式處理測試數據集（只讀取，不修改）
        - 單個文件處理：處理指定的單個文件

        3. 處理流程：
        - 從指定目錄讀取文件
        - 使用選定的模型進行推理
        - 結果保存到Predict/Result/目錄

        4. 基本參數設定：
        - 置信度閾值：低於閾值的預測不會繪製邊界框 (默認0.25)
        - IoU閾值：NMS非極大值抑制的IoU閾值 (默認0.45)
        - 最大檢測數量：每張圖片最大檢測目標數量 (默認300)
        - 邊框線寬：檢測框邊框線寬度 (默認3)

        5. 顯示選項：
        - 顯示標籤：在檢測框上顯示類別標籤
        - 顯示置信度：在檢測框上顯示置信度數值
        - 顯示邊框：顯示檢測邊框
        - 顯示圖片：推理時顯示圖片窗口

        6. 保存選項：
        - 保存文本結果：保存檢測結果為文本文件
        - 保存置信度：在文本結果中保存置信度
        - 保存裁剪：保存檢測到的目標裁剪圖片

        7. 高級選項：
        - 啟用可視化：啟用特徵可視化功能
        - 數據增強：推理時使用數據增強
        - 類別無關NMS：使用類別無關的NMS
        - 視網膜遮罩：使用視網膜遮罩（僅用於分割任務）

        8. 輸出格式：
        - torch：PyTorch張量格式
        - numpy：NumPy數組格式
        - pandas：Pandas數據框格式

        9. NPY文件處理：
        - 單個4通道：(H, W, 4) -> 輸出 result_filename.jpg
        - 批量4通道：(N, H, W, 4) -> 輸出 result_filename_batch_000.jpg, ...

        10. 輸出結構：
            - Predict/Result/ - 所有處理結果統一保存
            - 包含可視化、熱力圖、對比圖等

        11. 使用步驟：
            - 選擇推理模式和基本參數
            - 調整高級推理參數（可選）
            - 將要處理的文件放入相應目錄
            - 選擇合適的模型文件
            - 點擊開始推理
        """)
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 推理控制
        control_group = QGroupBox("推理控制")
        control_layout = QHBoxLayout(control_group)
        
        self.inference_start_btn = QPushButton("🔍 開始推理")
        self.inference_start_btn.clicked.connect(self.start_inference)
        control_layout.addWidget(self.inference_start_btn)
        
        self.inference_stop_btn = QPushButton("⏹️ 停止推理")
        self.inference_stop_btn.clicked.connect(self.stop_inference)
        self.inference_stop_btn.setEnabled(False)
        control_layout.addWidget(self.inference_stop_btn)
        
        # 添加測試按鈕
        self.inference_test_btn = QPushButton("🧪 快速測試")
        self.inference_test_btn.clicked.connect(self.run_inference_test)
        self.inference_test_btn.setToolTip("運行快速測試驗證推理器功能")
        control_layout.addWidget(self.inference_test_btn)
        
        layout.addWidget(control_group)
        
        return tab
        
    def create_model_analyzer_tab(self):
        """創建模型分析標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型選擇區域
        model_selection_group = QGroupBox("模型選擇")
        model_selection_layout = QGridLayout(model_selection_group)
        
        # 檔案類型選擇器
        model_selection_layout.addWidget(QLabel("檔案類型:"), 0, 0)
        self.analyzer_file_type_combo = QComboBox()
        self.analyzer_file_type_combo.addItems(["全部", "YAML", "PT", "PTH"])
        self.analyzer_file_type_combo.currentTextChanged.connect(self.apply_file_type_filter)
        self.analyzer_file_type_combo.setMinimumWidth(100)
        model_selection_layout.addWidget(self.analyzer_file_type_combo, 0, 1)
        
        # 模型文件選擇
        model_selection_layout.addWidget(QLabel("選擇模型文件:"), 0, 2)
        self.analyzer_model_combo = QComboBox()
        self.analyzer_model_combo.setMinimumWidth(300)
        self.analyzer_model_combo.currentTextChanged.connect(self.update_analyzer_model_info)
        model_selection_layout.addWidget(self.analyzer_model_combo, 0, 3)
        
        # 控制按鈕
        self.refresh_analyzer_models_btn = QPushButton("🔄 刷新模型列表")
        self.refresh_analyzer_models_btn.clicked.connect(self.refresh_analyzer_model_list)
        model_selection_layout.addWidget(self.refresh_analyzer_models_btn, 1, 0)
        
        self.browse_analyzer_model_btn = QPushButton("📁 選擇其他資料夾")
        self.browse_analyzer_model_btn.clicked.connect(self.browse_analyzer_model_folder)
        model_selection_layout.addWidget(self.browse_analyzer_model_btn, 1, 1)
        
        self.analyze_model_btn = QPushButton("🔬 分析模型")
        self.analyze_model_btn.clicked.connect(self.analyze_selected_model)
        model_selection_layout.addWidget(self.analyze_model_btn, 1, 2)
        
        self.batch_analyze_btn = QPushButton("📊 批量分析")
        self.batch_analyze_btn.clicked.connect(self.batch_analyze_models)
        model_selection_layout.addWidget(self.batch_analyze_btn, 1, 3)
        
        # 模型信息顯示
        self.analyzer_model_status = QLabel("請選擇模型文件")
        self.analyzer_model_status.setStyleSheet("color: #666666; font-size: 11px;")
        model_selection_layout.addWidget(self.analyzer_model_status, 2, 0, 1, 4)
        
        layout.addWidget(model_selection_group)
        
        # 分析結果顯示區域
        results_group = QGroupBox("分析結果")
        results_layout = QVBoxLayout(results_group)
        
        self.analyzer_results = QTextEdit()
        self.analyzer_results.setReadOnly(True)
        self.analyzer_results.setFont(QFont("Consolas", 9))
        self.analyzer_results.setMinimumHeight(400)
        results_layout.addWidget(self.analyzer_results)
        
        # 結果控制按鈕
        results_control_layout = QHBoxLayout()
        
        self.save_analysis_btn = QPushButton("💾 保存分析結果")
        self.save_analysis_btn.clicked.connect(self.save_analysis_results)
        results_control_layout.addWidget(self.save_analysis_btn)
        
        self.clear_analysis_btn = QPushButton("🗑️ 清空結果")
        self.clear_analysis_btn.clicked.connect(self.clear_analysis_results)
        results_control_layout.addWidget(self.clear_analysis_btn)
        
        results_layout.addLayout(results_control_layout)
        
        layout.addWidget(results_group)
        
        return tab
    
    def create_model_modifier_tab(self):
        """創建模型修改器標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 標題
        title_group = QGroupBox("🔧 模型修改器")
        title_layout = QVBoxLayout(title_group)
        
        title_label = QLabel("模型通道數修改器")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0078d4; margin: 10px;")
        title_layout.addWidget(title_label)
        
        desc_label = QLabel("修改 PyTorch 模型的輸入通道數，解決通道數不匹配問題")
        desc_label.setStyleSheet("color: #666666; font-size: 12px; margin-bottom: 10px;")
        title_layout.addWidget(desc_label)
        
        layout.addWidget(title_group)
        
        # 模型選擇
        model_group = QGroupBox("模型選擇")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("輸入模型:"), 0, 0)
        self.modifier_input_model_edit = QLineEdit()
        self.modifier_input_model_edit.setPlaceholderText("選擇要修改的 PyTorch 模型文件 (.pt)")
        model_layout.addWidget(self.modifier_input_model_edit, 1, 0)
        
        self.modifier_browse_input_btn = QPushButton("瀏覽")
        self.modifier_browse_input_btn.clicked.connect(self.browse_modifier_input_model)
        model_layout.addWidget(self.modifier_browse_input_btn, 1, 1)
        
        # 輸出模型
        model_layout.addWidget(QLabel("輸出模型:"), 2, 0)
        self.modifier_output_model_edit = QLineEdit()
        self.modifier_output_model_edit.setPlaceholderText("修改後的模型保存路徑")
        model_layout.addWidget(self.modifier_output_model_edit, 3, 0)
        
        self.modifier_browse_output_btn = QPushButton("瀏覽")
        self.modifier_browse_output_btn.clicked.connect(self.browse_modifier_output_model)
        model_layout.addWidget(self.modifier_browse_output_btn, 3, 1)
        
        layout.addWidget(model_group)
        
        # 通道數設置
        channel_group = QGroupBox("通道數設置")
        channel_layout = QGridLayout(channel_group)
        
        channel_layout.addWidget(QLabel("原始通道數:"), 0, 0)
        self.modifier_original_channels_spin = QSpinBox()
        self.modifier_original_channels_spin.setRange(1, 10)
        self.modifier_original_channels_spin.setValue(3)
        self.modifier_original_channels_spin.setToolTip("模型當前的第一層輸入通道數")
        channel_layout.addWidget(self.modifier_original_channels_spin, 0, 1)
        
        channel_layout.addWidget(QLabel("目標通道數:"), 1, 0)
        self.modifier_target_channels_spin = QSpinBox()
        self.modifier_target_channels_spin.setRange(1, 10)
        self.modifier_target_channels_spin.setValue(4)
        self.modifier_target_channels_spin.setToolTip("修改後的第一層輸入通道數")
        channel_layout.addWidget(self.modifier_target_channels_spin, 1, 1)
        
        # 權重初始化方法
        channel_layout.addWidget(QLabel("權重初始化:"), 2, 0)
        self.modifier_weight_method_combo = QComboBox()
        self.modifier_weight_method_combo.addItems([
            "複製原始權重 + 平均值",
            "複製原始權重 + 零初始化", 
            "複製原始權重 + 隨機初始化",
            "完全隨機初始化"
        ])
        self.modifier_weight_method_combo.setToolTip("新通道的權重初始化方法")
        channel_layout.addWidget(self.modifier_weight_method_combo, 2, 1)
        
        layout.addWidget(channel_group)
        
        # 模型信息顯示
        info_group = QGroupBox("模型信息")
        info_layout = QVBoxLayout(info_group)
        
        self.modifier_model_info_text = QTextEdit()
        self.modifier_model_info_text.setReadOnly(True)
        self.modifier_model_info_text.setMaximumHeight(150)
        self.modifier_model_info_text.setPlainText("請選擇模型文件以查看詳細信息")
        info_layout.addWidget(self.modifier_model_info_text)
        
        layout.addWidget(info_group)
        
        # 修改器說明
        desc_group = QGroupBox("修改器說明")
        desc_layout = QVBoxLayout(desc_group)
        
        desc_text = QTextEdit()
        desc_text.setPlainText("""
        模型修改器功能說明：

        1. 通道數修改：
        - 自動檢測模型第一層的輸入通道數
        - 支持增加或減少通道數
        - 智能權重初始化

        2. 權重初始化方法：
        - 複製原始權重 + 平均值：新通道使用原始通道的平均值
        - 複製原始權重 + 零初始化：新通道權重設為零
        - 複製原始權重 + 隨機初始化：新通道使用隨機權重
        - 完全隨機初始化：所有權重重新隨機初始化

        3. 適用場景：
        - 3通道模型 → 4通道數據
        - 4通道模型 → 3通道數據
        - 其他通道數不匹配問題

        4. 注意事項：
        - 修改後的模型需要重新訓練
        - 建議使用「複製原始權重 + 平均值」方法
        - 修改前請備份原始模型
        """)
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(200)
        desc_layout.addWidget(desc_text)
        
        layout.addWidget(desc_group)
        
        # 控制按鈕
        control_group = QGroupBox("操作控制")
        control_layout = QHBoxLayout(control_group)
        
        self.modifier_analyze_btn = QPushButton("🔍 分析模型")
        self.modifier_analyze_btn.clicked.connect(self.analyze_model_for_modification)
        control_layout.addWidget(self.modifier_analyze_btn)
        
        self.modifier_modify_btn = QPushButton("🔧 修改模型")
        self.modifier_modify_btn.clicked.connect(self.modify_model_channels)
        control_layout.addWidget(self.modifier_modify_btn)
        
        self.modifier_clear_btn = QPushButton("🗑️ 清空")
        self.modifier_clear_btn.clicked.connect(self.clear_modifier_fields)
        control_layout.addWidget(self.modifier_clear_btn)
        
        layout.addWidget(control_group)
        
        return tab
    
    def create_stereo_tab(self):
        """創建立體視覺標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 立體視覺訓練設置
        stereo_group = QGroupBox("立體視覺深度估計訓練")
        stereo_layout = QGridLayout(stereo_group)
        
        # 數據集設置
        stereo_layout.addWidget(QLabel("立體數據集路徑:"), 0, 0)
        self.stereo_dataset_edit = QLineEdit()
        self.stereo_dataset_edit.setPlaceholderText("選擇立體視覺數據集目錄")
        stereo_layout.addWidget(self.stereo_dataset_edit, 0, 1)
        
        self.stereo_dataset_btn = QPushButton("📁 瀏覽")
        self.stereo_dataset_btn.clicked.connect(self.browse_stereo_dataset)
        stereo_layout.addWidget(self.stereo_dataset_btn, 0, 2)
        
        # 模型設置
        stereo_layout.addWidget(QLabel("預訓練模型:"), 1, 0)
        self.stereo_model_combo = QComboBox()
        self.stereo_model_combo.addItems([
            "raftstereo-sceneflow.pth",
            "raftstereo-middlebury.pth", 
            "raftstereo-eth3d.pth",
            "raftstereo-realtime.pth",
            "iraftstereo_rvc.pth"
        ])
        stereo_layout.addWidget(self.stereo_model_combo, 1, 1, 1, 2)
        
        # 訓練參數
        stereo_layout.addWidget(QLabel("批次大小:"), 2, 0)
        self.stereo_batch_size = QSpinBox()
        self.stereo_batch_size.setRange(1, 32)
        self.stereo_batch_size.setValue(6)
        stereo_layout.addWidget(self.stereo_batch_size, 2, 1)
        
        stereo_layout.addWidget(QLabel("學習率:"), 2, 2)
        self.stereo_lr = QDoubleSpinBox()
        self.stereo_lr.setRange(0.00001, 0.01)
        self.stereo_lr.setValue(0.0002)
        self.stereo_lr.setDecimals(5)
        stereo_layout.addWidget(self.stereo_lr, 2, 3)
        
        stereo_layout.addWidget(QLabel("訓練步數:"), 3, 0)
        self.stereo_steps = QSpinBox()
        self.stereo_steps.setRange(1000, 1000000)
        self.stereo_steps.setValue(100000)
        stereo_layout.addWidget(self.stereo_steps, 3, 1)
        
        stereo_layout.addWidget(QLabel("圖像尺寸:"), 3, 2)
        self.stereo_image_size = QLineEdit("320,720")
        self.stereo_image_size.setPlaceholderText("寬度,高度")
        stereo_layout.addWidget(self.stereo_image_size, 3, 3)
        
        # 高級設置
        advanced_group = QGroupBox("高級設置")
        advanced_layout = QGridLayout(advanced_group)
        
        advanced_layout.addWidget(QLabel("相關實現:"), 0, 0)
        self.stereo_corr_impl = QComboBox()
        self.stereo_corr_impl.addItems(["reg", "alt", "reg_cuda", "alt_cuda"])
        self.stereo_corr_impl.setCurrentText("reg")
        advanced_layout.addWidget(self.stereo_corr_impl, 0, 1)
        
        advanced_layout.addWidget(QLabel("相關層數:"), 0, 2)
        self.stereo_corr_levels = QSpinBox()
        self.stereo_corr_levels.setRange(1, 8)
        self.stereo_corr_levels.setValue(4)
        advanced_layout.addWidget(self.stereo_corr_levels, 0, 3)
        
        advanced_layout.addWidget(QLabel("訓練迭代:"), 1, 0)
        self.stereo_train_iters = QSpinBox()
        self.stereo_train_iters.setRange(1, 50)
        self.stereo_train_iters.setValue(16)
        advanced_layout.addWidget(self.stereo_train_iters, 1, 1)
        
        advanced_layout.addWidget(QLabel("驗證迭代:"), 1, 2)
        self.stereo_valid_iters = QSpinBox()
        self.stereo_valid_iters.setRange(1, 100)
        self.stereo_valid_iters.setValue(32)
        advanced_layout.addWidget(self.stereo_valid_iters, 1, 3)
        
        # 混合精度
        self.stereo_mixed_precision = QCheckBox("混合精度訓練")
        self.stereo_mixed_precision.setChecked(True)
        advanced_layout.addWidget(self.stereo_mixed_precision, 2, 0, 1, 2)
        
        # 共享骨幹網絡
        self.stereo_shared_backbone = QCheckBox("共享骨幹網絡")
        advanced_layout.addWidget(self.stereo_shared_backbone, 2, 2, 1, 2)
        
        layout.addWidget(stereo_group)
        layout.addWidget(advanced_group)
        
        # 控制按鈕
        control_group = QGroupBox("訓練控制")
        control_layout = QHBoxLayout(control_group)
        
        self.stereo_start_btn = QPushButton("🚀 開始訓練")
        self.stereo_start_btn.clicked.connect(self.start_stereo_training)
        control_layout.addWidget(self.stereo_start_btn)
        
        self.stereo_stop_btn = QPushButton("⏹️ 停止訓練")
        self.stereo_stop_btn.clicked.connect(self.stop_stereo_training)
        self.stereo_stop_btn.setEnabled(False)
        control_layout.addWidget(self.stereo_stop_btn)
        
        self.stereo_clear_btn = QPushButton("🗑️ 清空設置")
        self.stereo_clear_btn.clicked.connect(self.clear_stereo_settings)
        control_layout.addWidget(self.stereo_clear_btn)
        
        layout.addWidget(control_group)
        
        return tab
    
    def create_log_tab(self):
        """創建日誌標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 日誌控制
        log_control_layout = QHBoxLayout()
        
        self.clear_log_btn = QPushButton("🗑️ 清空日誌")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_control_layout.addWidget(self.clear_log_btn)
        
        self.save_log_btn = QPushButton("💾 保存日誌")
        self.save_log_btn.clicked.connect(self.save_log)
        log_control_layout.addWidget(self.save_log_btn)
        
        layout.addLayout(log_control_layout)
        
        # 日誌顯示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_text)
        
        return tab
        
    def create_status_bar(self):
        """創建狀態欄"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 狀態標籤
        self.status_label = QLabel("就緒")
        self.status_bar.addWidget(self.status_label)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def auto_load_configs(self):
        """自動載入配置"""
        self.log_message("🔄 初始化完成，等待用戶操作...")
        
        # 先刷新模型列表
        self.refresh_model_list()
        
        # 載入設置（包含恢復上次選擇）
        self.load_settings()
        
        # 更新模型信息
        self.update_model_info()
        
        # 根據記錄的數值刷新標準訓練模型部分
        self.auto_refresh_standard_training()
    
    def save_settings(self):
        """保存GUI設置到配置文件 (Save GUI settings to config file)"""
        try:
            import yaml
            
            # 確保config目錄存在
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 收集當前設置
            settings = {
                'convert': {
                    'source_path': self.convert_source_edit.text() if hasattr(self, 'convert_source_edit') else "",
                    'output_path': self.convert_output_edit.text() if hasattr(self, 'convert_output_edit') else "",
                    'use_depth': self.use_depth_radio.isChecked() if hasattr(self, 'use_depth_radio') else True,
                    'use_stereo': self.stereo_radio.isChecked() if hasattr(self, 'stereo_radio') else False,
                    'folder_count': self.folder_count_spin.value() if hasattr(self, 'folder_count_spin') else 1,
                },
                'standard_training': {
                    'epochs': self.epochs_spin.value() if hasattr(self, 'epochs_spin') else 50,
                    'batch_size': self.batch_size_spin.value() if hasattr(self, 'batch_size_spin') else 16,
                    'learning_rate': self.learning_rate_spin.value() if hasattr(self, 'learning_rate_spin') else 1,
                    'imgsz': self.imgsz_spin.value() if hasattr(self, 'imgsz_spin') else 640,
                    'save_period': self.save_period_spin.value() if hasattr(self, 'save_period_spin') else 10,
                    'scale': self.scale_spin.value() if hasattr(self, 'scale_spin') else 50,
                    'mosaic': self.mosaic_spin.value() if hasattr(self, 'mosaic_spin') else 100,
                    'mixup': self.mixup_spin.value() if hasattr(self, 'mixup_spin') else 0,
                    'copy_paste': self.copy_paste_spin.value() if hasattr(self, 'copy_paste_spin') else 10,
                    # 新增的HSV和BGR增強參數
                    'hsv_h': self.hsv_h_spin.value() if hasattr(self, 'hsv_h_spin') else 0,
                    'hsv_s': self.hsv_s_spin.value() if hasattr(self, 'hsv_s_spin') else 0,
                    'hsv_v': self.hsv_v_spin.value() if hasattr(self, 'hsv_v_spin') else 0,
                    'bgr': self.bgr_spin.value() if hasattr(self, 'bgr_spin') else 0,
                    'auto_augment': self.auto_augment_combo.currentData() if hasattr(self, 'auto_augment_combo') else None,
                    # 新增的幾何變換參數
                    'degrees': self.degrees_spin.value() if hasattr(self, 'degrees_spin') else 0,
                    'translate': self.translate_spin.value() if hasattr(self, 'translate_spin') else 0,
                    'shear': self.shear_spin.value() if hasattr(self, 'shear_spin') else 0,
                    'perspective': self.perspective_spin.value() if hasattr(self, 'perspective_spin') else 0,
                    # 新增的翻轉和裁剪參數
                    'flipud': self.flipud_spin.value() if hasattr(self, 'flipud_spin') else 0,
                    'fliplr': self.fliplr_spin.value() if hasattr(self, 'fliplr_spin') else 0,
                    'erasing': self.erasing_spin.value() if hasattr(self, 'erasing_spin') else 0,
                    'crop_fraction': self.crop_fraction_spin.value() if hasattr(self, 'crop_fraction_spin') else 0,
                    # 新增的訓練控制參數
                    'close_mosaic': self.close_mosaic_spin.value() if hasattr(self, 'close_mosaic_spin') else 10,
                    'workers': self.workers_spin.value() if hasattr(self, 'workers_spin') else 0,
                    'optimizer': self.optimizer_combo.currentText() if hasattr(self, 'optimizer_combo') else 'SGD',
                    'amp': self.amp_checkbox.isChecked() if hasattr(self, 'amp_checkbox') else True,
                    'dataset_path': self.train_dataset_combo.currentData() if hasattr(self, 'train_dataset_combo') else "",
                    'model_file': self.model_file_combo.currentData() if hasattr(self, 'model_file_combo') else "",
                    'last_used_dataset': self.train_dataset_combo.currentData() if hasattr(self, 'train_dataset_combo') else "",
                    'last_used_model': self.model_file_combo.currentData() if hasattr(self, 'model_file_combo') else "",
                    'training_mode': 'retrain' if hasattr(self, 'retrain_radio') and self.retrain_radio.isChecked() else 'pretrained',
                },
                'inference': {
                    'model_path': self.inference_model_edit.text() if hasattr(self, 'inference_model_edit') else "yolov12n_4channel.pt",
                    'confidence_threshold': self.inference_confidence_spin.value() if hasattr(self, 'inference_confidence_spin') else 25,
                    'num_classes': self.inference_num_classes_spin.value() if hasattr(self, 'inference_num_classes_spin') else 1,
                    'iou_threshold': self.inference_iou_spin.value() if hasattr(self, 'inference_iou_spin') else 45,
                    'max_det': self.inference_max_det_spin.value() if hasattr(self, 'inference_max_det_spin') else 300,
                    'line_width': self.inference_line_width_spin.value() if hasattr(self, 'inference_line_width_spin') else 3,
                    'show_labels': self.inference_show_labels_check.isChecked() if hasattr(self, 'inference_show_labels_check') else True,
                    'show_conf': self.inference_show_conf_check.isChecked() if hasattr(self, 'inference_show_conf_check') else True,
                    'show_boxes': self.inference_show_boxes_check.isChecked() if hasattr(self, 'inference_show_boxes_check') else True,
                    'save_txt': self.inference_save_txt_check.isChecked() if hasattr(self, 'inference_save_txt_check') else True,
                    'save_conf': self.inference_save_conf_check.isChecked() if hasattr(self, 'inference_save_conf_check') else True,
                    'save_crop': self.inference_save_crop_check.isChecked() if hasattr(self, 'inference_save_crop_check') else False,
                    'visualize': self.inference_visualize_check.isChecked() if hasattr(self, 'inference_visualize_check') else True,
                    'augment': self.inference_augment_check.isChecked() if hasattr(self, 'inference_augment_check') else False,
                    'agnostic_nms': self.inference_agnostic_nms_check.isChecked() if hasattr(self, 'inference_agnostic_nms_check') else False,
                    'retina_masks': self.inference_retina_masks_check.isChecked() if hasattr(self, 'inference_retina_masks_check') else False,
                    'format': self.inference_format_combo.currentText() if hasattr(self, 'inference_format_combo') else "torch",
                    'verbose': self.inference_verbose_check.isChecked() if hasattr(self, 'inference_verbose_check') else False,
                    'show': self.inference_show_check.isChecked() if hasattr(self, 'inference_show_check') else False,
                    'mode': self.inference_mode_combo.currentText() if hasattr(self, 'inference_mode_combo') else "Data目錄處理模式",
                    'dataset_path': self.dataset_path_edit.text() if hasattr(self, 'dataset_path_edit') else "",
                },
                'model_analyzer': {
                    'selected_model': self.analyzer_model_combo.currentData() if hasattr(self, 'analyzer_model_combo') else None,
                },
                'model_modifier': {
                    'input_model': self.modifier_input_model_edit.text() if hasattr(self, 'modifier_input_model_edit') else "",
                    'output_model': self.modifier_output_model_edit.text() if hasattr(self, 'modifier_output_model_edit') else "",
                    'original_channels': self.modifier_original_channels_spin.value() if hasattr(self, 'modifier_original_channels_spin') else 3,
                    'target_channels': self.modifier_target_channels_spin.value() if hasattr(self, 'modifier_target_channels_spin') else 4,
                    'weight_method': self.modifier_weight_method_combo.currentText() if hasattr(self, 'modifier_weight_method_combo') else "複製原始權重 + 平均值",
                },
                'window': {
                    'last_tab_index': self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
                    'geometry': {
                        'x': self.geometry().x(),
                        'y': self.geometry().y(),
                        'width': self.geometry().width(),
                        'height': self.geometry().height(),
                    }
                },
                'last_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 寫入配置文件
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                yaml.dump(settings, f, allow_unicode=True, default_flow_style=False)
            
            # 調試信息：記錄保存的位置
            self.log_message(f"💾 設置已保存到: {self.settings_file.name}")
            self.log_message(f"📍 視窗位置已保存: ({self.geometry().x()}, {self.geometry().y()}) 大小: {self.geometry().width()}x{self.geometry().height()}")
            
        except Exception as e:
            self.log_message(f"[WARNING] 保存設置失敗: {e}")
    
    def load_settings(self):
        """從配置文件加載GUI設置 (Load GUI settings from config file)"""
        # 初始化窗口位置載入標記
        self._window_geometry_loaded = False
        
        try:
            import yaml
            
            # 檢查配置文件是否存在
            if not self.settings_file.exists():
                self.log_message("ℹ️ 未找到設置文件，使用默認設置")
                return
            
            # 讀取配置文件
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            if not settings:
                self.log_message("[WARNING] 設置文件為空，使用默認設置")
                return
            
            # 恢復數據轉換設置
            if 'convert' in settings:
                convert = settings['convert']
                if hasattr(self, 'convert_source_edit') and convert.get('source_path'):
                    self.convert_source_edit.setText(convert['source_path'])
                if hasattr(self, 'convert_output_edit') and convert.get('output_path'):
                    self.convert_output_edit.setText(convert['output_path'])
                if hasattr(self, 'use_depth_radio') and 'use_depth' in convert:
                    if convert.get('use_stereo', False):
                        self.stereo_radio.setChecked(True)
                    elif convert['use_depth']:
                        self.use_depth_radio.setChecked(True)
                    else:
                        self.no_depth_radio.setChecked(True)
                if hasattr(self, 'folder_count_spin') and convert.get('folder_count'):
                    self.folder_count_spin.setValue(convert['folder_count'])
            
            
            # 恢復標準訓練設置
            if 'standard_training' in settings:
                standard = settings['standard_training']
                if hasattr(self, 'epochs_spin') and standard.get('epochs'):
                    self.epochs_spin.setValue(standard['epochs'])
                if hasattr(self, 'batch_size_spin') and standard.get('batch_size'):
                    self.batch_size_spin.setValue(standard['batch_size'])
                if hasattr(self, 'learning_rate_spin') and standard.get('learning_rate'):
                    self.learning_rate_spin.setValue(standard['learning_rate'])
                if hasattr(self, 'imgsz_spin') and standard.get('imgsz'):
                    self.imgsz_spin.setValue(standard['imgsz'])
                if hasattr(self, 'save_period_spin') and standard.get('save_period'):
                    self.save_period_spin.setValue(standard['save_period'])
                if hasattr(self, 'scale_spin') and standard.get('scale'):
                    self.scale_spin.setValue(standard['scale'])
                if hasattr(self, 'mosaic_spin') and standard.get('mosaic'):
                    self.mosaic_spin.setValue(standard['mosaic'])
                if hasattr(self, 'mixup_spin') and standard.get('mixup'):
                    self.mixup_spin.setValue(standard['mixup'])
                if hasattr(self, 'copy_paste_spin') and standard.get('copy_paste'):
                    self.copy_paste_spin.setValue(standard['copy_paste'])
                
                # 恢復新增的HSV和BGR增強參數
                if hasattr(self, 'hsv_h_spin') and standard.get('hsv_h') is not None:
                    self.hsv_h_spin.setValue(standard['hsv_h'])
                if hasattr(self, 'hsv_s_spin') and standard.get('hsv_s') is not None:
                    self.hsv_s_spin.setValue(standard['hsv_s'])
                if hasattr(self, 'hsv_v_spin') and standard.get('hsv_v') is not None:
                    self.hsv_v_spin.setValue(standard['hsv_v'])
                if hasattr(self, 'bgr_spin') and standard.get('bgr') is not None:
                    self.bgr_spin.setValue(standard['bgr'])
                if hasattr(self, 'auto_augment_combo') and standard.get('auto_augment') is not None:
                    # 找到對應的索引
                    for i in range(self.auto_augment_combo.count()):
                        if self.auto_augment_combo.itemData(i) == standard['auto_augment']:
                            self.auto_augment_combo.setCurrentIndex(i)
                            break
                
                # 恢復新增的幾何變換參數
                if hasattr(self, 'degrees_spin') and standard.get('degrees') is not None:
                    self.degrees_spin.setValue(standard['degrees'])
                if hasattr(self, 'translate_spin') and standard.get('translate') is not None:
                    self.translate_spin.setValue(standard['translate'])
                if hasattr(self, 'shear_spin') and standard.get('shear') is not None:
                    self.shear_spin.setValue(standard['shear'])
                if hasattr(self, 'perspective_spin') and standard.get('perspective') is not None:
                    self.perspective_spin.setValue(standard['perspective'])
                
                # 恢復新增的翻轉和裁剪參數
                if hasattr(self, 'flipud_spin') and standard.get('flipud') is not None:
                    self.flipud_spin.setValue(standard['flipud'])
                if hasattr(self, 'fliplr_spin') and standard.get('fliplr') is not None:
                    self.fliplr_spin.setValue(standard['fliplr'])
                if hasattr(self, 'erasing_spin') and standard.get('erasing') is not None:
                    self.erasing_spin.setValue(standard['erasing'])
                if hasattr(self, 'crop_fraction_spin') and standard.get('crop_fraction') is not None:
                    self.crop_fraction_spin.setValue(standard['crop_fraction'])
                
                # 恢復新增的訓練控制參數
                if hasattr(self, 'close_mosaic_spin') and standard.get('close_mosaic') is not None:
                    self.close_mosaic_spin.setValue(standard['close_mosaic'])
                if hasattr(self, 'workers_spin') and standard.get('workers') is not None:
                    self.workers_spin.setValue(standard['workers'])
                if hasattr(self, 'optimizer_combo') and standard.get('optimizer'):
                    self.optimizer_combo.setCurrentText(standard['optimizer'])
                if hasattr(self, 'amp_checkbox') and standard.get('amp') is not None:
                    self.amp_checkbox.setChecked(standard['amp'])
                
                # 恢復訓練模式
                if hasattr(self, 'pretrained_radio') and hasattr(self, 'retrain_radio'):
                    training_mode = standard.get('training_mode', 'pretrained')
                    if training_mode == 'retrain':
                        self.retrain_radio.setChecked(True)
                        self.pretrained_radio.setChecked(False)
                    else:
                        self.pretrained_radio.setChecked(True)
                        self.retrain_radio.setChecked(False)
                
                # 恢復上次使用的資料集和模型
                self._restore_last_used_selections(standard)
            
            # 恢復推理設置
            if 'inference' in settings:
                inference = settings['inference']
                if hasattr(self, 'inference_model_edit') and inference.get('model_path'):
                    self.inference_model_edit.setText(inference['model_path'])
                if hasattr(self, 'inference_confidence_spin') and inference.get('confidence_threshold'):
                    self.inference_confidence_spin.setValue(inference['confidence_threshold'])
                if hasattr(self, 'inference_num_classes_spin') and inference.get('num_classes'):
                    self.inference_num_classes_spin.setValue(inference['num_classes'])
                if hasattr(self, 'inference_iou_spin') and inference.get('iou_threshold'):
                    self.inference_iou_spin.setValue(inference['iou_threshold'])
                if hasattr(self, 'inference_max_det_spin') and inference.get('max_det'):
                    self.inference_max_det_spin.setValue(inference['max_det'])
                if hasattr(self, 'inference_line_width_spin') and inference.get('line_width'):
                    self.inference_line_width_spin.setValue(inference['line_width'])
                if hasattr(self, 'inference_show_labels_check') and inference.get('show_labels') is not None:
                    self.inference_show_labels_check.setChecked(inference['show_labels'])
                if hasattr(self, 'inference_show_conf_check') and inference.get('show_conf') is not None:
                    self.inference_show_conf_check.setChecked(inference['show_conf'])
                if hasattr(self, 'inference_show_boxes_check') and inference.get('show_boxes') is not None:
                    self.inference_show_boxes_check.setChecked(inference['show_boxes'])
                if hasattr(self, 'inference_save_txt_check') and inference.get('save_txt') is not None:
                    self.inference_save_txt_check.setChecked(inference['save_txt'])
                if hasattr(self, 'inference_save_conf_check') and inference.get('save_conf') is not None:
                    self.inference_save_conf_check.setChecked(inference['save_conf'])
                if hasattr(self, 'inference_save_crop_check') and inference.get('save_crop') is not None:
                    self.inference_save_crop_check.setChecked(inference['save_crop'])
                if hasattr(self, 'inference_visualize_check') and inference.get('visualize') is not None:
                    self.inference_visualize_check.setChecked(inference['visualize'])
                if hasattr(self, 'inference_augment_check') and inference.get('augment') is not None:
                    self.inference_augment_check.setChecked(inference['augment'])
                if hasattr(self, 'inference_agnostic_nms_check') and inference.get('agnostic_nms') is not None:
                    self.inference_agnostic_nms_check.setChecked(inference['agnostic_nms'])
                if hasattr(self, 'inference_retina_masks_check') and inference.get('retina_masks') is not None:
                    self.inference_retina_masks_check.setChecked(inference['retina_masks'])
                if hasattr(self, 'inference_format_combo') and inference.get('format'):
                    self.inference_format_combo.setCurrentText(inference['format'])
                if hasattr(self, 'inference_verbose_check') and inference.get('verbose') is not None:
                    self.inference_verbose_check.setChecked(inference['verbose'])
                if hasattr(self, 'inference_show_check') and inference.get('show') is not None:
                    self.inference_show_check.setChecked(inference['show'])
                if hasattr(self, 'inference_mode_combo') and inference.get('mode'):
                    self.inference_mode_combo.setCurrentText(inference['mode'])
                if hasattr(self, 'dataset_path_edit') and inference.get('dataset_path'):
                    self.dataset_path_edit.setText(inference['dataset_path'])
            
            # 恢復模型分析器設置
            if 'model_analyzer' in settings:
                analyzer = settings['model_analyzer']
                if hasattr(self, 'analyzer_model_combo') and analyzer.get('selected_model'):
                    # 嘗試恢復選中的模型
                    for i in range(self.analyzer_model_combo.count()):
                        if self.analyzer_model_combo.itemData(i) == analyzer['selected_model']:
                            self.analyzer_model_combo.setCurrentIndex(i)
                            break
            
            # 恢復模型修改器設置
            if 'model_modifier' in settings:
                modifier = settings['model_modifier']
                if hasattr(self, 'modifier_input_model_edit') and modifier.get('input_model'):
                    self.modifier_input_model_edit.setText(modifier['input_model'])
                if hasattr(self, 'modifier_output_model_edit') and modifier.get('output_model'):
                    self.modifier_output_model_edit.setText(modifier['output_model'])
                if hasattr(self, 'modifier_original_channels_spin') and modifier.get('original_channels'):
                    self.modifier_original_channels_spin.setValue(modifier['original_channels'])
                if hasattr(self, 'modifier_target_channels_spin') and modifier.get('target_channels'):
                    self.modifier_target_channels_spin.setValue(modifier['target_channels'])
                if hasattr(self, 'modifier_weight_method_combo') and modifier.get('weight_method'):
                    self.modifier_weight_method_combo.setCurrentText(modifier['weight_method'])
            
            # 恢復窗口設置
            if 'window' in settings:
                window = settings['window']
                if 'last_tab_index' in window and hasattr(self, 'tab_widget'):
                    self.tab_widget.setCurrentIndex(window['last_tab_index'])
                # 載入視窗幾何形狀（如果存在），並進行邊界檢查
                if 'geometry' in window:
                    geo = window['geometry']
                    if all(k in geo for k in ['x', 'y', 'width', 'height']):
                        # 獲取螢幕可用區域
                        screen = QApplication.primaryScreen()
                        available_geometry = screen.availableGeometry()
                        
                        # 提取保存的位置和大小
                        saved_x = geo['x']
                        saved_y = geo['y']
                        saved_width = geo['width']
                        saved_height = geo['height']
                        
                        # 邊界檢查，確保視窗完全在螢幕內
                        margin = 10  # 減少邊距到10像素
                        
                        # 計算螢幕邊界
                        min_x = available_geometry.x() + margin
                        max_x = available_geometry.x() + available_geometry.width() - saved_width - margin
                        min_y = available_geometry.y() + margin
                        max_y = available_geometry.y() + available_geometry.height() - saved_height - margin
                        
                        # 調試信息：顯示螢幕可用區域
                        self.log_message(f"🖥️ 螢幕可用區域: x={available_geometry.x()}, y={available_geometry.y()}, w={available_geometry.width()}, h={available_geometry.height()}")
                        self.log_message(f"📏 計算的邊界: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
                        
                        # 限制位置在螢幕範圍內，但不強制移動到邊界
                        # 如果保存的位置在螢幕範圍內，就使用保存的位置
                        if (min_x <= saved_x <= max_x and min_y <= saved_y <= max_y):
                            x = saved_x
                            y = saved_y
                            self.log_message(f"✅ 保存的位置在螢幕範圍內，直接使用: ({x}, {y})")
                        else:
                            x = max(min_x, min(saved_x, max_x))
                            y = max(min_y, min(saved_y, max_y))
                            self.log_message(f"⚠️ 保存的位置超出螢幕範圍，調整為: ({x}, {y})")
                        
                        # 調試信息：記錄位置變化
                        self.log_message(f"📍 載入視窗位置: 保存的({saved_x}, {saved_y}) 大小: {saved_width}x{saved_height}")
                        if saved_x != x or saved_y != y:
                            self.log_message(f"🔧 視窗位置已調整: 原始({saved_x}, {saved_y}) → 調整後({x}, {y})")
                        
                        # 設置視窗幾何形狀
                        self.setGeometry(x, y, saved_width, saved_height)
                        # 標記已載入窗口位置
                        self._window_geometry_loaded = True
                        self.log_message(f"✅ 窗口位置已載入並設置: ({x}, {y}) 大小: {saved_width}x{saved_height}")
            
            last_saved = settings.get('last_saved', '未知')
            self.log_message(f"[OK] 已加載設置 (上次保存: {last_saved})")
            
        except Exception as e:
            self.log_message(f"[WARNING] 加載設置失敗: {e}")
    
    def _detect_dataset_file_types(self, dataset_path):
        """檢測資料集中的檔案類型"""
        try:
            dataset_dir = Path(dataset_path)
            file_types = set()
            
            # 定義所有支援的訓練檔案類型
            supported_extensions = [
                # 常見圖片格式
                '*.jpg', '*.jpeg', '*.JPG', '*.JPEG',
                '*.png', '*.PNG',
                '*.bmp', '*.BMP',
                '*.tiff', '*.tif', '*.TIFF', '*.TIF',
                '*.webp', '*.WEBP',
                '*.gif', '*.GIF',
                
                # NumPy 陣列格式
                '*.npy', '*.NPY',
                '*.npz', '*.NPZ',
                
                # 其他深度學習格式
                '*.h5', '*.hdf5', '*.H5', '*.HDF5',
                '*.pkl', '*.pickle', '*.PKL', '*.PICKLE',
                '*.pt', '*.pth', '*.PT', '*.PTH',
                
                # 壓縮格式
                '*.zip', '*.ZIP',
                '*.tar', '*.gz', '*.TAR', '*.GZ',
                
                # 影片格式（用於影片訓練）
                '*.mp4', '*.avi', '*.mov', '*.mkv',
                '*.MP4', '*.AVI', '*.MOV', '*.MKV',
                
                # 音訊格式（用於多模態訓練）
                '*.wav', '*.mp3', '*.flac', '*.aac',
                '*.WAV', '*.MP3', '*.FLAC', '*.AAC',
                
                # 文字格式（用於多模態訓練）
                '*.txt', '*.json', '*.xml', '*.csv',
                '*.TXT', '*.JSON', '*.XML', '*.CSV',
                
                # 其他格式
                '*.bin', '*.dat', '*.raw',
                '*.BIN', '*.DAT', '*.RAW'
            ]
            
            # 檢查訓練目錄
            train_dir = dataset_dir / 'images' / 'train'
            if train_dir.exists():
                for ext in supported_extensions:
                    if list(train_dir.glob(ext)):
                        # 標準化副檔名顯示
                        ext_clean = ext[1:].upper()  # 移除 * 並轉大寫
                        if ext_clean in ['JPG', 'JPEG']:
                            file_types.add('JPG')
                        elif ext_clean in ['TIFF', 'TIF']:
                            file_types.add('TIFF')
                        elif ext_clean in ['H5', 'HDF5']:
                            file_types.add('HDF5')
                        elif ext_clean in ['PKL', 'PICKLE']:
                            file_types.add('PKL')
                        elif ext_clean in ['PT', 'PTH']:
                            file_types.add('PT')
                        elif ext_clean in ['TAR', 'GZ']:
                            file_types.add('TAR')
                        elif ext_clean in ['MP4', 'AVI', 'MOV', 'MKV']:
                            file_types.add('VIDEO')
                        elif ext_clean in ['WAV', 'MP3', 'FLAC', 'AAC']:
                            file_types.add('AUDIO')
                        elif ext_clean in ['TXT', 'JSON', 'XML', 'CSV']:
                            file_types.add('TEXT')
                        else:
                            file_types.add(ext_clean)
            
            # 檢查驗證目錄
            val_dir = dataset_dir / 'images' / 'val'
            if val_dir.exists():
                for ext in supported_extensions:
                    if list(val_dir.glob(ext)):
                        # 標準化副檔名顯示
                        ext_clean = ext[1:].upper()  # 移除 * 並轉大寫
                        if ext_clean in ['JPG', 'JPEG']:
                            file_types.add('JPG')
                        elif ext_clean in ['TIFF', 'TIF']:
                            file_types.add('TIFF')
                        elif ext_clean in ['H5', 'HDF5']:
                            file_types.add('HDF5')
                        elif ext_clean in ['PKL', 'PICKLE']:
                            file_types.add('PKL')
                        elif ext_clean in ['PT', 'PTH']:
                            file_types.add('PT')
                        elif ext_clean in ['TAR', 'GZ']:
                            file_types.add('TAR')
                        elif ext_clean in ['MP4', 'AVI', 'MOV', 'MKV']:
                            file_types.add('VIDEO')
                        elif ext_clean in ['WAV', 'MP3', 'FLAC', 'AAC']:
                            file_types.add('AUDIO')
                        elif ext_clean in ['TXT', 'JSON', 'XML', 'CSV']:
                            file_types.add('TEXT')
                        else:
                            file_types.add(ext_clean)
            
            # 如果沒有找到檔案，檢查根目錄
            if not file_types:
                for ext in supported_extensions:
                    if list(dataset_dir.glob(ext)):
                        # 標準化副檔名顯示
                        ext_clean = ext[1:].upper()  # 移除 * 並轉大寫
                        if ext_clean in ['JPG', 'JPEG']:
                            file_types.add('JPG')
                        elif ext_clean in ['TIFF', 'TIF']:
                            file_types.add('TIFF')
                        elif ext_clean in ['H5', 'HDF5']:
                            file_types.add('HDF5')
                        elif ext_clean in ['PKL', 'PICKLE']:
                            file_types.add('PKL')
                        elif ext_clean in ['PT', 'PTH']:
                            file_types.add('PT')
                        elif ext_clean in ['TAR', 'GZ']:
                            file_types.add('TAR')
                        elif ext_clean in ['MP4', 'AVI', 'MOV', 'MKV']:
                            file_types.add('VIDEO')
                        elif ext_clean in ['WAV', 'MP3', 'FLAC', 'AAC']:
                            file_types.add('AUDIO')
                        elif ext_clean in ['TXT', 'JSON', 'XML', 'CSV']:
                            file_types.add('TEXT')
                        else:
                            file_types.add(ext_clean)
            
            return sorted(file_types) if file_types else ['未知']
            
        except Exception as e:
            self.log_message(f"[WARNING] 檢測檔案類型失敗: {e}")
            return ['未知']
    
    def _restore_last_used_selections(self, standard_settings):
        """恢復上次使用的資料集和模型選擇"""
        try:
            last_dataset = standard_settings.get('last_used_dataset', '')
            last_model = standard_settings.get('last_used_model', '')
            
            # 更新上次使用信息顯示
            if hasattr(self, 'last_used_info'):
                if last_dataset and last_model:
                    dataset_name = Path(last_dataset).name if last_dataset else "未知"
                    model_name = Path(last_model).name if last_model else "未知"
                    self.last_used_info.setText(f"📝 上次使用: 資料集={dataset_name}, 模型={model_name}")
                else:
                    self.last_used_info.setText("📝 首次使用，無歷史記錄")
            
            # 恢復上次使用的資料集
            if last_dataset and hasattr(self, 'train_dataset_combo'):
                # 尋找匹配的資料集
                dataset_restored = False
                for i in range(self.train_dataset_combo.count()):
                    if self.train_dataset_combo.itemData(i) == last_dataset:
                        self.train_dataset_combo.setCurrentIndex(i)
                        self.log_message(f"[OK] 已恢復上次使用的資料集: {self.train_dataset_combo.currentText()}")
                        dataset_restored = True
                        # 更新資料集信息
                        self.update_train_dataset_info()
                        break
                
                if not dataset_restored:
                    self.log_message(f"[WARNING] 未找到上次使用的資料集: {last_dataset}")
            
            # 恢復上次使用的模型
            if last_model and hasattr(self, 'model_file_combo'):
                # 尋找匹配的模型
                model_restored = False
                for i in range(self.model_file_combo.count()):
                    if self.model_file_combo.itemData(i) == last_model:
                        self.model_file_combo.setCurrentIndex(i)
                        self.log_message(f"[OK] 已恢復上次使用的模型: {self.model_file_combo.currentText()}")
                        model_restored = True
                        # 更新模型信息
                        self.update_selected_model_info()
                        break
                
                if not model_restored:
                    self.log_message(f"[WARNING] 未找到上次使用的模型: {last_model}")
                        
        except Exception as e:
            self.log_message(f"[WARNING] 恢復上次選擇失敗: {e}")
    
    def find_latest_dataset(self, dataset_dir="."):
        """尋找最新的數據集目錄"""
        dataset_path = Path(dataset_dir)
        
        # 尋找所有數據集目錄
        dataset_dirs = list(dataset_path.glob("dataset_*"))
        
        if not dataset_dirs:
            return None
        
        # 過濾出包含標準圖像文件的數據集（排除4通道NPY數據集）
        valid_datasets = []
        for dataset_dir in dataset_dirs:
            train_dir = dataset_dir / 'images' / 'train'
            if train_dir.exists():
                # 檢查是否包含標準圖像文件
                image_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
                npy_files = list(train_dir.glob('*.npy'))
                
                # 如果包含圖像文件且不包含NPY文件，則為標準數據集
                if image_files and not npy_files:
                    valid_datasets.append(dataset_dir)
        
        if not valid_datasets:
            return None
            
        # 按修改時間排序，取最新的
        latest_dataset = max(valid_datasets, key=lambda x: x.stat().st_mtime)
        return str(latest_dataset)
    
    def find_latest_config(self, dataset_dir="."):
        """尋找最新的數據配置"""
        dataset_path = Path(dataset_dir)
        
        # 首先尋找 dataset_*/data_config.yaml
        config_files = list(dataset_path.glob("dataset_*/data_config.yaml"))
        
        if not config_files:
            # 嘗試根目錄的 data_config.yaml
            root_config = dataset_path / "data_config.yaml"
            if root_config.exists():
                config_files = [root_config]
        
        if not config_files:
            return None
            
        # 按修改時間排序，取最新的
        latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
        return str(latest_config)
    
    def log_message(self, message):
        """添加日誌消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # 檢查 log_text 是否存在
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.append(formatted_message)
            
            # 自動滾動到底部
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        else:
            # 如果 log_text 不存在，靜默處理
            pass
    
    def update_status(self, message):
        """更新狀態欄"""
        self.status_label.setText(message)
        QApplication.processEvents()
          
    def show_progress(self, show=True, current=0, total=0, text=""):
        """顯示/隱藏進度條"""
        self.progress_bar.setVisible(show)
        if show and total > 0:
            # 確定進度模式
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            if text:
                self.progress_bar.setFormat(f"{text} ({current}/{total})")
            else:
                self.progress_bar.setFormat(f"進度: {current}/{total}")
        elif show:
            # 不確定進度模式
            self.progress_bar.setRange(0, 0)
            if text:
                self.progress_bar.setFormat(text)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setFormat("")
    
    # 文件瀏覽方法
    def browse_convert_source(self):
        """瀏覽轉換源路徑"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "選擇Forest數據集根目錄"
        )
        if folder_path:
            self.convert_source_edit.setText(folder_path)
    
    def browse_convert_output(self):
        """瀏覽轉換輸出路徑"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "選擇輸出路徑"
        )
        if folder_path:
            self.convert_output_edit.setText(folder_path)
    
    def _validate_source_path(self, path_text, show_warning=True):
        """驗證源路徑 - 统一的路径验证函数 (Unified path validation)"""
        if not path_text:
            if show_warning:
                QMessageBox.warning(self, "警告 Warning", "請選擇源數據路徑 Please select source data path")
            return None
        
        source_path = Path(path_text)
        if not source_path.exists():
            if show_warning:
                QMessageBox.warning(self, "警告 Warning", "源路徑不存在，請檢查路徑是否正確 Source path does not exist")
            return None
        
        return source_path
    
    def auto_detect_folders(self):
        """自動偵測資料夾數量"""
        source_path = self._validate_source_path(self.convert_source_edit.text())
        if not source_path:
            return
        
        try:
            # 偵測Forest格式資料夾
            forest_folders = [f for f in source_path.iterdir() if f.is_dir() and f.name.startswith('Forest_Video_')]
            
            if forest_folders:
                self.folder_status_label.setText(f"[OK] 偵測到 {len(forest_folders)} 個Forest資料夾")
                self.folder_status_label.setStyleSheet("color: #28a745; font-size: 11px;")
                self.folder_count_spin.setRange(1, len(forest_folders))
                self.folder_count_spin.setValue(len(forest_folders))  # 預設為偵測到的全部數量
                self.log_message(f"[SEARCH] 偵測到 {len(forest_folders)} 個Forest資料夾，預設處理全部")
            else:
                # 檢查是否為單一資料夾格式
                required_folders = ['Img', 'YOLO_Label']
                has_required = all((source_path / folder).exists() for folder in required_folders)
                
                if has_required:
                    self.folder_status_label.setText("[OK] 偵測到單一資料夾格式")
                    self.folder_status_label.setStyleSheet("color: #28a745; font-size: 11px;")
                    self.folder_count_spin.setRange(1, 1)
                    self.folder_count_spin.setValue(1)
                    self.log_message("[SEARCH] 偵測到單一資料夾格式")
                else:
                    self.folder_status_label.setText("[ERROR] 未偵測到有效的資料夾格式")
                    self.folder_status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
                    self.folder_count_spin.setRange(1, 1)
                    self.folder_count_spin.setValue(1)
                    self.log_message("[ERROR] 未偵測到有效的資料夾格式")
                    
        except Exception as e:
            self.folder_status_label.setText(f"[ERROR] 偵測失敗: {str(e)}")
            self.folder_status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
            self.log_message(f"[ERROR] 偵測資料夾失敗: {e}")
    
    def browse_train_dataset(self):
        """瀏覽訓練資料集資料夾"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "選擇資料集資料夾"
        )
        if folder_path:
            # 檢查是否包含data_config.yaml
            config_file = Path(folder_path) / "data_config.yaml"
            if config_file.exists():
                try:
                    # 讀取配置文件信息
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    channels = config_data.get('channels', 3)
                    nc = config_data.get('nc', 1)
                    
                    # 如果配置文件中沒有類別數量，嘗試從預定義類別獲取
                    if nc == 1 and 'names' not in config_data:
                        nc = self.get_dynamic_class_count()
                        self.log_message(f"📋 從預定義類別獲取類別數量: {nc}")
                    
                    # 構建顯示名稱
                    display_name = f"{Path(folder_path).name} ({channels}通道, {nc}類別)"
                    
                    # 添加到下拉選單
                    self.train_dataset_combo.addItem(display_name, folder_path)
                    self.train_dataset_combo.setCurrentText(display_name)
                    
                    self.log_message(f"[OK] 已添加資料集: {Path(folder_path).name}")
                    self.update_train_dataset_info()
                    
                except Exception as e:
                    self.log_message(f"[WARNING] 讀取配置文件失敗: {e}")
                    # 不再弹出警告窗口，只记录日志
                    # QMessageBox.warning(self, "警告", f"讀取配置文件失敗: {e}")
            else:
                self.log_message("[WARNING] 選擇的資料夾不包含data_config.yaml文件")
                # 不再弹出警告窗口，只记录日志
                # QMessageBox.warning(self, "警告", "選擇的資料夾不包含data_config.yaml文件")
    
    def auto_find_train_dataset(self):
        """自動尋找訓練資料集 - 使用與自定義訓練相同的方式"""
        # 清空現有選項
        self.train_dataset_combo.clear()
        
        try:
            # 尋找包含data_config.yaml文件的數據集目錄（與自定義訓練邏輯一致）
            dataset_dirs = list(Path("Dataset").glob("dataset_*"))
            standard_datasets = []
            
            for dataset_dir in dataset_dirs:
                config_file = dataset_dir / 'data_config.yaml'
                if config_file.exists():
                    # 驗證配置文件
                    try:
                        from Code.YOLO_standard_trainer import ConfigDetector
                        config_info = ConfigDetector.validate_config(str(config_file))
                        
                        if config_info['valid']:
                            # 獲取數據集信息
                            channels = config_info.get('channels', '未知')
                            nc = config_info.get('nc', '未知')
                            standard_datasets.append({
                                'path': str(dataset_dir),
                                'name': dataset_dir.name,
                                'channels': channels,
                                'nc': nc,
                                'config_info': config_info
                            })
                    except Exception as e:
                        self.log_message(f"[WARNING] 配置文件驗證失敗 {dataset_dir.name}: {e}")
                        continue
            
            if standard_datasets:
                # 按修改時間排序
                standard_datasets.sort(key=lambda x: Path(x['path']).stat().st_mtime, reverse=True)
                
                # 填充下拉選單
                for dataset_info in standard_datasets:
                    # 檢測檔案類型
                    file_types = self._detect_dataset_file_types(dataset_info['path'])
                    file_types_str = ', '.join(file_types)
                    
                    # 構建顯示名稱
                    display_name = f"{dataset_info['name']} ({dataset_info['channels']}通道, {dataset_info['nc']}類別, {file_types_str})"
                    self.train_dataset_combo.addItem(display_name, dataset_info['path'])
                
                # 自動選擇最新的資料集
                self.train_dataset_combo.setCurrentIndex(0)
                latest_dataset = standard_datasets[0]
                self.log_message(f"[OK] 找到 {len(standard_datasets)} 個標準資料集，已選擇最新的: {latest_dataset['name']}")
                
                if len(standard_datasets) > 1:
                    other_datasets = [d['name'] for d in standard_datasets[1:]]
                    self.log_message(f"📋 其他可用資料集: {other_datasets}")
                
                # 更新資料集信息
                self.update_train_dataset_info()
            else:
                self.log_message("[WARNING] 未找到包含有效data_config.yaml的資料集")
                # 不再弹出警告窗口，只记录日志
                # QMessageBox.warning(self, "警告", "在Dataset目錄中未找到有效的資料集")
                
        except Exception as e:
            self.log_message(f"[ERROR] 自動尋找資料集失敗: {e}")
            # 不再弹出警告窗口，只记录日志
            # QMessageBox.warning(self, "警告", f"自動尋找資料集失敗: {e}")
    
    def update_train_dataset_info(self):
        """更新訓練資料集信息 - 使用與自定義訓練相同的方式"""
        # 從下拉選單獲取選擇的資料集路徑
        dataset_path = self.train_dataset_combo.currentData()
        if not dataset_path:
            # 如果沒有data，嘗試從currentText獲取
            dataset_path = self.train_dataset_combo.currentText()
        
        if not dataset_path:
            self.train_dataset_status.setText("請選擇資料集")
            self.train_dataset_status.setStyleSheet("color: #666666; font-size: 11px;")
            return
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            self.train_dataset_status.setText("[ERROR] 資料夾不存在")
            self.train_dataset_status.setStyleSheet("color: #dc3545; font-size: 11px;")
            return
        
        # 檢查是否包含data_config.yaml
        config_file = dataset_dir / "data_config.yaml"
        if config_file.exists():
            try:
                # 驗證配置文件
                from Code.YOLO_standard_trainer import ConfigDetector
                config_info = ConfigDetector.validate_config(str(config_file))
                
                if config_info['valid']:
                    channels = config_info['channels']
                    nc = config_info['nc']
                    
                    # 檢查數據集結構（與自定義訓練邏輯一致）
                    train_dir = dataset_dir / 'images' / 'train'
                    val_dir = dataset_dir / 'images' / 'val'
                    
                    train_count = 0
                    val_count = 0
                    
                    if train_dir.exists():
                        train_count = len(list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')))
                    if val_dir.exists():
                        val_count = len(list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png')))
                    
                    # 檢測檔案類型
                    file_types = self._detect_dataset_file_types(dataset_path)
                    file_types_str = ', '.join(file_types)
                    
                    status_text = f"[OK] 資料集有效: {channels}通道, 類別數: {nc}, 檔案類型: {file_types_str}"
                    if train_count > 0 or val_count > 0:
                        status_text += f" (訓練: {train_count}, 驗證: {val_count})"
                    
                    self.train_dataset_status.setText(status_text)
                    self.train_dataset_status.setStyleSheet("color: #28a745; font-size: 11px;")
                    
                    # 自動檢測圖片尺寸
                    self.auto_detect_image_size(dataset_dir)
                    
                    # 數據集信息更新後，刷新模型列表以顯示正確的通道數
                    self.refresh_model_list()
                else:
                    self.train_dataset_status.setText(f"[WARNING] 配置文件格式錯誤: {config_info.get('error', '未知錯誤')}")
                    self.train_dataset_status.setStyleSheet("color: #ffc107; font-size: 11px;")
            except Exception as e:
                self.train_dataset_status.setText(f"[WARNING] 配置文件驗證失敗: {str(e)}")
                self.train_dataset_status.setStyleSheet("color: #ffc107; font-size: 11px;")
        else:
            self.train_dataset_status.setText("[ERROR] 未找到data_config.yaml文件")
            self.train_dataset_status.setStyleSheet("color: #dc3545; font-size: 11px;")
    
    def auto_detect_image_size(self, dataset_dir):
        """自動檢測資料集中的圖片尺寸"""
        try:
            import cv2
            import numpy as np
            
            # 檢查訓練圖片
            train_dir = dataset_dir / 'images' / 'train'
            if train_dir.exists():
                # 尋找圖片文件
                image_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')) + list(train_dir.glob('*.npy'))
                
                if image_files:
                    # 讀取第一張圖片
                    first_image = image_files[0]
                    
                    if first_image.suffix == '.npy':
                        # 處理NPY文件（4通道）
                        image_data = np.load(first_image)
                        if len(image_data.shape) == 3:
                            height, width, channels = image_data.shape
                        else:
                            height, width = image_data.shape[:2]
                            channels = 1
                    else:
                        # 處理普通圖片文件
                        image = cv2.imread(str(first_image))
                        if image is not None:
                            height, width, channels = image.shape
                        else:
                            raise Exception("無法讀取圖片文件")
                    
                    # 更新GUI顯示
                    self.image_size_label.setText(f"{width}×{height}")
                    self.image_size_label.setStyleSheet("color: #28a745; font-size: 11px; padding: 4px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 3px;")
                    
                    self.log_message(f"[SEARCH] 檢測到圖片尺寸: {width}×{height}")
                    return True
                else:
                    self.image_size_label.setText("未找到圖片文件")
                    self.image_size_label.setStyleSheet("color: #ffc107; font-size: 11px; padding: 4px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 3px;")
                    return False
            else:
                self.image_size_label.setText("未找到訓練圖片目錄")
                self.image_size_label.setStyleSheet("color: #dc3545; font-size: 11px; padding: 4px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 3px;")
                return False
                
        except Exception as e:
            self.image_size_label.setText(f"檢測失敗: {str(e)}")
            self.image_size_label.setStyleSheet("color: #dc3545; font-size: 11px; padding: 4px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 3px;")
            self.log_message(f"[WARNING] 圖片尺寸檢測失敗: {e}")
            return False
    
    def update_custom_dataset_info(self):
        """更新RGBD訓練數據集信息"""
        # 從下拉選單獲取選擇的資料集路徑
        dataset_path = self.train_custom_dataset_combo.currentData()
        if not dataset_path:
            # 如果沒有data，嘗試從currentText獲取
            dataset_path = self.train_custom_dataset_combo.currentText()
        
        if not dataset_path:
            self.custom_image_size_label.setText("未檢測到")
            self.custom_image_size_label.setStyleSheet("color: #666666; font-size: 11px; padding: 4px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 3px;")
            return
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            self.custom_image_size_label.setText("資料夾不存在")
            self.custom_image_size_label.setStyleSheet("color: #dc3545; font-size: 11px; padding: 4px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 3px;")
            self.log_message(f"[ERROR] 自定義資料集路徑不存在: {dataset_path}")
            return
        
        # 自動檢測圖片尺寸
        self.auto_detect_custom_image_size(dataset_dir)
    
    def auto_detect_custom_image_size(self, dataset_dir):
        """自動檢測RGBD訓練數據集中的圖片尺寸"""
        try:
            import numpy as np
            
            self.log_message(f"[SEARCH] 開始檢測NPY圖片尺寸: {dataset_dir}")
            
            # 檢查標準的images/train目錄結構
            train_dir = dataset_dir / 'images' / 'train'
            if train_dir.exists():
                self.log_message(f"[FOLDER] 找到標準目錄結構: {train_dir}")
                npy_files = list(train_dir.glob('*.npy'))
            else:
                # 如果沒有標準結構，搜索整個目錄
                self.log_message(f"[FOLDER] 未找到標準目錄結構，搜索整個目錄: {dataset_dir}")
                npy_files = list(dataset_dir.glob('**/*.npy'))
            
            self.log_message(f"[CHART] 找到 {len(npy_files)} 個NPY文件")
            
            if npy_files:
                # 讀取第一個NPY文件
                first_npy = npy_files[0]
                self.log_message(f"📄 讀取NPY文件: {first_npy}")
                image_data = np.load(first_npy)
                
                if len(image_data.shape) == 3:
                    height, width, channels = image_data.shape
                else:
                    height, width = image_data.shape[:2]
                    channels = 1
                
                # 更新GUI顯示
                self.custom_image_size_label.setText(f"{width}×{height} ({channels}通道)")
                self.custom_image_size_label.setStyleSheet("color: #28a745; font-size: 11px; padding: 4px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 3px;")
                
                self.log_message(f"[SEARCH] 檢測到NPY圖片尺寸: {width}×{height}")
                return True
            else:
                self.custom_image_size_label.setText("未找到NPY文件")
                self.custom_image_size_label.setStyleSheet("color: #ffc107; font-size: 11px; padding: 4px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 3px;")
                self.log_message(f"[WARNING] 在 {dataset_dir} 中未找到NPY文件")
                return False
                
        except Exception as e:
            self.custom_image_size_label.setText(f"檢測失敗: {str(e)}")
            self.custom_image_size_label.setStyleSheet("color: #dc3545; font-size: 11px; padding: 4px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 3px;")
            self.log_message(f"[WARNING] NPY圖片尺寸檢測失敗: {e}")
            return False
     
    def browse_inference_model(self):
        """瀏覽推理模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型文件", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.inference_model_edit.setText(file_path)
    
    def _get_architecture_info(self):
        """獲取架構信息 - 從YAML目錄動態加載"""
        try:
            # 查找YAML目錄
            yaml_dir = Path("Model_file/yaml")
            
            if not yaml_dir.exists():
                self.log_message("[WARNING] 未找到YAML目錄")
                return {}
            
            self.log_message(f"[FOLDER] 使用YAML目錄: {yaml_dir}")
            
            # 查找YAML文件
            yaml_files = list(yaml_dir.glob("*.yaml"))
            if not yaml_files:
                self.log_message("[WARNING] YAML目錄中未找到YAML文件")
                return {}
            
            # 統計YAML目錄的文件
            self.log_message(f"[CHART] YAML目錄統計: 找到 {len(yaml_files)} 個YAML文件")
            
            # 嘗試加載YAML文件
            architectures = {}
            for yaml_file in yaml_files:
                try:
                    # 直接使用YAML文件名作為顯示名稱
                    yaml_name = yaml_file.stem  # 去掉.yaml擴展名
                    arch_type = yaml_name.lower().replace("yolo12", "")
                    
                    # 如果沒有提取到類型，使用默認
                    if not arch_type:
                        arch_type = "default"
                    
                    # 對於YAML文件，我們不需要獲取參數數量
                    param_count = 0
                    
                    architectures[arch_type] = {
                        "name": yaml_name,  # 直接顯示YAML文件名
                        "description": f"{yaml_name}架構",
                        "param_count": param_count,
                        "recommended": arch_type in ["n", "s", "m", "l", "x"],
                        "class_name": yaml_name,
                        "file": yaml_file.name
                    }
                    
                except Exception as e:
                    self.log_message(f"[WARNING] 加載{yaml_file}失敗: {e}")
                    continue
            
            if architectures:
                self.log_message(f"[OK] 從YAML目錄加載了 {len(architectures)} 個架構")
                return architectures
            else:
                self.log_message("[WARNING] 未找到有效架構")
                return {}
                
        except Exception as e:
            self.log_message(f"[ERROR] 加載YAML架構失敗: {e}")
            return {}
    
    def _get_model_param_count(self, pt_file):
        """獲取模型參數數量"""
        try:
            import torch
            
            # 嘗試使用weights_only=False加載（適用於PyTorch 2.6+）
            try:
                checkpoint = torch.load(pt_file, map_location='cpu', weights_only=False)
            except Exception:
                # 如果失敗，嘗試默認方式
                checkpoint = torch.load(pt_file, map_location='cpu')
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint
            
            # 計算參數數量 - 處理不同類型的模型對象
            if hasattr(model_state, 'parameters'):
                # 如果是模型對象，使用parameters()方法
                total_params = sum(p.numel() for p in model_state.parameters())
            elif hasattr(model_state, 'values'):
                # 如果是字典，使用values()方法
                total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
            elif isinstance(model_state, dict):
                # 如果是字典，遍歷所有值
                total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
            else:
                # 其他情況，嘗試直接計算
                total_params = 0
                for key, value in model_state.items() if hasattr(model_state, 'items') else []:
                    if isinstance(value, torch.Tensor):
                        total_params += value.numel()
            
            # 格式化參數數量
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.1f}M"
            elif total_params >= 1e3:
                return f"{total_params/1e3:.1f}K"
            else:
                return f"{total_params:,}"
                
        except Exception as e:
            # 靜默處理參數數量獲取錯誤
            return "未知"
    
    def _load_architecture_options(self):
        """動態加載架構選項"""
        try:
            # 防止重複加載
            if hasattr(self, '_loading_architectures') and self._loading_architectures:
                return
            self._loading_architectures = True
            
            # 清空現有選項
            self.train_custom_arch_combo.clear()
            
            # 獲取架構信息
            arch_info = self._get_architecture_info()
            # 更新緩存
            self._cached_arch_info = arch_info
            
            if not arch_info:
                # 如果沒有架構信息，添加默認選項
                self.train_custom_arch_combo.addItem("無可用架構", "default")
                # 只有在log_text存在時才記錄日誌
                if hasattr(self, 'log_text'):
                    self.log_message("[WARNING] 未找到可用的架構選項")
                return
            
            # 添加架構選項 - 按照nsmlx順序排序
            recommended_arch = None
            
            # 定義nsmlx順序
            nsmlx_order = ['n', 's', 'm', 'l', 'x']
            
            # 先添加nsmlx順序的架構
            for arch_type in nsmlx_order:
                if arch_type in arch_info:
                    info = arch_info[arch_type]
                    display_text = info['name']  # 顯示PT文件名
                    if info['recommended']:
                        if recommended_arch is None:
                            recommended_arch = arch_type
                    
                    self.train_custom_arch_combo.addItem(display_text, arch_type)
            
            # 再添加其他架構（不在nsmlx順序中的）
            for arch_type, info in arch_info.items():
                if arch_type not in nsmlx_order:
                    display_text = info['name']  # 顯示PT文件名
                    if info['recommended']:
                        if recommended_arch is None:
                            recommended_arch = arch_type
                    
                    self.train_custom_arch_combo.addItem(display_text, arch_type)
            
            # 默認選擇推薦的架構
            if recommended_arch:
                # 找到推薦架構的索引
                for i in range(self.train_custom_arch_combo.count()):
                    if self.train_custom_arch_combo.itemData(i) == recommended_arch:
                        self.train_custom_arch_combo.setCurrentIndex(i)
                        break
            
            # 只有在log_text存在時才記錄日誌
            if hasattr(self, 'log_text'):
                self.log_message(f"[OK] 加載了 {len(arch_info)} 個架構選項")
            
        except Exception as e:
            # 只有在log_text存在時才記錄日誌
            if hasattr(self, 'log_text'):
                self.log_message(f"[ERROR] 加載架構選項失敗: {e}")
            # 添加默認選項
            self.train_custom_arch_combo.addItem("默認架構", "default")
        finally:
            # 重置加載標誌
            self._loading_architectures = False
    
    def update_arch_description(self):
        """更新架構描述"""
        # 检查是否存在架构组合框
        if not hasattr(self, 'train_custom_arch_combo'):
            return
            
        current_index = self.train_custom_arch_combo.currentIndex()
        if current_index >= 0:
            arch_type = self.train_custom_arch_combo.itemData(current_index)
            # 使用緩存的架構信息，避免重複加載
            if not hasattr(self, '_cached_arch_info'):
                self._cached_arch_info = self._get_architecture_info()
            arch_info = self._cached_arch_info.get(arch_type, {})
            
            if arch_info:
                name = arch_info.get('name', '')
                description = arch_info.get('description', '')
                param_count = arch_info.get('param_count', '')
                file_name = arch_info.get('file', '')
                
                # 構建詳細描述
                desc_text = f"📝 {description}"
                if param_count:
                    desc_text += f"\n🔢 參數數量: {param_count}"
                if file_name:
                    desc_text += f"\n[FOLDER] 來源: {file_name}"
                
                self.arch_desc_label.setText(desc_text)
            else:
                self.arch_desc_label.setText("📝 架構信息不可用")
    
    def on_architecture_changed(self):
        """架構選擇改變時的互斥邏輯"""
        # 如果選擇了架構，清空預訓練模型
        if self.train_custom_arch_combo.currentText():
            self.train_custom_model_edit.clear()
            self.train_custom_model_status.setText("已選擇架構，預訓練模型已清空")
            self.train_custom_model_status.setStyleSheet("color: #ff6b6b; font-size: 11px;")
    
    def on_pretrained_model_changed(self):
        """預訓練模型選擇改變時的互斥邏輯"""
        # 如果選擇了預訓練模型，清空架構選擇
        if self.train_custom_model_edit.text().strip():
            self.train_custom_arch_combo.setCurrentIndex(-1)  # 設置為空白
            self.arch_desc_label.setText("已選擇預訓練模型，架構選擇已清空")
            self.arch_desc_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        else:
            # 如果清空了預訓練模型，恢復架構描述樣式
            self.arch_desc_label.setStyleSheet("color: #666666; font-size: 11px;")
            self.update_arch_description()
    
    def auto_find_dataset(self):
        """自動尋找數據集"""
        try:
            # 簡單的數據集查找 - 掃描Dataset目錄
            dataset_root = Path("Dataset")
            if not dataset_root.exists():
                self.log_message("[ERROR] Dataset目錄不存在")
                return False
            
            # 查找所有數據集目錄
            dataset_dirs = list(dataset_root.glob("dataset_*"))
            if not dataset_dirs:
                self.log_message("[ERROR] 未找到數據集目錄")
                return False
            
            # 選擇最新的數據集
            latest_dataset = max(dataset_dirs, key=lambda x: x.stat().st_mtime)
            self.check_dataset_edit.setText(str(latest_dataset))
            self.log_message(f"[OK] 找到數據集: {latest_dataset.name}")
            return True
                
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"自動檢測數據集時發生錯誤: {str(e)}")
            return False
   
    # 任務控制方法
    def _toggle_convert_buttons(self, is_running):
        """切換轉換按鈕狀態 - 统一的按钮管理 (Unified button management)"""
        self.convert_start_btn.setEnabled(not is_running)
        self.convert_stop_btn.setEnabled(is_running)
        self.show_progress(is_running)
    
    def _get_conversion_mode_info(self, use_depth, use_stereo):
        """獲取轉換模式信息 - 避免重复逻辑 (Avoid duplicate logic)"""
        if use_stereo:
            return "立體視覺數據 Stereo Vision Data", "🔄 開始立體視覺數據轉換... Starting stereo data conversion..."
        elif use_depth:
            return "4通道RGBD數據 4-Channel RGBD Data", "🔄 開始4通道數據轉換... Starting 4-channel data conversion..."
        else:
            return "3通道RGB數據 3-Channel RGB Data", "🔄 開始3通道數據轉換... Starting 3-channel data conversion..."
    
    def start_convert(self):
        """開始數據轉換"""
        # 驗證源路徑
        source_path = self._validate_source_path(self.convert_source_edit.text())
        if not source_path:
            return
        
        # 切換按鈕狀態
        self._toggle_convert_buttons(True)
        
        # 獲取深度圖選項
        use_depth = self.use_depth_radio.isChecked()
        use_stereo = self.stereo_radio.isChecked()
        
        # 獲取資料夾數量限制
        folder_count_limit = self.folder_count_spin.value()
        # 現在直接使用選擇的數量，不需要特殊處理0值
        
        # 創建工作線程
        self.worker_thread = WorkerThread(
            "convert",
            source_path=self.convert_source_edit.text(),
            output_path=self.convert_output_edit.text() if self.convert_output_edit.text() else None,
            use_depth=use_depth,
            use_stereo=use_stereo,
            folder_count_limit=folder_count_limit
        )
        self.worker_thread.progress.connect(self.update_status)
        self.worker_thread.finished.connect(self.on_convert_finished)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.start()
    
    def stop_convert(self):
        """停止數據轉換"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        
        self._toggle_convert_buttons(False)
        self.log_message("⏹️ 數據轉換已停止 Data conversion stopped")

    def start_training(self):
        """開始標準訓練"""
        # 保存當前設置（包括資料集和模型選擇）
        self.save_settings()
        
        # 在選擇資料集進入訓練前，先刷新一次模型列表
        self.log_message("🔄 開始訓練前刷新模型列表...")
        self.refresh_model_list()
        self.log_message("[OK] 模型列表刷新完成")
        
        # 刷新後重新恢復上次使用的選擇
        self.log_message("🔄 恢復上次使用的資料集和模型選擇...")
        try:
            import yaml
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = yaml.safe_load(f)
                if settings and 'standard_training' in settings:
                    self._restore_last_used_selections(settings['standard_training'])
                    self.log_message("[OK] 已恢復上次使用的選擇")
        except Exception as e:
            self.log_message(f"[WARNING] 恢復上次選擇失敗: {e}")
        
        # 從下拉選單獲取選擇的資料集路徑
        dataset_path = self.train_dataset_combo.currentData()
        if not dataset_path:
            # 如果沒有data，嘗試從currentText獲取
            dataset_path = self.train_dataset_combo.currentText()
        
        if not dataset_path:
            self.log_message("[WARNING] 請選擇資料集")
            # 不再弹出警告窗口，只记录日志
            # QMessageBox.warning(self, "警告", "請選擇資料集")
            return
        
        # 檢查資料集資料夾是否存在
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            self.log_message("[WARNING] 資料集資料夾不存在，請檢查路徑是否正確")
            # 不再弹出警告窗口，只记录日志
            # QMessageBox.warning(self, "警告", "資料集資料夾不存在，請檢查路徑是否正確")
            return
        
        # 檢查是否包含data_config.yaml
        config_file = dataset_path / "data_config.yaml"
        if not config_file.exists():
            self.log_message("[WARNING] 資料集中未找到data_config.yaml文件")
            # 不再弹出警告窗口，只记录日志
            # QMessageBox.warning(self, "警告", "資料集中未找到data_config.yaml文件")
            return
        
        # 讀取配置文件信息
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 顯示配置信息
            nc = config.get('nc', 1)
            names = config.get('names', [])
            channels = config.get('channels', 3)
            
            # 如果配置文件中沒有類別數量，嘗試從預定義類別獲取
            if nc == 1 and not names:
                try:
                    from config.predefined_classes import load_predefined_classes
                    predefined_classes = load_predefined_classes()
                    nc = len(predefined_classes)
                    names = predefined_classes
                    self.log_message(f"📋 從預定義類別獲取類別信息: {nc} 個類別")
                except Exception as e:
                    self.log_message(f"⚠️ 無法載入預定義類別: {e}")
                    nc = self.get_dynamic_class_count()
            
            self.log_message(f"📋 訓練配置信息:")
            self.log_message(f"   類別數量: {nc}")
            self.log_message(f"   類別名稱: {names}")
            self.log_message(f"   通道數: {channels}")
            
        except Exception as e:
            self.log_message(f"[WARNING] 讀取配置文件失敗: {e}")
        
        selected_config = str(config_file)
        
        # 禁用按鈕
        self.train_start_btn.setEnabled(False)
        self.train_stop_btn.setEnabled(True)
        self.show_progress(True)
        
        # 根據訓練模式確定模型路徑
        training_mode = 'retrain' if self.retrain_radio.isChecked() else 'pretrained'
        
        if training_mode == 'retrain':
            # 重新訓練模式 - 使用YAML配置文件
            if hasattr(self, 'model_file_combo') and self.model_file_combo.currentData():
                selected_model = self.model_file_combo.currentData()
                # 獲取模型大小
                model_size = self.train_model_size_combo.currentText().split()[0] if hasattr(self, 'train_model_size_combo') else 'n'
                
                self.log_message(f"📋 重新訓練模式 - YAML配置: {selected_model}")
                self.log_message(f"📋 重新訓練模式 - 模型大小: {model_size}")
                self.log_message(f"📋 重新訓練模式 - 將訓練為: {Path(selected_model).stem}{model_size} 模型")
            else:
                # 如果沒有選擇YAML文件，使用默認的yolo12.yaml
                selected_model = "Model_file/YAML/yolo12.yaml"
                model_size = self.train_model_size_combo.currentText().split()[0] if hasattr(self, 'train_model_size_combo') else 'n'
                self.log_message(f"📋 重新訓練模式 - 使用默認配置: {selected_model}")
                self.log_message(f"📋 重新訓練模式 - 模型大小: {model_size}")
                self.log_message(f"📋 重新訓練模式 - 將訓練為: yolo12{model_size} 模型")
        else:
            # 預訓練模式 - 使用PT模型文件
            if hasattr(self, 'model_file_combo') and self.model_file_combo.currentData():
                selected_model = self.model_file_combo.currentData()
                self.log_message(f"📋 預訓練模式 - 使用PT模型: {selected_model}")
            else:
                # 如果沒有選擇PT文件，使用默認模型
                selected_model = "Model_file/PT_File/yolov12n.pt"
                self.log_message(f"📋 預訓練模式 - 使用默認PT模型: {selected_model}")
        
        # 檢查模型文件是否存在
        if not Path(selected_model).exists():
            QMessageBox.warning(self, "警告", f"模型文件不存在: {selected_model}")
            return
        
        # 添加調試信息
        self.log_message(f"[SEARCH] 選中的模型路徑: {selected_model}")
        self.log_message(f"[SEARCH] 模型文件存在: {Path(selected_model).exists()}")
        self.log_message(f"[SEARCH] 當前工作目錄: {Path.cwd()}")
        
        if Path(selected_model).exists():
            file_size = Path(selected_model).stat().st_size / (1024 * 1024)
            self.log_message(f"[SEARCH] 模型文件大小: {file_size:.1f} MB")
            self.log_message(f"[SEARCH] 模型絕對路徑: {Path(selected_model).resolve()}")
        
        # 獲取訓練參數
        epochs = self.epochs_spin.value()
        learning_rate = self.learning_rate_spin.value() * 0.001  # 轉換為實際學習率
        batch_size = self.batch_size_spin.value()
        
        # 新增的高級訓練參數
        imgsz = self.imgsz_spin.value()
        save_period = self.save_period_spin.value()
        scale = self.scale_spin.value() * 0.01  # 轉換為實際縮放比例
        mosaic = self.mosaic_spin.value() * 0.01  # 轉換為實際Mosaic值
        mixup = self.mixup_spin.value() * 0.01  # 轉換為實際Mixup值
        copy_paste = self.copy_paste_spin.value() * 0.01  # 轉換為實際Copy-paste值
        
        # 新增的HSV和BGR增強參數
        hsv_h = self.hsv_h_spin.value() * 0.01  # 轉換為實際HSV色相值
        hsv_s = self.hsv_s_spin.value() * 0.01  # 轉換為實際HSV飽和度值
        hsv_v = self.hsv_v_spin.value() * 0.01  # 轉換為實際HSV明度值
        bgr = self.bgr_spin.value() * 0.01  # 轉換為實際BGR值
        auto_augment = self.auto_augment_combo.currentData()  # 獲取自動增強策略
        
        # 新增的幾何變換參數
        degrees = self.degrees_spin.value()  # 旋轉角度
        translate = self.translate_spin.value() * 0.01  # 平移距離
        shear = self.shear_spin.value() * 0.01  # 剪切角度
        perspective = self.perspective_spin.value() * 0.01  # 透視變換
        
        # 新增的翻轉和裁剪參數
        flipud = self.flipud_spin.value() * 0.01  # 上下翻轉
        fliplr = self.fliplr_spin.value() * 0.01  # 左右翻轉
        erasing = self.erasing_spin.value() * 0.01  # 隨機擦除
        crop_fraction = self.crop_fraction_spin.value() * 0.01  # 裁剪比例
        
        # 新增的訓練控制參數
        close_mosaic = self.close_mosaic_spin.value()  # 關閉Mosaic
        workers = self.workers_spin.value()  # 工作進程
        optimizer = self.optimizer_combo.currentText()  # 優化器
        amp = self.amp_checkbox.isChecked()  # AMP混合精度
        
        self.log_message(f"🎯 訓練參數: 輪數={epochs}, 學習率={learning_rate}, 批次大小={batch_size}")
        self.log_message(f"[CHART] 高級參數: 圖像大小={imgsz}, 縮放={scale}, Mosaic={mosaic}, Mixup={mixup}, Copy-paste={copy_paste}")
        self.log_message(f"🎨 增強參數: HSV色相={hsv_h}, HSV飽和度={hsv_s}, HSV明度={hsv_v}, BGR={bgr}, 自動增強={auto_augment}")
        self.log_message(f"🔄 幾何變換: 旋轉={degrees}°, 平移={translate}, 剪切={shear}, 透視={perspective}")
        self.log_message(f"🔄 翻轉裁剪: 上下翻轉={flipud}, 左右翻轉={fliplr}, 擦除={erasing}, 裁剪={crop_fraction}")
        self.log_message(f"⚙️ 訓練控制: 關閉Mosaic={close_mosaic}, 工作進程={workers}, 優化器={optimizer}, AMP={amp}")
        
        # 獲取訓練模式
        training_mode = 'retrain' if self.retrain_radio.isChecked() else 'pretrained'
        self.log_message(f"🎯 訓練模式: {'重新訓練 (YAML)' if training_mode == 'retrain' else '預訓練模型 (PT)'}")
        
        # 獲取模型大小參數（僅在重新訓練模式下需要）
        model_size = None
        if training_mode == 'retrain' and hasattr(self, 'train_model_size_combo'):
            model_size = self.train_model_size_combo.currentText().split()[0]
            self.log_message(f"📋 模型大小參數: {model_size}")
        
        # 創建工作線程
        self.worker_thread = WorkerThread(
            "train",
            config_path=selected_config,
            model_file=selected_model,
            training_mode=training_mode,
            model_size=model_size,  # 添加模型大小參數
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
            # 新增的幾何變換參數
            degrees=degrees,
            translate=translate,
            shear=shear,
            perspective=perspective,
            # 新增的翻轉和裁剪參數
            flipud=flipud,
            fliplr=fliplr,
            erasing=erasing,
            crop_fraction=crop_fraction,
            # 新增的訓練控制參數
            close_mosaic=close_mosaic,
            workers=workers,
            optimizer=optimizer,
            amp=amp
        )
        self.worker_thread.progress.connect(self.update_status)
        self.worker_thread.finished.connect(self.on_training_finished)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.start()
    
    def stop_training(self):
        """停止標準訓練"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.show_progress(False)
        self.log_message("⏹️ 訓練已停止")
    
    def start_inference(self):
        """開始推理"""
        if not self.inference_model_edit.text():
            QMessageBox.warning(self, "警告", "請選擇模型文件")
            return
        
        # 檢查模型文件是否存在
        model_path = Path(self.inference_model_edit.text())
        if not model_path.exists():
            QMessageBox.warning(self, "警告", "模型文件不存在，請檢查路徑是否正確")
            return
        
        # 禁用按鈕
        self.inference_start_btn.setEnabled(False)
        self.inference_stop_btn.setEnabled(True)
        self.inference_test_btn.setEnabled(False)
        self.show_progress(True)
        
        # 獲取推理參數
        confidence_threshold = self.inference_confidence_spin.value() / 100.0
        num_classes = self.inference_num_classes_spin.value()
        inference_mode = self.inference_mode_combo.currentText()
        
        # 獲取高級推理參數
        iou_threshold = self.inference_iou_spin.value() / 100.0
        max_det = self.inference_max_det_spin.value()
        line_width = self.inference_line_width_spin.value()
        show_labels = self.inference_show_labels_check.isChecked()
        show_conf = self.inference_show_conf_check.isChecked()
        show_boxes = self.inference_show_boxes_check.isChecked()
        save_txt = self.inference_save_txt_check.isChecked()
        save_conf = self.inference_save_conf_check.isChecked()
        save_crop = self.inference_save_crop_check.isChecked()
        visualize = self.inference_visualize_check.isChecked()
        augment = self.inference_augment_check.isChecked()
        agnostic_nms = self.inference_agnostic_nms_check.isChecked()
        retina_masks = self.inference_retina_masks_check.isChecked()
        output_format = self.inference_format_combo.currentText()
        verbose = self.inference_verbose_check.isChecked()
        show = self.inference_show_check.isChecked()
        
        # 根據模式獲取數據集路徑
        dataset_path = None
        if inference_mode == "數據集測試模式":
            dataset_path = self.dataset_path_edit.text().strip()
            if not dataset_path:
                dataset_path = None  # 自動查找最新數據集
        
        # 創建工作線程
        self.worker_thread = WorkerThread(
            "inference",
            model_path=self.inference_model_edit.text(),
            confidence_threshold=confidence_threshold,
            num_classes=num_classes,
            inference_mode=inference_mode,
            dataset_path=dataset_path,
            # 高級推理參數
            iou_threshold=iou_threshold,
            max_det=max_det,
            line_width=line_width,
            show_labels=show_labels,
            show_conf=show_conf,
            show_boxes=show_boxes,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            visualize=visualize,
            augment=augment,
            agnostic_nms=agnostic_nms,
            retina_masks=retina_masks,
            output_format=output_format,
            verbose=verbose,
            show=show
        )
        self.worker_thread.progress.connect(self.update_status)
        self.worker_thread.finished.connect(self.on_inference_finished)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.start()
    
    def stop_inference(self):
        """停止推理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        
        self.inference_start_btn.setEnabled(True)
        self.inference_stop_btn.setEnabled(False)
        self.inference_test_btn.setEnabled(True)
        self.show_progress(False)
        self.log_message("⏹️ 推理已停止")
    
    def on_inference_mode_changed(self, mode):
        """推理模式變化處理"""
        if mode == "數據集測試模式":
            self.dataset_group.setVisible(True)
        else:
            self.dataset_group.setVisible(False)
    
    def browse_inference_dataset(self):
        """瀏覽推理數據集"""
        dataset_path = QFileDialog.getExistingDirectory(
            self, 
            "選擇數據集目錄", 
            "Dataset",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if dataset_path:
            self.dataset_path_edit.setText(dataset_path)
            self.log_message(f"[FOLDER] 選擇數據集: {dataset_path}")
    
    def run_inference_test(self):
        """運行推理測試"""
        try:
            self.log_message("🧪 開始推理測試...")
            
            # 檢查模型文件
            model_path = self.inference_model_edit.text()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "警告", "請先選擇有效的模型文件")
                return
            
            # 創建測試工作線程
            self.worker_thread = WorkerThread(
                "inference_test",
                model_path=model_path,
                confidence_threshold=self.inference_confidence_spin.value() / 100.0,
                architecture_type=self.inference_architecture_combo.currentText(),
                num_classes=self.inference_num_classes_spin.value()
            )
            self.worker_thread.progress.connect(self.update_status)
            self.worker_thread.finished.connect(self.on_inference_test_finished)
            self.worker_thread.log_message.connect(self.log_message)
            self.worker_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ 推理測試失敗: {e}")
            QMessageBox.critical(self, "錯誤", f"推理測試失敗：{e}")
    
    def on_inference_test_finished(self, success, message):
        """推理測試完成回調"""
        if success:
            self.log_message("✅ 推理測試完成")
            QMessageBox.information(self, "測試成功", "推理測試完成！推理器功能正常。")
        else:
            self.log_message(f"❌ 推理測試失敗: {message}")
            QMessageBox.critical(self, "測試失敗", f"推理測試失敗：{message}")

    def check_data_directory(self):
        """檢查Predict/Data目錄"""
        data_dir = Path("Predict/Data")
        if not data_dir.exists():
            QMessageBox.information(self, "信息", "Predict/Data目錄不存在，將自動創建")
            data_dir.mkdir(parents=True, exist_ok=True)
            self.log_message("[FOLDER] 已創建Predict/Data目錄")
            return
        
        # 統計文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        gif_files = list(data_dir.glob("*.gif")) + list(data_dir.glob("*.GIF"))
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_dir.glob(f"*{ext}"))
            image_files.extend(data_dir.glob(f"*{ext.upper()}"))
        
        # 去重複
        image_files = list(set(image_files))
        gif_files = list(set(gif_files))
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(data_dir.glob(f"*{ext}"))
            video_files.extend(data_dir.glob(f"*{ext.upper()}"))
        
        # 去重複
        video_files = list(set(video_files))
        
        # 統計NPY文件
        npy_files = list(data_dir.glob("*.npy")) + list(data_dir.glob("*.NPY"))
        npy_files = list(set(npy_files))  # 去重複
        
        message = f"Predict/Data目錄文件統計:\n"
        message += f"圖片: {len(image_files)} 個\n"
        message += f"NPY: {len(npy_files)} 個\n"
        message += f"GIF: {len(gif_files)} 個\n"
        message += f"影片: {len(video_files)} 個\n"
        message += f"總計: {len(image_files) + len(npy_files) + len(gif_files) + len(video_files)} 個文件"
        
        QMessageBox.information(self, "Predict/Data目錄檢查", message)
        self.log_message(f"[CHART] Predict/Data目錄檢查: {len(image_files)} 圖片, {len(npy_files)} NPY, {len(gif_files)} GIF, {len(video_files)} 影片")
    
    # 任務完成回調
    def on_convert_finished(self, success, message):
        """數據轉換完成回調 - 优化版本 (Optimized callback)"""
        # 統一的按鈕狀態管理
        self._toggle_convert_buttons(False)
        
        if success:
            QMessageBox.information(self, "成功 Success", "數據轉換完成！Data conversion completed!")
            self.update_status("數據轉換完成 Data conversion completed")
            # 轉換完成後自動更新配置文件路徑
            self.auto_load_configs()
        else:
            QMessageBox.critical(self, "錯誤 Error", f"數據轉換失敗 Failed：{message}")
            self.update_status("數據轉換失敗 Data conversion failed")
     
    def on_training_finished(self, success, message):
        """訓練完成回調"""
        # 恢復按鈕狀態
        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.show_progress(False)
        
        if success:
            QMessageBox.information(self, "成功", f"模型訓練完成！\n{message}")
            self.update_status("訓練完成")
            self.log_message("[SUCCESS] 訓練成功完成！")
            self.log_message("🔧 train_batch 可視化已自動修復 - 生成 train_batch0_fixed.jpg")
        else:
            QMessageBox.critical(self, "錯誤", f"訓練失敗：{message}")
            self.update_status("訓練失敗")
            self.log_message(f"[ERROR] 訓練失敗: {message}")
    
    def on_inference_finished(self, success, message):
        """推理完成回調"""
        self.inference_start_btn.setEnabled(True)
        self.inference_stop_btn.setEnabled(False)
        self.inference_test_btn.setEnabled(True)
        self.show_progress(False)
        
        if success:
            QMessageBox.information(self, "成功", "推理完成！")
            self.update_status("推理完成")
            self.log_message("✅ 推理完成")
            self.log_message(f"[FOLDER] 結果保存在: Predict/Result/")
        else:
            QMessageBox.critical(self, "錯誤", f"推理失敗：{message}")
            self.update_status("推理失敗")
            self.log_message(f"❌ 推理失敗: {message}")
    
    # 日誌控制方法
    def clear_log(self):
        """清空日誌"""
        self.log_text.clear()
        self.log_message("[DELETE] 日誌已清空")
    
    def save_log(self):
        """保存日誌"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日誌", f"yolo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt)"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            self.log_message(f"💾 日誌已保存到: {file_path}")
    
    # 模型分析方法
    def refresh_analyzer_model_list(self):
        """刷新分析器模型列表 - 支援.pt、.pth和.yaml檔案"""
        self.log_message("🔄 刷新分析器模型列表...")
        
        # 清空現有列表
        self.analyzer_model_combo.clear()
        
        # 掃描所有模型文件 (.pt, .pth, .yaml)
        model_files = []
        
        # 掃描Model_file目錄
        model_dir = Path("Model_file")
        if model_dir.exists():
            # 掃描根目錄的文件
            for ext in ["*.pt", "*.pth", "*.yaml", "*.yml"]:
                for model_file in model_dir.glob(ext):
                    if model_file.suffix == ".pt":
                        file_type = "PT"
                    elif model_file.suffix == ".pth":
                        file_type = "PTH"
                    else:
                        file_type = "YAML"
                    model_files.append(("Model_file", model_file.name, str(model_file), file_type))
            
            # 掃描子目錄的文件
            for subdir in model_dir.iterdir():
                if subdir.is_dir():
                    for ext in ["*.pt", "*.pth", "*.yaml", "*.yml"]:
                        for model_file in subdir.glob(ext):
                            if model_file.suffix == ".pt":
                                file_type = "PT"
                            elif model_file.suffix == ".pth":
                                file_type = "PTH"
                            else:
                                file_type = "YAML"
                            model_files.append((f"Model_file/{subdir.name}", model_file.name, str(model_file), file_type))
        
        # 掃描根目錄的文件
        for ext in ["*.pt", "*.pth", "*.yaml", "*.yml"]:
            for model_file in Path(".").glob(ext):
                if model_file.suffix == ".pt":
                    file_type = "PT"
                elif model_file.suffix == ".pth":
                    file_type = "PTH"
                else:
                    file_type = "YAML"
                model_files.append(("根目錄", model_file.name, str(model_file), file_type))
        
        # 儲存所有模型文件以供篩選使用
        self.all_model_files = model_files
        
        # 應用檔案類型篩選
        self.apply_file_type_filter()
    
    def apply_file_type_filter(self):
        """根據選擇的檔案類型篩選模型列表"""
        if not hasattr(self, 'all_model_files'):
            return
        
        # 清空現有列表
        self.analyzer_model_combo.clear()
        
        # 獲取選擇的檔案類型
        selected_type = self.analyzer_file_type_combo.currentText()
        
        # 篩選模型文件
        if selected_type == "全部":
            filtered_files = self.all_model_files
        else:
            filtered_files = [f for f in self.all_model_files if f[3] == selected_type]
        
        # 按檔案類型排序，.pt檔案優先
        filtered_files.sort(key=lambda x: (x[3] != "PT", x[1].lower()))
        
        # 添加到下拉選單
        for category, filename, full_path, file_type in filtered_files:
            display_text = f"{category}/{filename} ({file_type})"
            self.analyzer_model_combo.addItem(display_text, full_path)
        
        if filtered_files:
            pt_count = sum(1 for _, _, _, file_type in filtered_files if file_type == "PT")
            pth_count = sum(1 for _, _, _, file_type in filtered_files if file_type == "PTH")
            yaml_count = sum(1 for _, _, _, file_type in filtered_files if file_type == "YAML")
            self.log_message(f"[OK] 找到 {len(filtered_files)} 個模型文件 ({pt_count} .pt, {pth_count} .pth, {yaml_count} .yaml)")
            # 自動選擇第一個模型
            self.update_analyzer_model_info()
        else:
            self.log_message(f"[WARNING] 未找到 {selected_type} 類型的模型文件")
            self.analyzer_model_status.setText(f"[ERROR] 未找到 {selected_type} 類型的模型文件")
    
    def browse_analyzer_model_folder(self):
        """瀏覽其他資料夾中的模型文件"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "選擇包含模型文件的資料夾"
        )
        if folder_path:
            self.log_message(f"[FOLDER] 選擇資料夾: {folder_path}")
            self.scan_custom_folder_for_models(folder_path)
    
    def scan_custom_folder_for_models(self, folder_path):
        """掃描自定義資料夾中的模型文件 - 支援.pt、.pth和.yaml檔案"""
        self.log_message(f"[SEARCH] 掃描資料夾: {folder_path}")
        
        # 掃描所有支援的模型文件類型
        model_files = []
        folder_path = Path(folder_path)
        
        if folder_path.exists():
            # 掃描根目錄的模型文件
            for ext in ["*.pt", "*.pth", "*.yaml", "*.yml"]:
                for model_file in folder_path.glob(ext):
                    if model_file.suffix == ".pt":
                        file_type = "PT"
                    elif model_file.suffix == ".pth":
                        file_type = "PTH"
                    else:
                        file_type = "YAML"
                    model_files.append(("根目錄", model_file.name, str(model_file), file_type))
            
            # 掃描子目錄的模型文件
            for subdir in folder_path.iterdir():
                if subdir.is_dir():
                    for ext in ["*.pt", "*.pth", "*.yaml", "*.yml"]:
                        for model_file in subdir.glob(ext):
                            if model_file.suffix == ".pt":
                                file_type = "PT"
                            elif model_file.suffix == ".pth":
                                file_type = "PTH"
                            else:
                                file_type = "YAML"
                            relative_path = subdir.relative_to(folder_path)
                            model_files.append((f"{relative_path}", model_file.name, str(model_file), file_type))
        
        # 儲存所有模型文件以供篩選使用
        self.all_model_files = model_files
        
        # 應用檔案類型篩選
        self.apply_file_type_filter()
        
        if model_files:
            pt_count = sum(1 for _, _, _, file_type in model_files if file_type == "PT")
            pth_count = sum(1 for _, _, _, file_type in model_files if file_type == "PTH")
            yaml_count = sum(1 for _, _, _, file_type in model_files if file_type == "YAML")
            self.log_message(f"[OK] 在資料夾中找到 {len(model_files)} 個模型文件 ({pt_count} .pt, {pth_count} .pth, {yaml_count} .yaml)")
        else:
            self.log_message("[WARNING] 在指定資料夾中未找到任何模型文件")
            self.analyzer_model_status.setText("[ERROR] 在指定資料夾中未找到模型文件")
    
    def update_analyzer_model_info(self):
        """更新分析器選中模型的信息"""
        if self.analyzer_model_combo.count() == 0:
            return
        
        current_text = self.analyzer_model_combo.currentText()
        model_path = self.analyzer_model_combo.currentData()
        
        if model_path and Path(model_path).exists():
            try:
                # 嘗試獲取詳細模型信息
                import sys
                code_path = Path(__file__).parent / 'Code'
                if str(code_path) not in sys.path:
                    sys.path.insert(0, str(code_path))
                
                try:
                    import Read_Model  # type: ignore
                    model_summary = Read_Model.get_model_summary(model_path)
                    self.analyzer_model_status.setText(f"✅ {model_summary}")
                    self.analyzer_model_status.setStyleSheet("color: #28a745; font-size: 11px;")
                    self.log_message(f"✅ 選中分析模型: {model_summary}")
                except ImportError:
                    # 如果無法導入Read_Model，顯示基本信息
                    file_size = Path(model_path).stat().st_size / (1024 * 1024)
                    self.analyzer_model_status.setText(f"✅ 模型存在: {current_text} ({file_size:.1f} MB)")
                    self.analyzer_model_status.setStyleSheet("color: #28a745; font-size: 11px;")
                    self.log_message(f"✅ 選中分析模型: {current_text} ({file_size:.1f} MB)")
                except Exception as e:
                    # 如果分析失敗，顯示基本信息
                    file_size = Path(model_path).stat().st_size / (1024 * 1024)
                    self.analyzer_model_status.setText(f"⚠️ {current_text} ({file_size:.1f} MB) - 分析失敗")
                    self.analyzer_model_status.setStyleSheet("color: #ffc107; font-size: 11px;")
                    self.log_message(f"⚠️ 選中分析模型: {current_text} ({file_size:.1f} MB) - 分析失敗: {e}")
            except Exception:
                # 如果出現任何錯誤，顯示基本信息
                file_size = Path(model_path).stat().st_size / (1024 * 1024)
                self.analyzer_model_status.setText(f"✅ 模型存在: {current_text} ({file_size:.1f} MB)")
                self.analyzer_model_status.setStyleSheet("color: #28a745; font-size: 11px;")
                self.log_message(f"✅ 選中分析模型: {current_text} ({file_size:.1f} MB)")
        else:
            self.analyzer_model_status.setText(f"❌ 模型文件不存在: {current_text}")
            self.analyzer_model_status.setStyleSheet("color: #dc3545; font-size: 11px;")
            self.log_message(f"❌ 分析模型文件不存在: {current_text}")
    
    def analyze_selected_model(self):
        """分析選中的模型"""
        if self.analyzer_model_combo.count() == 0:
            QMessageBox.warning(self, "警告", "請先刷新模型列表")
            return
        
        model_path = self.analyzer_model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "警告", "請選擇一個模型文件")
            return
        
        if not Path(model_path).exists():
            QMessageBox.warning(self, "警告", f"模型文件不存在: {model_path}")
            return
        
        self.log_message(f"🔬 開始分析模型: {Path(model_path).name}")
        self.analyzer_results.clear()
        self.analyzer_results.append("🔬 模型分析開始...")
        self.analyzer_results.append("=" * 50)
        
        try:
            # 導入Read_Model.py的功能
            import sys
            code_path = Path(__file__).parent / 'Code'
            if str(code_path) not in sys.path:
                sys.path.insert(0, str(code_path))
            
            # 動態導入Read_Model模組 (Dynamic import of Read_Model module)
            try:
                import Read_Model  # type: ignore
            except ImportError as e:
                self.analyzer_results.append(f"❌ 無法導入Read_Model模組: {e}")
                self.log_message(f"❌ 無法導入Read_Model模組: {e}")
                return
            
            # 重定向print輸出到分析結果區域
            import io
            from contextlib import redirect_stdout
            
            # 創建字符串緩衝區
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer):
                success = Read_Model.display_model_architecture(model_path)
            
            # 獲取輸出內容
            analysis_output = output_buffer.getvalue()
            
            if success:
                self.analyzer_results.append(analysis_output)
                self.analyzer_results.append("\n" + "=" * 50)
                
                # 添加模型摘要信息
                try:
                    model_summary = Read_Model.get_model_summary(model_path)
                    self.analyzer_results.append(f"\n📋 模型摘要: {model_summary}")
                    
                    # 獲取詳細模型信息
                    model_info = Read_Model.get_model_info(model_path)
                    if 'error' not in model_info:
                        self.analyzer_results.append(f"\n🔍 詳細信息:")
                        if model_info.get('input_channels'):
                            self.analyzer_results.append(f"  輸入通道數: {model_info['input_channels']}")
                        if model_info.get('num_classes'):
                            self.analyzer_results.append(f"  類別數量: {model_info['num_classes']}")
                        if model_info.get('total_parameters', 0) > 0:
                            self.analyzer_results.append(f"  總參數: {model_info['total_parameters']:,}")
                            self.analyzer_results.append(f"  可訓練參數: {model_info['trainable_parameters']:,}")
                        
                        # 顯示精度信息
                        if model_info.get('precision'):
                            self.analyzer_results.append(f"\n🎯 參數精度:")
                            total_params = model_info['total_parameters']
                            for dtype, count in model_info['precision'].items():
                                percentage = (count / total_params) * 100 if total_params > 0 else 0
                                self.analyzer_results.append(f"  {dtype}: {count:,} ({percentage:.1f}%)")
                        
                        # 顯示訓練信息
                        if model_info.get('training_info'):
                            self.analyzer_results.append(f"\n🏋️ 訓練信息:")
                            for key, value in model_info['training_info'].items():
                                self.analyzer_results.append(f"  {key}: {value}")
                        
                        # 顯示.pth檔案的狀態字典信息
                        if 'state_dict_info' in model_info:
                            self.analyzer_results.append(f"\n📊 狀態字典信息 (.pth格式):")
                            self.analyzer_results.append(f"  參數層數: {len(model_info['state_dict_info'])}")
                            
                            # 顯示前5個參數層
                            count = 0
                            for key, info in model_info['state_dict_info'].items():
                                if count < 5:
                                    self.analyzer_results.append(f"    {key}: {info['shape']} ({info['dtype']})")
                                    count += 1
                                else:
                                    break
                            
                            if len(model_info['state_dict_info']) > 5:
                                self.analyzer_results.append(f"    ... 還有 {len(model_info['state_dict_info']) - 5} 個參數層")
                        
                        # 顯示層類型統計
                        if 'layer_info' in model_info:
                            self.analyzer_results.append(f"\n🏗️ 層類型統計:")
                            for layer_type, count in model_info['layer_info'].items():
                                self.analyzer_results.append(f"  {layer_type}: {count} 層")
                
                except Exception as e:
                    self.analyzer_results.append(f"⚠️ 獲取詳細信息時出錯: {e}")
                
                self.analyzer_results.append("\n" + "=" * 50)
                self.analyzer_results.append("✅ 模型分析完成!")
                self.log_message("✅ 模型分析完成")
            else:
                self.analyzer_results.append("❌ 模型分析失敗")
                self.log_message("❌ 模型分析失敗")
                
        except Exception as e:
            error_msg = f"[ERROR] 分析模型時出錯: {str(e)}"
            self.analyzer_results.append(error_msg)
            self.log_message(error_msg)
    
    def batch_analyze_models(self):
        """批量分析所有模型"""
        if self.analyzer_model_combo.count() == 0:
            QMessageBox.warning(self, "警告", "請先刷新模型列表")
            return
        
        # 獲取所有模型路徑
        model_paths = []
        for i in range(self.analyzer_model_combo.count()):
            model_path = self.analyzer_model_combo.itemData(i)
            if model_path and Path(model_path).exists():
                model_paths.append(model_path)
        
        if not model_paths:
            QMessageBox.warning(self, "警告", "沒有找到有效的模型文件")
            return
        
        self.log_message(f"📊 開始批量分析 {len(model_paths)} 個模型")
        self.analyzer_results.clear()
        self.analyzer_results.append("📊 批量模型分析開始...")
        self.analyzer_results.append("=" * 60)
        
        try:
            # 導入Read_Model模組
            import sys
            code_path = Path(__file__).parent / 'Code'
            if str(code_path) not in sys.path:
                sys.path.insert(0, str(code_path))
            
            try:
                import Read_Model  # type: ignore
            except ImportError as e:
                self.analyzer_results.append(f"❌ 無法導入Read_Model模組: {e}")
                self.log_message(f"❌ 無法導入Read_Model模組: {e}")
                return
            
            # 執行批量分析
            batch_results = Read_Model.analyze_model_batch(model_paths)
            
            # 顯示批量分析結果
            self.analyzer_results.append(f"📈 批量分析結果:")
            self.analyzer_results.append(f"  總模型數: {batch_results['total_models']}")
            self.analyzer_results.append(f"  成功分析: {batch_results['successful_analyses']}")
            self.analyzer_results.append(f"  失敗分析: {batch_results['failed_analyses']}")
            
            if batch_results.get('summary'):
                summary = batch_results['summary']
                if 'avg_parameters' in summary:
                    self.analyzer_results.append(f"\n📊 參數統計:")
                    self.analyzer_results.append(f"  平均參數: {summary['avg_parameters']:,.0f}")
                    self.analyzer_results.append(f"  最少參數: {summary['min_parameters']:,}")
                    self.analyzer_results.append(f"  最多參數: {summary['max_parameters']:,}")
            
            # 顯示每個模型的摘要
            self.analyzer_results.append(f"\n📋 模型摘要:")
            for model_path, model_info in batch_results['models'].items():
                model_name = Path(model_path).name
                if 'error' in model_info:
                    self.analyzer_results.append(f"  ❌ {model_name}: {model_info['error']}")
                else:
                    summary = Read_Model.get_model_summary(model_path)
                    self.analyzer_results.append(f"  ✅ {model_name}: {summary}")
            
            self.analyzer_results.append("\n" + "=" * 60)
            self.analyzer_results.append("✅ 批量分析完成!")
            self.log_message("✅ 批量分析完成")
            
        except Exception as e:
            error_msg = f"❌ 批量分析時出錯: {str(e)}"
            self.analyzer_results.append(error_msg)
            self.log_message(error_msg)
    
    def save_analysis_results(self):
        """保存分析結果到文件"""
        if not self.analyzer_results.toPlainText().strip():
            QMessageBox.warning(self, "警告", "沒有分析結果可保存")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析結果", "model_analysis.txt", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.analyzer_results.toPlainText())
            self.log_message(f"💾 分析結果已保存到: {file_path}")
    
    def clear_analysis_results(self):
        """清空分析結果"""
        self.analyzer_results.clear()
        self.log_message("[DELETE] 分析結果已清空")
    
    def update_model_info(self):
        """更新模型文件信息"""
        # 刷新模型列表
        self.refresh_model_list()
    
    def refresh_model_list(self):
        """刷新模型列表"""
        # 確保 log_message 能正常工作
        if hasattr(self, 'log_text') and self.log_text:
            self.log_message("🔄 刷新模型列表...")
        # 移除終端輸出
        
        # 清空現有列表
        self.model_file_combo.clear()
        
        # 獲取當前選擇的模型類型
        model_type = self.model_type_combo.currentData() if hasattr(self, 'model_type_combo') else "standard"
        
        # 根據模型類型選擇目錄
        if model_type == "YAML":
            subfolder = "yaml"
            self.log_message("[SEARCH] 掃描YAML模型...")
        else:
            subfolder = "PT_File"
            self.log_message("[SEARCH] 掃描PT模型...")
        
        # 掃描Model_file/指定子目錄/中的.pt文件
        pt_files = []
        # 嘗試多個可能的路徑
        possible_paths = [
            Path(f"Model_file/{subfolder}"),  # 相對路徑
            Path(__file__).parent / f"Model_file/{subfolder}",  # 從腳本目錄
            Path.cwd() / f"Model_file/{subfolder}",  # 從工作目錄
        ]
        
        # 添加詳細的調試信息
        self.log_message(f"[SEARCH] 開始搜索Model_file/{subfolder}目錄...")
        self.log_message(f"[FOLDER] 當前工作目錄: {Path.cwd()}")
        self.log_message(f"[FOLDER] 腳本目錄: {Path(__file__).parent}")
        
        target_dir = None
        for i, path in enumerate(possible_paths):
            abs_path = path.resolve()
            exists = path.exists()
            self.log_message(f"[SEARCH] 路徑 {i+1}: {path}")
            self.log_message(f"   絕對路徑: {abs_path}")
            self.log_message(f"   存在: {exists}")
            if exists:
                target_dir = path
                self.log_message(f"[OK] 找到目錄: {path}")
                break
        
        if not target_dir:
            target_dir = Path(f"Model_file/{subfolder}")  # 默認路徑
            self.log_message(f"[WARNING] 使用默認路徑: {target_dir}")
        
        # 最終檢查
        self.log_message(f"[FOLDER] 最終使用目錄: {target_dir.absolute()}")
        self.log_message(f"[FOLDER] 目錄存在: {target_dir.exists()}")
        
        if target_dir.exists():
            if model_type == "YAML":
                # 掃描YAML目錄的.yaml文件
                for yaml_file in target_dir.glob("*.yaml"):
                    self.log_message(f"📄 找到YAML文件: {yaml_file.name}")
                    pt_files.append((subfolder, yaml_file.name, str(yaml_file)))
                
                # 統計指定目錄的YAML文件
                self.log_message(f"[CHART] {subfolder}目錄統計: 找到 {len(pt_files)} 個YAML文件")
            else:
                # 掃描指定目錄的.pt文件
                for pt_file in target_dir.glob("*.pt"):
                    self.log_message(f"📄 找到模型文件: {pt_file.name}")
                    pt_files.append((subfolder, pt_file.name, str(pt_file)))
                
                # 統計指定目錄的PT文件
                self.log_message(f"[CHART] {subfolder}目錄統計: 找到 {len(pt_files)} 個PT文件")
        else:
            self.log_message(f"[ERROR] 目錄不存在: {target_dir.absolute()}")
            if model_type == "YAML":
                self.log_message(f"[CHART] {subfolder}目錄統計: 找到 0 個YAML文件")
            else:
                self.log_message(f"[CHART] {subfolder}目錄統計: 找到 0 個PT文件")
        
        
        # 按照nsmlx順序排序模型文件
        def get_model_priority(filename):
            """獲取模型優先級，用於nsmlx排序"""
            filename_lower = filename.lower()
            if 'yolov12n' in filename_lower or 'nano' in filename_lower:
                return 0  # n - 最高優先級
            elif 'yolov12s' in filename_lower or 'small' in filename_lower:
                return 1  # s
            elif 'yolov12m' in filename_lower or 'medium' in filename_lower:
                return 2  # m
            elif 'yolov12l' in filename_lower or 'large' in filename_lower:
                return 3  # l
            elif 'yolov12x' in filename_lower or 'xlarge' in filename_lower:
                return 4  # x
            else:
                return 5  # 其他模型放在最後
        
        # 按照nsmlx順序排序
        pt_files.sort(key=lambda x: get_model_priority(x[1]))
        
        
        # 然後添加模型文件
        for category, filename, full_path in pt_files:
            if model_type == "YAML":
                # YAML 模型文件處理
                display_text = f"{category}/{filename} (YAML配置)"
                abs_path = Path(full_path).resolve()
                self.log_message(f"[SEARCH] 添加YAML模型: display_text='{display_text}', full_path='{full_path}', abs_path='{abs_path}'")
                self.model_file_combo.addItem(display_text, str(abs_path))
            else:
                # PT 模型文件處理
                # 檢查模型實際的輸入通道數
                model_channels = self._get_model_input_channels(full_path)
                if model_channels:
                    display_text = f"{category}/{filename} ({model_channels}通道)"
                else:
                    display_text = f"{category}/{filename}"
                
                # 確保使用絕對路徑
                abs_path = Path(full_path).resolve()
                self.log_message(f"[SEARCH] 添加模型: display_text='{display_text}', full_path='{full_path}', abs_path='{abs_path}'")
                self.model_file_combo.addItem(display_text, str(abs_path))
        
        if pt_files:
            if model_type == "YAML":
                self.log_message(f"[OK] 找到 {len(pt_files)} 個YAML模型文件")
            else:
                self.log_message(f"[OK] 找到 {len(pt_files)} 個{model_type}模型文件")
            # 自動選擇第一個模型
            self.update_selected_model_info()
        else:
            if model_type == "YAML":
                self.log_message(f"[WARNING] 未找到任何YAML模型文件")
                self.train_model_status.setText(f"[ERROR] 未找到YAML模型文件")
            else:
                self.log_message(f"[WARNING] 未找到任何{model_type}模型文件")
                self.train_model_status.setText(f"[ERROR] 未找到{model_type}模型文件")
        
        # 顯示總體統計信息
        self.log_message("=" * 50)
        if model_type == "YAML":
            self.log_message("[CHART] YAML文件總體統計:")
            self.log_message(f"   [FOLDER] YAML目錄: {len(pt_files)} 個YAML文件")
        else:
            self.log_message("[CHART] PT文件總體統計:")
            self.log_message(f"   [FOLDER] 4_channel目錄: 請查看上方架構加載日誌")
            self.log_message(f"   [FOLDER] standard目錄: {len(pt_files)} 個PT文件")
        self.log_message("=" * 50)
    
    def _get_model_input_channels(self, model_path):
        """獲取模型的實際輸入通道數"""
        try:
            import torch
            # 載入模型並檢查第一層的輸入通道數
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 處理不同的模型格式
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
            
            # 檢查第一層卷積層的輸入通道數
            if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
                first_conv = model.model[0]
                if hasattr(first_conv, 'conv') and hasattr(first_conv.conv, 'in_channels'):
                    return first_conv.conv.in_channels
            elif hasattr(model, '__getitem__'):
                first_conv = model[0]
                if hasattr(first_conv, 'conv') and hasattr(first_conv.conv, 'in_channels'):
                    return first_conv.conv.in_channels
            
            return None
        except Exception as e:
            self.log_message(f"[WARNING] 檢查模型通道數失敗: {e}")
            return None
    
    def _get_selected_dataset_channels(self):
        """獲取當前選擇的數據集通道數"""
        try:
            # 從下拉選單獲取選擇的資料集路徑
            dataset_path = self.train_dataset_combo.currentData()
            if not dataset_path:
                # 如果沒有data，嘗試從currentText獲取
                dataset_path = self.train_dataset_combo.currentText()
            
            if not dataset_path:
                return None
            
            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                return None
            
            # 檢查是否包含data_config.yaml
            config_file = dataset_dir / "data_config.yaml"
            if config_file.exists():
                try:
                    # 驗證配置文件
                    from Code.YOLO_standard_trainer import ConfigDetector
                    config_info = ConfigDetector.validate_config(str(config_file))
                    
                    if config_info['valid']:
                        return config_info.get('channels', '未知')
                except Exception as e:
                    self.log_message(f"[WARNING] 獲取數據集通道數失敗: {e}")
                    return None
            
            return None
        except Exception as e:
            self.log_message(f"[WARNING] 獲取數據集通道數時出錯: {e}")
            return None
    
    def on_training_mode_changed(self):
        """訓練模式改變時的處理邏輯 - 簡化版本"""
        try:
            if self.pretrained_radio.isChecked():
                # 預訓練模型模式
                self.log_message("🔄 切換到預訓練模型模式")
                self.train_model_size_combo.setVisible(False)  # 隱藏模型大小選擇
                
                # 重新啟用PT模型文件選擇
                if hasattr(self, 'model_file_combo'):
                    self.model_file_combo.setEnabled(True)
                    self.model_file_combo.setPlaceholderText("選擇PT模型文件")
                    # 自動刷新PT模型列表
                    self.log_message("🔄 自動刷新PT模型列表...")
                    self.refresh_model_list()
                    self.log_message("✅ PT模型列表刷新完成")
                
                # 自動刷新模型類別
                self.auto_refresh_model_categories()
                self.current_mode_label.setText("當前模式：預訓練模型 (PT)")
                self.current_mode_label.setStyleSheet("""
                    QLabel {
                        color: #007bff;
                        font-size: 13px;
                        font-weight: bold;
                        padding: 8px;
                        background-color: #d1ecf1;
                        border: 1px solid #bee5eb;
                        border-radius: 4px;
                        margin-bottom: 5px;
                    }
                """)
                self.train_model_status.setText("預訓練模式：將使用PT模型文件進行微調訓練")
                self.train_model_status.setStyleSheet("""
                    QLabel {
                        color: #28a745;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 6px 10px;
                        background-color: #d4edda;
                        border: 1px solid #c3e6cb;
                        border-radius: 4px;
                        margin: 5px 0;
                    }
                """)
                self.log_message("📋 預訓練模式：將使用PT模型文件進行微調訓練")
            elif self.retrain_radio.isChecked():
                # 重新訓練模式
                self.log_message("🔄 切換到重新訓練模式")
                self.train_model_size_combo.setVisible(True)  # 顯示模型大小選擇
                
                # 清空並重新配置模型文件選擇為YAML模式
                if hasattr(self, 'model_file_combo'):
                    self.model_file_combo.clear()
                    self.model_file_combo.setEnabled(True)  # 啟用選擇框
                    self.model_file_combo.setPlaceholderText("選擇YAML配置文件")
                
                # 自動刷新YAML模型列表
                self.log_message("🔄 自動刷新YAML模型列表...")
                self.refresh_yaml_model_list()
                self.log_message("✅ YAML模型列表刷新完成")
                
                # 自動刷新模型類別
                self.auto_refresh_model_categories()
                
                self.current_mode_label.setText("當前模式：重新訓練 (YAML)")
                self.current_mode_label.setStyleSheet("""
                    QLabel {
                        color: #28a745;
                        font-size: 13px;
                        font-weight: bold;
                        padding: 8px;
                        background-color: #d4edda;
                        border: 1px solid #c3e6cb;
                        border-radius: 4px;
                        margin-bottom: 5px;
                    }
                """)
                self.train_model_status.setText("重新訓練模式：將使用YAML配置文件從頭開始訓練")
                self.train_model_status.setStyleSheet("""
                    QLabel {
                        color: #28a745;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 6px 10px;
                        background-color: #d4edda;
                        border: 1px solid #c3e6cb;
                        border-radius: 4px;
                        margin: 5px 0;
                    }
                """)
                self.log_message("📋 重新訓練模式：將使用YAML配置文件從頭開始訓練")
            
        except Exception as e:
            self.log_message(f"[ERROR] 訓練模式切換失敗: {e}")
    
    def refresh_yaml_model_list(self):
        """刷新YAML模型列表"""
        try:
            self.log_message("[SEARCH] 掃描YAML模型文件...")
            
            # 掃描Model_file/YAML/目錄中的YAML文件
            yaml_dir = Path("Model_file/YAML")
            yaml_files = []
            
            if yaml_dir.exists():
                for yaml_file in yaml_dir.glob("*.yaml"):
                    yaml_files.append(yaml_file)
            
            if yaml_files:
                self.log_message(f"[OK] 找到 {len(yaml_files)} 個YAML模型文件")
                # 更新模型文件選擇框（如果存在）
                if hasattr(self, 'model_file_combo'):
                    self.model_file_combo.clear()
                    for yaml_file in yaml_files:
                        display_text = f"YAML/{yaml_file.name}"
                        self.model_file_combo.addItem(display_text, str(yaml_file))
            else:
                self.log_message("[WARNING] 未找到任何YAML模型文件")
                if hasattr(self, 'model_file_combo'):
                    self.model_file_combo.clear()
                    self.model_file_combo.addItem("未找到YAML文件", "")
        except Exception as e:
            self.log_message(f"[ERROR] 刷新YAML模型列表失敗: {e}")
    
    def smart_refresh_model_list(self):
        """智能刷新模型列表 - 根據當前模式選擇正確的刷新方法"""
        try:
            if hasattr(self, 'retrain_radio') and self.retrain_radio.isChecked():
                # 重新訓練模式 - 刷新YAML模型列表
                self.refresh_yaml_model_list()
            else:
                # 預訓練模式 - 刷新PT模型列表
                self.refresh_model_list()
        except Exception as e:
            self.log_message(f"[ERROR] 智能刷新模型列表失敗: {e}")
    
    def get_dynamic_class_count(self):
        """動態獲取類別數量"""
        try:
            from config.predefined_classes import load_predefined_classes
            predefined_classes = load_predefined_classes()
            return len(predefined_classes)
        except Exception as e:
            self.log_message(f"⚠️ 無法載入預定義類別，使用默認值: {e}")
            return 1
    
    def auto_refresh_standard_training(self):
        """根據記錄的數值自動刷新標準訓練模型部分"""
        try:
            self.log_message("🔄 自動刷新標準訓練模型部分...")
            
            # 刷新模型列表
            self.refresh_model_list()
            
            # 刷新YAML模型列表
            self.refresh_yaml_model_list()
            
            # 根據保存的設置更新訓練模式
            if hasattr(self, 'pretrained_radio') and hasattr(self, 'retrain_radio'):
                # 檢查保存的訓練模式
                try:
                    import yaml
                    if self.settings_file.exists():
                        with open(self.settings_file, 'r', encoding='utf-8') as f:
                            settings = yaml.safe_load(f)
                        
                        training_mode = settings.get('standard_training', {}).get('training_mode', 'pretrained')
                        
                        if training_mode == 'retrain':
                            self.retrain_radio.setChecked(True)
                            self.train_model_size_combo.setVisible(True)
                            self.log_message("📋 恢復重新訓練模式")
                        else:
                            self.pretrained_radio.setChecked(True)
                            self.train_model_size_combo.setVisible(False)
                            self.log_message("📋 恢復預訓練模式")
                        
                        # 觸發模式切換事件
                        self.on_training_mode_changed()
                except Exception as e:
                    self.log_message(f"⚠️ 恢復訓練模式失敗: {e}")
                    # 默認使用預訓練模式
                    self.pretrained_radio.setChecked(True)
                    self.on_training_mode_changed()
            
            # 自動查找最新的數據集
            self.auto_find_train_dataset()
            
            self.log_message("✅ 標準訓練模型部分刷新完成")
            
        except Exception as e:
            self.log_message(f"[ERROR] 自動刷新標準訓練失敗: {e}")
    
    def auto_refresh_model_categories(self):
        """自動刷新模型類別 - 根據當前模式更新模型分類"""
        try:
            if hasattr(self, 'retrain_radio') and self.retrain_radio.isChecked():
                # 重新訓練模式 - 更新為YAML模型類別
                self.log_message("🔄 更新模型類別為YAML模式")
                if hasattr(self, 'train_model_status'):
                    self.train_model_status.setText("YAML模式：可選擇不同的模型架構")
                    self.train_model_status.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
            else:
                # 預訓練模式 - 更新為PT模型類別
                self.log_message("🔄 更新模型類別為PT模式")
                if hasattr(self, 'train_model_status'):
                    self.train_model_status.setText("PT模式：可選擇不同的預訓練模型")
                    self.train_model_status.setStyleSheet("color: #007bff; font-size: 12px; font-weight: bold;")
        except Exception as e:
            self.log_message(f"[ERROR] 自動刷新模型類別失敗: {e}")

    
    
    def _get_detailed_model_info(self, model_path):
        """获取详细的模型信息"""
        try:
            import torch
            model_info = {
                'param_count': 0,
                'input_channels': 3,
                'model_type': 'Unknown',
                'version': 'Unknown'
            }
            
            # 尝试获取模型参数数量
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    if hasattr(model, 'parameters'):
                        model_info['param_count'] = sum(p.numel() for p in model.parameters())
                    if hasattr(model, 'yaml'):
                        model_info['model_type'] = 'YOLO'
                    if hasattr(model, 'version'):
                        model_info['version'] = str(model.version)
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model_info['param_count'] = sum(p.numel() for p in checkpoint['model'].parameters()) if hasattr(checkpoint['model'], 'parameters') else 0
            except Exception as e:
                self.log_message(f"[WARNING] 无法获取模型参数信息: {e}")
            
            # 获取输入通道数
            model_info['input_channels'] = self._get_model_input_channels(model_path) or 3
            
            return model_info
        except Exception as e:
            self.log_message(f"[WARNING] 获取模型信息失败: {e}")
            return {'param_count': 0, 'input_channels': 3, 'model_type': 'Unknown', 'version': 'Unknown'}
    
    def _update_detailed_model_arch_desc(self, model_path, model_name, model_info):
        """更新详细的模型架构描述"""
        try:
            # 格式化参数数量
            param_count = model_info.get('param_count', 0)
            if param_count > 0:
                if param_count >= 1e9:
                    param_str = f"{param_count/1e9:.1f}B"
                elif param_count >= 1e6:
                    param_str = f"{param_count/1e6:.1f}M"
                elif param_count >= 1e3:
                    param_str = f"{param_count/1e3:.1f}K"
                else:
                    param_str = str(param_count)
            else:
                param_str = "未知"
            
            # 获取文件大小
            file_size = Path(model_path).stat().st_size / (1024 * 1024)
            file_size_str = f"{file_size:.1f} MB" if file_size < 1024 else f"{file_size/1024:.1f} GB"
            
            # 确定模型类型和特点
            model_name_lower = model_name.lower()
            if 'nano' in model_name_lower or 'n' in model_name_lower:
                model_type = "Nano (超轻量)"
                characteristics = "• 速度最快，资源消耗最少\n• 适合实时推理和移动设备\n• 精度相对较低"
            elif 'small' in model_name_lower or 's' in model_name_lower:
                model_type = "Small (轻量)"
                characteristics = "• 平衡速度和精度\n• 适合边缘计算设备\n• 推荐用于一般应用"
            elif 'medium' in model_name_lower or 'm' in model_name_lower:
                model_type = "Medium (中等)"
                characteristics = "• 精度和速度的良好平衡\n• 适合大多数应用场景\n• 推荐用于生产环境"
            elif 'large' in model_name_lower or 'l' in model_name_lower:
                model_type = "Large (大型)"
                characteristics = "• 高精度，速度较慢\n• 适合对精度要求高的场景\n• 需要较强的计算资源"
            elif 'xlarge' in model_name_lower or 'x' in model_name_lower:
                model_type = "XLarge (超大型)"
                characteristics = "• 最高精度，速度最慢\n• 适合研究和开发\n• 需要强大的计算资源"
            else:
                model_type = "Custom (自定义)"
                characteristics = "• 自定义模型架构\n• 根据具体需求设计\n• 需要根据实际情况评估"
            
            # 构建详细描述
            desc = f"""
                模型详细信息:
                • 模型名称: {model_name}
                • 模型类型: {model_type}
                • 参数量: {param_str}
                • 文件大小: {file_size_str}
                • 输入通道: {model_info.get('input_channels', 3)}通道
                • 模型版本: {model_info.get('version', 'Unknown')}

                模型特点:
                {characteristics}

                使用建议:
                • 根据硬件配置选择合适的模型大小
                • 考虑精度和速度的平衡
                • 建议先用小模型测试，再使用大模型训练
            """.strip()
            
            self.train_arch_desc_label.setText(desc)
            
        except Exception as e:
            self.log_message(f"[WARNING] 更新模型描述失败: {e}")
            self.train_arch_desc_label.setText(f"模型信息: {model_name}\n• 文件大小: {file_size_str}\n• 输入通道: {model_info.get('input_channels', 3)}通道")
    
    def _update_standard_model_arch_desc(self, model_path, model_name):
        """更新標準模型架構描述"""
        try:
            # 獲取模型參數數量
            param_count = self._get_model_param_count(Path(model_path))
            
            # 構建架構描述
            desc_text = f"📝 {model_name}架構"
            if param_count:
                desc_text += f"\n🔢 參數數量: {param_count}"
            desc_text += f"\n[FOLDER] 來源: {Path(model_path).name}"
            
            self.train_arch_desc_label.setText(desc_text)
            
        except Exception as e:
            self.train_arch_desc_label.setText(f"📝 {model_name}架構\n[WARNING] 無法獲取詳細信息")
            self.log_message(f"[WARNING] 更新標準模型架構描述失敗: {e}")
    
    def check_selected_model(self):
        """檢查選中的模型"""
        self.log_message("[SEARCH] 檢查選中模型...")
        self.update_selected_model_info()
    
    def update_standard_model_info(self):
        """更新標準模型信息 - 已移除，由模型選擇下拉選單處理"""
        pass
    
    def update_selected_model_info(self):
        """更新選中的模型信息"""
        try:
            if hasattr(self, 'model_file_combo') and self.model_file_combo.currentData():
                model_path = self.model_file_combo.currentData()
                if Path(model_path).exists():
                    self.log_message(f"[OK] 已選擇模型: {Path(model_path).name}")
                    # 更新模型狀態顯示
                    if hasattr(self, 'train_model_status'):
                        self.train_model_status.setText(f"已選擇模型: {Path(model_path).name}")
                        self.train_model_status.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                else:
                    self.log_message(f"[WARNING] 模型文件不存在: {model_path}")
            else:
                self.log_message("[INFO] 未選擇模型文件")
        except Exception as e:
            self.log_message(f"[ERROR] 更新模型信息失敗: {e}")
    
    
    def check_standard_model(self):
        """檢查標準模型"""
        self.log_message("[SEARCH] 檢查標準模型...")
        self.update_standard_model_info()

    # 模型修改器相關方法
    def browse_modifier_input_model(self):
        """瀏覽輸入模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇要修改的模型文件", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.modifier_input_model_edit.setText(file_path)
            # 自動生成輸出文件名
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_modified{input_path.suffix}"
            self.modifier_output_model_edit.setText(str(output_path))
            # 自動分析模型
            self.analyze_model_for_modification()
    
    def browse_modifier_output_model(self):
        """瀏覽輸出模型文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存修改後的模型", ".", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.modifier_output_model_edit.setText(file_path)
    
    def analyze_model_for_modification(self):
        """分析模型結構"""
        input_path = self.modifier_input_model_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "警告", "請先選擇輸入模型文件")
            return
        
        if not Path(input_path).exists():
            QMessageBox.warning(self, "警告", "輸入模型文件不存在")
            return
        
        try:
            from Code.model_modifier import analyze_model_structure
            
            self.log_message("[SEARCH] 分析模型結構...")
            
            # 使用模組化分析功能
            result = analyze_model_structure(input_path)
            
            if 'error' in result:
                error_msg = f"[ERROR] {result['error']}"
                self.modifier_model_info_text.setPlainText(error_msg)
                self.log_message(error_msg)
                return
            
            if not result.get('success', False):
                error_msg = f"[ERROR] 分析失敗: {result.get('error', '未知錯誤')}"
                self.modifier_model_info_text.setPlainText(error_msg)
                self.log_message(error_msg)
                return
            
            # 構建顯示信息
            model_info = []
            model_info.append(f"[FOLDER] 模型文件: {result['file_name']}")
            model_info.append(f"[CHART] 模型類型: {result['model_type']}")
            model_info.append(f"[SEARCH] 卷積層總數: {result['total_conv_layers']}")
            
            # 第一層卷積層信息
            first_conv = result['first_conv']
            model_info.append(f"[SEARCH] 第一層卷積層: {first_conv['name']}")
            model_info.append(f"   輸入通道數: {first_conv['in_channels']}")
            model_info.append(f"   輸出通道數: {first_conv['out_channels']}")
            model_info.append(f"   卷積核大小: {first_conv['kernel_size']}")
            model_info.append(f"   步長: {first_conv['stride']}")
            model_info.append(f"   填充: {first_conv['padding']}")
            model_info.append(f"   偏置: {'是' if first_conv['bias'] else '否'}")
            
            # 自動設置原始通道數
            self.modifier_original_channels_spin.setValue(first_conv['in_channels'])
            
            # 智能建議
            suggestions = result['suggestions']
            if suggestions['recommended_target']:
                self.modifier_target_channels_spin.setValue(suggestions['recommended_target'])
                model_info.append(f"💡 建議: {suggestions['reason']}")
            else:
                model_info.append(f"💡 建議: {suggestions['reason']}")
            
            # 顯示所有卷積層信息
            model_info.append("\n📋 所有卷積層:")
            for conv in result['all_conv_layers']:
                model_info.append(f"   {conv['name']}: {conv['in_channels']}→{conv['out_channels']}")
            
            # 顯示模型信息
            self.modifier_model_info_text.setPlainText("\n".join(model_info))
            
            self.log_message("[OK] 模型分析完成")
            
        except ImportError as e:
            error_msg = f"[ERROR] 模組導入失敗: {e}"
            self.modifier_model_info_text.setPlainText(error_msg)
            self.log_message(error_msg)
        except Exception as e:
            error_msg = f"[ERROR] 模型分析失敗: {e}"
            self.modifier_model_info_text.setPlainText(error_msg)
            self.log_message(error_msg)
    
    def modify_model_channels(self):
        """修改模型通道數"""
        input_path = self.modifier_input_model_edit.text().strip()
        output_path = self.modifier_output_model_edit.text().strip()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "請設置輸入和輸出模型路徑")
            return
        
        if not Path(input_path).exists():
            QMessageBox.warning(self, "警告", "輸入模型文件不存在")
            return
        
        original_channels = self.modifier_original_channels_spin.value()
        target_channels = self.modifier_target_channels_spin.value()
        weight_method_text = self.modifier_weight_method_combo.currentText()
        
        if original_channels == target_channels:
            QMessageBox.information(self, "提示", "原始通道數與目標通道數相同，無需修改")
            return
        
        try:
            from Code.model_modifier import modify_model_channels
            
            # 將中文方法名轉換為英文代碼
            weight_method_map = {
                "複製原始權重 + 平均值": "copy_avg",
                "複製原始權重 + 零初始化": "copy_zero", 
                "複製原始權重 + 隨機初始化": "copy_random",
                "完全隨機初始化": "full_random"
            }
            weight_method = weight_method_map.get(weight_method_text, "copy_avg")
            
            self.log_message("🔧 開始修改模型通道數...")
            self.log_message(f"   原始通道數: {original_channels}")
            self.log_message(f"   目標通道數: {target_channels}")
            self.log_message(f"   權重初始化: {weight_method_text}")
            
            # 使用模組化修改功能
            result = modify_model_channels(
                input_path, output_path, original_channels, target_channels, weight_method
            )
            
            if not result.get('success', False):
                error_msg = f"[ERROR] 模型修改失敗: {result.get('error', '未知錯誤')}"
                QMessageBox.critical(self, "錯誤", error_msg)
                self.log_message(error_msg)
                return
            
            # 顯示成功信息
            success_msg = f"[OK] 模型修改成功！\n\n"
            success_msg += f"[FOLDER] 輸出文件: {Path(output_path).name}\n"
            success_msg += f"🔧 通道數: {result['original_channels']} → {result['actual_channels']}\n"
            success_msg += f"[CHART] 權重初始化: {weight_method_text}\n"
            success_msg += f"💾 文件大小: {result['file_size_mb']} MB"
            
            # 添加驗證信息
            if result.get('verification', {}).get('success', False):
                verification = result['verification']
                if verification.get('match', False):
                    success_msg += f"\n[OK] 驗證通過: 實際通道數 {verification['actual_channels']} 符合預期"
                else:
                    success_msg += f"\n[WARNING] 驗證警告: 實際通道數 {verification['actual_channels']} 與預期不符"
            
            QMessageBox.information(self, "成功", success_msg)
            self.log_message("[OK] 模型修改完成")
            self.log_message(f"[FOLDER] 修改後的模型已保存: {output_path}")
            
        except ImportError as e:
            error_msg = f"[ERROR] 模組導入失敗: {e}"
            QMessageBox.critical(self, "錯誤", error_msg)
            self.log_message(error_msg)
        except Exception as e:
            error_msg = f"[ERROR] 模型修改失敗: {e}"
            QMessageBox.critical(self, "錯誤", error_msg)
            self.log_message(error_msg)
    
    def _load_model_types(self):
        """加載模型類型選項"""
        try:
            self.model_type_combo.clear()
            
            # 只添加 YAML 模型類型選項
            yaml_dir = Path("Model_file/yaml")
            if yaml_dir.exists():
                self.model_type_combo.addItem("YAML", "YAML")
                self.log_message("[OK] 加載了 YAML 模型類型")
            else:
                self.log_message("[WARNING] YAML 目錄不存在")
            
        except Exception as e:
            self.log_message(f"[ERROR] 加載模型類型失敗: {e}")
    
    def on_model_type_changed(self):
        """模型類型改變時更新模型文件選項"""
        try:
            # 清空模型文件選項
            self.model_file_combo.clear()
            
            # 獲取選中的模型類型
            current_data = self.model_type_combo.currentData()
            
            if not current_data:
                self.train_model_status.setText("請先選擇模型類型")
                # 隱藏模型大小選擇器
                self.train_model_size_combo.setVisible(False)
                return
            
            # 顯示/隱藏模型大小選擇器
            if current_data == "YAML":
                # 自定義訓練標籤頁的模型大小選擇器
                if hasattr(self, 'train_model_size_combo'):
                    self.train_model_size_combo.setVisible(True)
                    self.train_model_size_combo.setCurrentText("n")  # 默認選擇n
                
                # 標準訓練標籤頁的模型大小選擇器
                if hasattr(self, 'train_model_size_combo'):
                    self.train_model_size_combo.setVisible(True)
                    self.train_model_size_combo.setCurrentText("n")  # 默認選擇n
                
            else:
                # 隱藏所有模型大小選擇器
                if hasattr(self, 'train_model_size_combo'):
                    self.train_model_size_combo.setVisible(False)
                if hasattr(self, 'train_model_size_combo'):
                    self.train_model_size_combo.setVisible(False)
            
            # 構建模型文件路徑
            model_type_dir = Path("Model_file") / current_data
            if not model_type_dir.exists():
                self.log_message(f"[WARNING] 模型類型目錄不存在: {model_type_dir}")
                self.train_model_status.setText(f"目錄不存在: {model_type_dir}")
                # 即使目錄不存在，也要保持模型大小選擇器的顯示狀態
                return
            
            # 掃描模型文件
            model_files = []
            
            # 查找 .pt 文件
            pt_files = list(model_type_dir.glob("*.pt"))
            for pt_file in pt_files:
                model_files.append({
                    "name": pt_file.name,
                    "path": str(pt_file),
                    "type": "PT模型",
                    "size": self._get_file_size(pt_file)
                })
            
            # 查找 .yaml 文件
            yaml_files = list(model_type_dir.glob("*.yaml"))
            for yaml_file in yaml_files:
                model_files.append({
                    "name": yaml_file.name,
                    "path": str(yaml_file),
                    "type": "YAML配置",
                    "size": self._get_file_size(yaml_file)
                })
            
            # 按文件名排序
            model_files.sort(key=lambda x: x["name"])
            
            if not model_files:
                self.train_model_status.setText(f"在 {current_data} 中未找到模型文件")
                self.log_message(f"[WARNING] 在 {current_data} 中未找到模型文件")
                return
            
            # 添加到下拉框
            for model_file in model_files:
                display_text = f"{model_file['name']} ({model_file['type']}, {model_file['size']})"
                self.model_file_combo.addItem(display_text, model_file["path"])
            
            # 更新信息标签
            self.train_model_status.setText(f"在 {current_data} 中找到 {len(model_files)} 個文件")
            self.log_message(f"[OK] 在 {current_data} 中找到 {len(model_files)} 個模型文件")
            
            # 智能选择推荐文件
            if model_files:
                recommended_index = self._get_recommended_model_file(model_files, current_data)
                self.model_file_combo.setCurrentIndex(recommended_index)
            
        except Exception as e:
            self.log_message(f"[ERROR] 更新模型文件選項失敗: {e}")
            self.train_model_status.setText(f"錯誤: {e}")
    
    def on_model_file_changed(self):
        """模型文件改變時更新信息"""
        try:
            current_data = self.model_file_combo.currentData()
            if not current_data:
                self.train_model_status.setText("")
                return
            
            model_path = Path(current_data)
            if not model_path.exists():
                self.train_model_status.setText("[ERROR] 模型文件不存在")
                return
            
            # 更新模型文件信息
            file_size = self._get_file_size(model_path)
            file_type = "PT模型" if model_path.suffix == ".pt" else "YAML配置"
            
            info_text = f"[FOLDER] {model_path.name} | {file_type} | {file_size}"
            self.train_model_status.setText(info_text)
            
            # 更新預訓練模型輸入框
            self.train_custom_model_edit.setText(str(model_path))
            
        except Exception as e:
            self.log_message(f"[ERROR] 更新模型文件信息失敗: {e}")
    
    def on_model_size_changed(self):
        """模型大小改變時更新模型路徑"""
        try:
            # 獲取當前選中的模型類型和大小
            model_type = self.model_type_combo.currentData()
            model_size = self.train_model_size_combo.currentText()
            
            
            if not model_type or not model_size:
                return
            
            # 只有YAML類型才需要處理動態路徑
            if model_type == "YAML":
                # 構建動態路徑 - 使用與train.py相同的邏輯
                base_name = "yolo12"  # 基礎名稱
                dynamic_path = f"Model_file/yaml/{base_name}{model_size}.yaml"
                
                # 總是使用動態路徑（無論文件是否存在）
                self.train_custom_model_edit.setText(dynamic_path)
                
                # 檢查文件是否存在並記錄日誌
                if Path(dynamic_path).exists():
                    self.log_message(f"[OK] 動態路徑: {dynamic_path} (文件存在)")
                else:
                    self.log_message(f"[INFO] 動態路徑: {dynamic_path} (文件不存在，但路徑已生成)")
            
        except Exception as e:
            self.log_message(f"[ERROR] 更新模型大小失敗: {e}")
    
    def on_train_model_size_changed(self):
        """標準訓練標籤頁模型大小改變時更新模型路徑"""
        try:
            # 獲取當前選中的模型類型和大小
            model_type = self.model_type_combo.currentData()
            model_size = self.train_model_size_combo.currentText()
            
            if not model_type or not model_size:
                return
            
            # 只有YAML類型才需要處理動態路徑
            if model_type == "YAML":
                # 構建動態路徑 - 使用與train.py相同的邏輯
                base_name = "yolo12"  # 基礎名稱
                dynamic_path = f"Model_file/yaml/{base_name}{model_size}.yaml"
                
                # 更新模型文件選擇器
                if hasattr(self, 'model_file_combo'):
                    # 查找對應的模型文件
                    for i in range(self.model_file_combo.count()):
                        item_data = self.model_file_combo.itemData(i)
                        if item_data and dynamic_path in str(item_data):
                            self.model_file_combo.setCurrentIndex(i)
                            # 更新模型信息
                            self.update_selected_model_info()
                            break
                
                # 檢查文件是否存在並記錄日誌
                if Path(dynamic_path).exists():
                    self.log_message(f"[OK] 動態路徑: {dynamic_path} (文件存在)")
                else:
                    self.log_message(f"[INFO] 動態路徑: {dynamic_path} (文件不存在，但路徑已生成)")
            
        except Exception as e:
            self.log_message(f"[ERROR] 更新標準訓練模型大小失敗: {e}")
    
    def browse_model_file(self):
        """瀏覽模型文件"""
        try:
            # 獲取當前選中的模型類型
            current_data = self.model_type_combo.currentData()
            if not current_data:
                QMessageBox.warning(self, "警告", "請先選擇模型類型")
                return
            
            # 構建起始目錄
            start_dir = Path("Model_file") / current_data
            if not start_dir.exists():
                start_dir = Path("Model_file")
            
            # 選擇文件
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "選擇模型文件",
                str(start_dir),
                "模型文件 (*.pt *.yaml);;PT模型 (*.pt);;YAML配置 (*.yaml);;所有文件 (*)"
            )
            
            if file_path:
                # 更新模型文件下拉框
                model_path = Path(file_path)
                display_text = f"{model_path.name} ({'PT模型' if model_path.suffix == '.pt' else 'YAML配置'}, {self._get_file_size(model_path)})"
                
                # 檢查是否已存在
                for i in range(self.model_file_combo.count()):
                    if self.model_file_combo.itemData(i) == file_path:
                        self.model_file_combo.setCurrentIndex(i)
                        return
                
                # 添加新選項
                self.model_file_combo.addItem(display_text, file_path)
                self.model_file_combo.setCurrentIndex(self.model_file_combo.count() - 1)
                
        except Exception as e:
            self.log_message(f"[ERROR] 瀏覽模型文件失敗: {e}")
    
    def _get_file_size(self, file_path):
        """獲取文件大小"""
        try:
            size_bytes = file_path.stat().st_size
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        except:
            return "未知大小"
    
    def _get_recommended_model_file(self, model_files, model_type):
        """獲取推薦的模型文件索引"""
        try:
            # 根據新的目錄結構推薦文件
            if model_type == "PT_File":
                # PT文件優先選擇yolo12n（輕量級）
                for i, model_file in enumerate(model_files):
                    if "yolo12n" in model_file["name"].lower():
                        return i
                # 如果沒有yolo12n，選擇yolo11n
                for i, model_file in enumerate(model_files):
                    if "yolo11n" in model_file["name"].lower():
                        return i
                # 如果沒有yolo11n，選擇yolo12s
                for i, model_file in enumerate(model_files):
                    if "yolo12s" in model_file["name"].lower():
                        return i
            elif model_type == "YAML":
                # YAML配置優先選擇yolo12
                for i, model_file in enumerate(model_files):
                    if "yolo12" in model_file["name"].lower():
                        return i
            
            # 如果沒有找到推薦文件，返回第一個
            return 0
            
        except Exception as e:
            self.log_message(f"[WARNING] 獲取推薦模型文件失敗: {e}")
            return 0

    def clear_modifier_fields(self):
        """清空修改器字段"""
        self.modifier_input_model_edit.clear()
        self.modifier_output_model_edit.clear()
        self.modifier_original_channels_spin.setValue(3)
        self.modifier_target_channels_spin.setValue(4)
        self.modifier_weight_method_combo.setCurrentIndex(0)
        self.modifier_model_info_text.setPlainText("請選擇模型文件以查看詳細信息")
        self.log_message("[DELETE] 修改器字段已清空")
    
    def browse_stereo_dataset(self):
        """瀏覽立體視覺數據集"""
        folder = QFileDialog.getExistingDirectory(self, "選擇立體視覺數據集目錄")
        if folder:
            self.stereo_dataset_edit.setText(folder)
            self.log_message(f"[BROWSE] 立體視覺數據集路徑: {folder}")
    
    def start_stereo_training(self):
        """開始立體視覺訓練"""
        try:
            # 檢查數據集路徑
            dataset_path = self.stereo_dataset_edit.text().strip()
            if not dataset_path:
                QMessageBox.warning(self, "警告", "請選擇立體視覺數據集路徑")
                return
            
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "警告", "數據集路徑不存在")
                return
            
            # 準備訓練參數
            args = self._prepare_stereo_args()
            
            # 更新UI狀態
            self.stereo_start_btn.setEnabled(False)
            self.stereo_stop_btn.setEnabled(True)
            
            # 創建工作線程
            self.stereo_worker = WorkerThread('stereo_training', **args)
            self.stereo_worker.progress.connect(self.update_status)
            self.stereo_worker.finished.connect(self.on_stereo_training_finished)
            self.stereo_worker.log_message.connect(self.log_message)
            self.stereo_worker.epoch_progress.connect(self.update_epoch_progress)
            
            # 開始訓練
            self.stereo_worker.start()
            
            self.log_message("🚀 開始立體視覺深度估計訓練...")
            self.log_message("🚀 Starting stereo vision depth estimation training...")
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"啟動立體視覺訓練失敗: {str(e)}")
            self.log_message(f"❌ 立體視覺訓練啟動失敗: {str(e)}")
    
    def stop_stereo_training(self):
        """停止立體視覺訓練"""
        if hasattr(self, 'stereo_worker') and self.stereo_worker.isRunning():
            self.stereo_worker.stop()
            self.log_message("⏹️ 正在停止立體視覺訓練...")
            self.log_message("⏹️ Stopping stereo vision training...")
    
    def on_stereo_training_finished(self, success, message):
        """立體視覺訓練完成回調"""
        self.stereo_start_btn.setEnabled(True)
        self.stereo_stop_btn.setEnabled(False)
        
        if success:
            self.log_message("✅ 立體視覺訓練完成！")
            self.log_message("✅ Stereo vision training completed!")
            QMessageBox.information(self, "完成", "立體視覺訓練成功完成！")
        else:
            self.log_message(f"❌ 立體視覺訓練失敗: {message}")
            self.log_message(f"❌ Stereo vision training failed: {message}")
            QMessageBox.critical(self, "錯誤", f"立體視覺訓練失敗: {message}")
    
    def clear_stereo_settings(self):
        """清空立體視覺設置"""
        self.stereo_dataset_edit.clear()
        self.stereo_model_combo.setCurrentIndex(0)
        self.stereo_batch_size.setValue(6)
        self.stereo_lr.setValue(0.0002)
        self.stereo_steps.setValue(100000)
        self.stereo_image_size.setText("320,720")
        self.stereo_corr_impl.setCurrentText("reg")
        self.stereo_corr_levels.setValue(4)
        self.stereo_train_iters.setValue(16)
        self.stereo_valid_iters.setValue(32)
        self.stereo_mixed_precision.setChecked(True)
        self.stereo_shared_backbone.setChecked(False)
        self.log_message("[DELETE] 立體視覺設置已清空")
    
    def _prepare_stereo_args(self):
        """準備立體視覺訓練參數"""
        # 解析圖像尺寸
        try:
            image_size = [int(x.strip()) for x in self.stereo_image_size.text().split(',')]
            if len(image_size) != 2:
                image_size = [320, 720]
        except:
            image_size = [320, 720]
        
        args = {
            'dataset_path': self.stereo_dataset_edit.text().strip(),
            'model_name': self.stereo_model_combo.currentText(),
            'batch_size': self.stereo_batch_size.value(),
            'lr': self.stereo_lr.value(),
            'num_steps': self.stereo_steps.value(),
            'image_size': image_size,
            'corr_implementation': self.stereo_corr_impl.currentText(),
            'corr_levels': self.stereo_corr_levels.value(),
            'train_iters': self.stereo_train_iters.value(),
            'valid_iters': self.stereo_valid_iters.value(),
            'mixed_precision': self.stereo_mixed_precision.isChecked(),
            'shared_backbone': self.stereo_shared_backbone.isChecked(),
            'train_datasets': ['sceneflow'],  # 默認使用SceneFlow數據集
            'wdecay': 0.00001,
            'name': 'raft-stereo-custom'
        }
        
        return args


def main():
    """主函數 - 整合啟動檢查和GUI啟動"""
    # 靜默啟動，不在終端顯示信息
    
    import cv2
    import numpy as np
    import yaml
    from pathlib import Path
    
    app = None
    try:
        # 創建PyQt5應用程序
        app = QApplication(sys.argv)
        
        # 設置應用程序信息
        app.setApplicationName("YOLO 統一啟動器")
        app.setOrganizationName("YOLO Project")
        
        # 創建主窗口
        window = YOLOLauncherGUI()
        window.show()
        
        # 運行應用程序
        exit_code = app.exec_()
        # 靜默退出
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        # 靜默處理用戶中斷
        if app:
            app.quit()
        sys.exit(0)
    except Exception as e:
        # 显示启动错误
        print(f"❌ GUI启动失败: {e}")
        print(f"❌ GUI startup failed: {e}")
        import traceback
        traceback.print_exc()
        if app:
            app.quit()
        sys.exit(1)


if __name__ == '__main__':
    main()
