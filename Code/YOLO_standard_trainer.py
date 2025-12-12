"""
YOLO 標準訓練器模組
使用現代化的 YOLO 載入模式，支援 YAML 配置文件和靈活的訓練參數
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
import yaml

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class YOLOStandardTrainer:
    """YOLO 標準訓練器類別"""
    
    def __init__(self, config_path=None, model_path=None, 
                 epochs=50, learning_rate=0.01, batch_size=16, 
                 imgsz=640, save_period=10, 
                 # 数据增强参数
                 scale=0.5, mosaic=1.0, mixup=0.0, 
                 copy_paste=0, hsv_h=0, hsv_s=0, hsv_v=0, 
                 bgr=0, auto_augment=None, 
                 # 几何变换参数
                 degrees=0, translate=0, shear=0, perspective=0,
                 # 翻转和裁剪参数
                 flipud=0, fliplr=0, erasing=0, crop_fraction=0,
                 # 优化器参数
                 weight_decay=0.0005, momentum=0.937, beta1=0.9, beta2=0.999,
                 # 学习率调度参数
                 lr_scheduler='auto', lr_decay=0.1, warmup_epochs=3, warmup_momentum=0.8,
                 # 验证参数
                 val_frequency=1, val_iters=32, early_stopping_patience=50, early_stopping_min_delta=0.001,
                 # 设备参数
                 device='auto', multi_gpu=False, gpu_memory_optimization=True, data_loading_optimization=True,
                 # 其他高级参数
                 close_mosaic=10, single_cls=False, cache=False, resume=False, workers=0, optimizer='SGD', amp=True,
                 progress_callback=None):
        """
        初始化標準訓練器
        
        Args:
            config_path (str): 配置文件路徑
            model_path (str): 模型文件路徑 (YAML 或 PT 文件)
            epochs (int): 訓練輪數
            learning_rate (float): 學習率
            batch_size (int): 批次大小
            imgsz (int): 圖像大小
            save_period (int): 檢查點保存週期
            # 数据增强参数
            scale (float): 縮放比例
            mosaic (float): Mosaic 數據增強
            mixup (float): Mixup 數據增強
            copy_paste (float): Copy-paste 數據增強
            hsv_h (float): HSV色相增強參數
            hsv_s (float): HSV飽和度增強參數
            hsv_v (float): HSV明度增強參數
            bgr (float): BGR通道增強參數
            auto_augment (str): 自動增強策略
            # 几何变换参数
            degrees (float): 旋轉角度
            translate (float): 平移距離
            shear (float): 剪切角度
            perspective (float): 透視變換
            # 翻转和裁剪参数
            flipud (float): 上下翻轉概率
            fliplr (float): 左右翻轉概率
            erasing (float): 隨機擦除概率
            crop_fraction (float): 裁剪比例
            # 优化器参数
            weight_decay (float): 權重衰減
            momentum (float): 動量
            beta1 (float): Adam優化器β1參數
            beta2 (float): Adam優化器β2參數
            # 学习率调度参数
            lr_scheduler (str): 學習率調度器
            lr_decay (float): 學習率衰減
            warmup_epochs (int): 預熱輪數
            warmup_momentum (float): 預熱動量
            # 验证参数
            val_frequency (int): 驗證頻率
            val_iters (int): 驗證迭代次數
            early_stopping_patience (int): 早停耐心值
            early_stopping_min_delta (float): 早停最小改善
            # 设备参数
            device (str): 設備選擇
            multi_gpu (bool): 多GPU訓練
            gpu_memory_optimization (bool): GPU內存優化
            data_loading_optimization (bool): 數據加載優化
            # 其他高级参数
            close_mosaic (int): 關閉Mosaic的epoch數
            single_cls (bool): 單類別訓練
            cache (bool): 數據緩存
            resume (bool): 恢復訓練
            workers (int): 工作進程數
            optimizer (str): 優化器類型
            amp (bool): 是否使用混合精度
            progress_callback (callable): 進度回調函數
        """
        self.config_path = config_path
        self.model_path = model_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.save_period = save_period
        
        # 数据增强参数
        self.scale = scale
        self.mosaic = mosaic
        self.mixup = mixup
        self.copy_paste = copy_paste
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.bgr = bgr
        self.auto_augment = auto_augment
        
        # 几何变换参数
        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.perspective = perspective
        
        # 翻转和裁剪参数
        self.flipud = flipud
        self.fliplr = fliplr
        self.erasing = erasing
        self.crop_fraction = crop_fraction
        
        # 优化器参数
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        
        # 学习率调度参数
        self.lr_scheduler = lr_scheduler
        self.lr_decay = lr_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        
        # 验证参数
        self.val_frequency = val_frequency
        self.val_iters = val_iters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # 设备参数
        self.device = device
        self.multi_gpu = multi_gpu
        self.gpu_memory_optimization = gpu_memory_optimization
        self.data_loading_optimization = data_loading_optimization
        
        # 其他高级参数
        self.close_mosaic = close_mosaic
        self.single_cls = single_cls
        self.cache = cache
        self.resume = resume
        self.workers = workers
        self.optimizer = optimizer
        self.amp = amp
        self.progress_callback = progress_callback
        
        # 设备检测
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def _create_model(self):
        """創建YOLO模型"""
        try:
            if self.progress_callback:
                self.progress_callback("📥 載入模型...")
            
            from ultralytics import YOLO
            
            # 檢查模型路徑
            model_path_obj = Path(self.model_path) if self.model_path else None
            
            if not model_path_obj or not model_path_obj.exists():
                # 嘗試從 Model_file/standard 目錄尋找
                standard_dir = Path.cwd() / 'Model_file' / 'standard'
                if standard_dir.exists():
                    model_files = list(standard_dir.glob('*.pt'))
                    if model_files:
                        model_path_obj = model_files[0]
                        if self.progress_callback:
                            self.progress_callback(f"✅ 找到模型文件: {model_path_obj.name}")
                    else:
                        raise FileNotFoundError(f"在 {standard_dir} 中未找到 .pt 文件")
                else:
                    raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 載入模型 - 支持 YAML 配置，增加錯誤處理
            try:
                if str(model_path_obj).endswith('.yaml'):
                    # 使用 YAML 配置文件創建模型
                    model = YOLO(str(model_path_obj))
                    if self.progress_callback:
                        self.progress_callback(f"✅ 使用YAML配置: {model_path_obj}")
                elif str(model_path_obj).startswith("ultralytics/"):
                    # 使用內建 YAML 配置創建模型
                    model = YOLO(model=str(model_path_obj))
                    if self.progress_callback:
                        self.progress_callback(f"✅ 使用內建YAML配置: {model_path_obj}")
                else:
                    # 使用預訓練模型
                    model = YOLO(str(model_path_obj))
                
                # 驗證模型是否正確載入
                if model is None:
                    raise ValueError("模型載入失敗，返回None")
                
                if self.progress_callback:
                    if str(model_path_obj).endswith('.yaml'):
                        self.progress_callback(f"✅ YAML配置文件載入成功: {model_path_obj.name}")
                    elif str(model_path_obj).startswith("ultralytics/"):
                        self.progress_callback(f"✅ 內建YAML配置載入成功: {model_path_obj}")
                    else:
                        self.progress_callback(f"✅ 預訓練模型載入成功: {model_path_obj.name}")
                
                return model
                
            except Exception as yaml_error:
                if self.progress_callback:
                    self.progress_callback(f"❌ YAML模型載入失敗: {yaml_error}")
                    self.progress_callback("🛑 停止訓練，請檢查YAML模型配置")
                
                # YAML模型載入失敗時，直接拋出錯誤，不進行回退
                raise ValueError(f"YAML模型載入失敗: {yaml_error}。請檢查模型配置或使用預訓練模型(.pt文件)。")
            
        except Exception as e:
            raise RuntimeError(f"載入模型時發生錯誤: {e}") from e
    
    def _detect_device(self):
        """智能設備檢測"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = '0'  # 使用第一個GPU
            if self.progress_callback:
                self.progress_callback(f"🎯 使用GPU訓練: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            if self.progress_callback:
                self.progress_callback("🎯 使用CPU訓練")
        
        return device
    
    def _get_model_input_channels(self):
        """獲取模型的輸入通道數"""
        try:
            if self.model_path and Path(self.model_path).exists():
                # 載入模型並檢查第一層的輸入通道數
                model_data = torch.load(self.model_path, map_location='cpu', weights_only=False)
                
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
                
                # 如果無法檢測，根據模型文件名推測
                model_name = Path(self.model_path).stem.lower()
                if '4channel' in model_name or '4ch' in model_name:
                    return 4
                elif '3channel' in model_name or '3ch' in model_name:
                    return 3
                else:
                    return 3  # 默認3通道
            else:
                # 如果沒有模型文件，根據配置推測
                if hasattr(self, 'config_path') and self.config_path:
                    config_path = Path(self.config_path)
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '4channel' in content.lower() or 'channels: 4' in content:
                                return 4
                return 3  # 默認3通道
        except Exception as e:
            if hasattr(self, 'progress_callback') and self.progress_callback:
                self.progress_callback(f"⚠️ 檢測模型通道數失敗: {e}")
            return 3  # 默認3通道
    
    def _has_test_set(self):
        """檢查是否有測試集"""
        try:
            if not self.config_path or not Path(self.config_path).exists():
                if self.progress_callback:
                    self.progress_callback(f"⚠️ 配置文件不存在: {self.config_path}")
                return False
            
            # 讀取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 檢查是否有測試路徑配置
            test_path = config_data.get('test', '')
            if not test_path:
                if self.progress_callback:
                    self.progress_callback("ℹ️ 配置文件中未找到 'test' 路徑配置")
                return False
            
            if self.progress_callback:
                self.progress_callback(f"📋 配置文件中的測試集路徑: {test_path}")
            
            # 處理相對路徑 - 相對於配置文件所在目錄
            config_dir = Path(self.config_path).parent
            if not Path(test_path).is_absolute():
                test_path_obj = config_dir / test_path
            else:
                test_path_obj = Path(test_path)
            
            if self.progress_callback:
                self.progress_callback(f"📂 測試集完整路徑: {test_path_obj}")
            
            # 檢查測試路徑是否存在
            if not test_path_obj.exists():
                if self.progress_callback:
                    self.progress_callback(f"❌ 測試集路徑不存在: {test_path_obj}")
                return False
            
            # 檢查測試路徑中是否有圖像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']
            has_images = False
            image_count = 0
            for ext in image_extensions:
                files = list(test_path_obj.glob(f'*{ext}')) + list(test_path_obj.glob(f'*{ext.upper()}'))
                if files:
                    has_images = True
                    image_count += len(files)
            
            if self.progress_callback:
                if has_images:
                    self.progress_callback(f"✅ 測試集檢測成功: 找到 {image_count} 個圖像文件")
                else:
                    self.progress_callback(f"❌ 測試集路徑存在但未找到圖像文件: {test_path_obj}")
            
            return has_images
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"⚠️ 檢查測試集時出錯: {e}")
            return False
    
    def train(self, progress_callback=None, log_callback=None):
        """
        執行標準訓練
        
        Args:
            progress_callback (callable): 進度回調函數
            log_callback (callable): 日誌回調函數
            
        Returns:
            dict: 訓練結果
        """
        try:
            if log_callback:
                log_callback("🎯 開始標準模型訓練...")
                log_callback(f"配置: {self.config_path}")
                log_callback(f"模型文件: {self.model_path}")
                log_callback(f"輪數: {self.epochs}")
                log_callback(f"學習率: {self.learning_rate}")
                log_callback(f"批次大小: {self.batch_size}")
                log_callback(f"圖像大小: {self.imgsz}")
                log_callback(f"縮放比例: {self.scale}")
                log_callback(f"Mosaic: {self.mosaic}")
                log_callback(f"Mixup: {self.mixup}")
                log_callback(f"Copy-paste: {self.copy_paste}")
            
            # 檢查配置文件是否存在
            if not Path(self.config_path).exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            # 載入模型
            model = self._create_model()
            
            if log_callback:
                log_callback("🚀 開始訓練...")
            
            # 智能設備檢測
            device = self._detect_device()
            
            # 檢測模型輸入通道數
            input_channels = self._get_model_input_channels()
            
            # 讀取數據集配置以確定通道類型
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    dataset_config = yaml.safe_load(f)
                channels = dataset_config.get('channels', 3)
                channel_type = 'RGBD' if channels == 4 else 'RGB'
            except Exception as e:
                if log_callback:
                    log_callback(f"⚠️ 讀取數據集配置失敗: {e}")
                channel_type = 'RGB'
            
            # 生成資料夾名稱格式：{檔名}_{通道類型}_{epoch數}epochs_{時間戳}
            from datetime import datetime
            import re
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            if self.model_path:
                model_name = Path(self.model_path).stem  # 獲取模型文件名（不含擴展名）
                # 移除模型名稱中已有的通道類型後綴 (避免重複)
                # 例如: yolo12n_RGBD -> yolo12n, yolo12n_RGB -> yolo12n
                model_name = re.sub(r'_(RGBD|RGB|4ch|3ch)$', '', model_name, flags=re.IGNORECASE)
                base_folder_name = f'{model_name}_{channel_type}_{self.epochs}epochs_{timestamp}'
            else:
                # 如果沒有模型路徑，使用默認格式
                base_folder_name = f'{channel_type}_{self.epochs}epochs_{timestamp}'
            
            # 檢查資料夾是否已存在，如果存在則添加序號
            folder_name = self._get_unique_folder_name(base_folder_name)
            
            # 定義主資料夾路徑（用於後續操作）
            main_folder = Path('runs') / folder_name
            
            if log_callback:
                log_callback(f"📁 輸出資料夾: {main_folder}")
            
            # 使用新的訓練模式，啟用ultralytics內建圖表生成，增加錯誤處理
            try:
                results = model.train(
                    data=self.config_path,
                    epochs=self.epochs,
                    device=device,
                    project='runs',  # 使用 runs 作為基礎目錄
                    name=folder_name,  # 使用自定義名稱
                    exist_ok=True,
                    lr0=self.learning_rate,
                    batch=self.batch_size,
                    imgsz=self.imgsz,
                    scale=self.scale,
                    mosaic=self.mosaic,
                    mixup=self.mixup,
                    copy_paste=self.copy_paste,
                    hsv_h=self.hsv_h,
                    hsv_s=self.hsv_s,
                    hsv_v=self.hsv_v,
                    bgr=self.bgr,
                    auto_augment=self.auto_augment,
                    # 新增的幾何變換參數
                    degrees=self.degrees,
                    translate=self.translate,
                    shear=self.shear,
                    perspective=self.perspective,
                    # 新增的翻轉和裁剪參數
                    flipud=self.flipud,
                    fliplr=self.fliplr,
                    erasing=self.erasing,
                    crop_fraction=self.crop_fraction,
                    # 新增的訓練控制參數
                    close_mosaic=self.close_mosaic,
                    workers=self.workers,
                    optimizer=self.optimizer,
                    amp=self.amp,
                    verbose=False,
                    save_period=self.save_period,  # 檢查點保存週期
                    plots=True,   # 啟用ultralytics內建圖表生成
                    save=True,   # 啟用保存功能
                    show=False,  # 不顯示圖表，只保存
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    show_labels=False,
                    show_conf=False,
                    line_width=0,
                    visualize=False,
                    augment=False,
                    agnostic_nms=False,
                    max_det=300,
                    half=False,
                    dnn=False,
                    retina_masks=False
                )
            except ValueError as ve:
                if "too many values to unpack" in str(ve):
                    if log_callback:
                        log_callback(f"❌ YAML模型解包錯誤: {ve}")
                        log_callback("🛑 停止訓練，YAML模型配置有問題")
                        log_callback("💡 建議: 請檢查YAML模型配置或使用預訓練模型(.pt文件)")
                    
                    # YAML模型解包錯誤時，直接停止訓練
                    raise ValueError(f"YAML模型解包錯誤: {ve}。請檢查YAML模型配置或使用預訓練模型(.pt文件)。")
                else:
                    raise ve  # 重新拋出非解包錯誤
            except TypeError as te:
                if "plot_images()" in str(te) or "labels" in str(te):
                    if log_callback:
                        log_callback(f"⚠️ 圖表生成錯誤: {te}")
                        log_callback("🔄 嘗試禁用圖表生成重新訓練...")
                    
                    # 重新嘗試訓練，禁用圖表生成
                    try:
                        results = model.train(
                            data=self.config_path,
                            epochs=self.epochs,
                            device=device,
                            project='runs',  # 使用 runs 作為基礎目錄
                            name=folder_name,  # 使用自定義名稱
                            exist_ok=True,
                            lr0=self.learning_rate,
                            batch=self.batch_size,
                            imgsz=self.imgsz,
                            scale=self.scale,
                            mosaic=self.mosaic,
                            mixup=self.mixup,
                            copy_paste=self.copy_paste,
                            hsv_h=self.hsv_h,
                            hsv_s=self.hsv_s,
                            hsv_v=self.hsv_v,
                            bgr=self.bgr,
                            auto_augment=self.auto_augment,
                            # 新增的幾何變換參數
                            degrees=self.degrees,
                            translate=self.translate,
                            shear=self.shear,
                            perspective=self.perspective,
                            # 新增的翻轉和裁剪參數
                            flipud=self.flipud,
                            fliplr=self.fliplr,
                            erasing=self.erasing,
                            crop_fraction=self.crop_fraction,
                            # 新增的訓練控制參數
                            close_mosaic=self.close_mosaic,
                            workers=self.workers,
                            optimizer=self.optimizer,
                            amp=self.amp,
                            verbose=False,
                            save_period=self.save_period,  # 檢查點保存週期
                            plots=False,  # 禁用圖表生成
                            save=True,
                            show=False,
                            save_txt=False,
                            save_conf=False,
                            save_crop=False,
                            show_labels=False,
                            show_conf=False,
                            line_width=0,
                            visualize=False,
                            augment=False,
                            agnostic_nms=False,
                            max_det=300,
                            half=False,
                            dnn=False,
                            retina_masks=False
                        )
                        if log_callback:
                            log_callback("✅ 重新訓練成功（已禁用圖表生成）")
                    except Exception as retry_error:
                        if log_callback:
                            log_callback(f"❌ 重新訓練失敗: {retry_error}")
                        raise retry_error
                else:
                    raise te  # 重新拋出非plot_images錯誤
            
            if log_callback:
                log_callback("✅ 標準訓練完成!")
            
            # 檢查是否有測試集，如果有才進行測試驗證
            test_results = None
            has_test = self._has_test_set()
            if log_callback:
                log_callback(f"🔍 測試集檢查結果: {'✅ 發現測試集' if has_test else '❌ 未發現測試集'}")
            
            if has_test:
                if log_callback:
                    log_callback("🔍 開始測試資料集驗證...")
                
                test_results = self._validate_on_test_set(model, main_folder, progress_callback, log_callback)
                
                if log_callback:
                    log_callback(f"📊 測試結果: {test_results}")
                    log_callback("📈 測試驗證圖表已生成並保存到 runs/ 目錄")
            else:
                if log_callback:
                    log_callback("ℹ️ 未檢測到測試集，跳過測試驗證")
            
            # 整理檢查點文件
            self._organize_checkpoints(main_folder)
            
            # ultralytics已自動生成圖表，無需額外生成
            if log_callback:
                log_callback("📊 ultralytics已自動生成訓練圖表")
                log_callback(f"📁 結果保存在: {main_folder}")
                log_callback(f"   📂 訓練結果: {main_folder}/train")
                log_callback(f"   📂 測試結果: {main_folder}/test")
                log_callback("📈 圖表文件包括: results.png, confusion_matrix.png, F1_curve.png, P_curve.png, PR_curve.png, R_curve.png, labels.jpg, labels_correlogram.jpg, train_batch*.jpg, val_batch*.jpg")
            
            return results
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ 訓練錯誤: {e}")
            raise
    
    def _organize_checkpoints(self, main_folder):
        """整理檢查點文件到History_pt資料夾"""
        try:
            import shutil
            
            # 訓練結果目錄
            train_dir = main_folder / 'train'
            if not train_dir.exists():
                return
            
            # 創建History_pt資料夾
            history_pt_dir = train_dir / 'History_pt'
            history_pt_dir.mkdir(exist_ok=True)
            
            # 檢查 weights 資料夾中的檢查點文件
            weights_dir = train_dir / 'weights'
            if weights_dir.exists():
                # 從 weights 資料夾移動檢查點文件
                checkpoint_files = list(weights_dir.glob('*.pt'))
                for pt_file in checkpoint_files:
                    if pt_file.name not in ['best.pt', 'last.pt']:  # 保留重要的檢查點
                        dest_path = history_pt_dir / pt_file.name
                        shutil.move(str(pt_file), str(dest_path))
                        print(f"📁 移動檢查點: {pt_file.name} -> History_pt/")
            
            # 也檢查 train 資料夾根目錄中的檢查點文件
            checkpoint_files = list(train_dir.glob('*.pt'))
            for pt_file in checkpoint_files:
                if pt_file.name not in ['best.pt', 'last.pt']:  # 保留重要的檢查點
                    dest_path = history_pt_dir / pt_file.name
                    shutil.move(str(pt_file), str(dest_path))
                    print(f"📁 移動檢查點: {pt_file.name} -> History_pt/")
            
            # 檢查是否有移動的文件
            history_files = list(history_pt_dir.glob('*.pt'))
            if history_files:
                print(f"✅ 檢查點文件已整理到: {history_pt_dir} (共 {len(history_files)} 個文件)")
            else:
                print("ℹ️ 未找到需要整理的歷史檢查點文件")
            
        except Exception as e:
            print(f"⚠️ 整理檢查點文件時出錯: {e}")
    
    def _validate_on_test_set(self, model, main_folder, progress_callback=None, log_callback=None):
        """使用測試資料集驗證模型"""
        try:
            if progress_callback:
                progress_callback("🔍 載入測試資料集...")
            
            if log_callback:
                log_callback("📂 準備進行測試驗證...")
                log_callback(f"📋 配置文件: {self.config_path}")
                log_callback(f"📂 輸出目錄: {main_folder}/test")
            
            # 使用YOLO的驗證功能
            if log_callback:
                log_callback("🚀 開始執行測試驗證...")
            
            test_results = model.val(
                data=self.config_path,
                split='test',
                device=self._detect_device(),
                project=str(main_folder),  # 使用主資料夾路徑
                name='test',  # 測試結果放在 test 子資料夾
                verbose=True,  # 啟用詳細輸出
                save_json=True,  # 保存JSON結果
                save_hybrid=False,
                plots=True,  # 啟用圖表生成
                save=True,  # 保存結果
                show=False,
                save_txt=True,  # 保存文本結果
                save_conf=True,  # 保存置信度
                save_crop=False,
                show_labels=True,  # 顯示標籤
                show_conf=True,  # 顯示置信度
                line_width=3,  # 增加線寬
                augment=False,
                agnostic_nms=False,
                max_det=300,
                half=False,
                dnn=False,
                retina_masks=False
            )
            
            if log_callback:
                log_callback("📊 測試驗證完成，檢查生成的文件...")
            
            if progress_callback:
                progress_callback("✅ 測試驗證完成")
            
            # 提取關鍵指標
            results_summary = {
                'mAP50': test_results.box.map50 if hasattr(test_results.box, 'map50') else 0.0,
                'mAP50-95': test_results.box.map if hasattr(test_results.box, 'map') else 0.0,
                'precision': test_results.box.mp if hasattr(test_results.box, 'mp') else 0.0,
                'recall': test_results.box.mr if hasattr(test_results.box, 'mr') else 0.0,
                'f1_score': test_results.box.f1 if hasattr(test_results.box, 'f1') else 0.0
            }
            
            if log_callback:
                log_callback(f"📊 測試集驗證結果:")
                log_callback(f"   mAP50: {results_summary['mAP50']:.4f}")
                log_callback(f"   mAP50-95: {results_summary['mAP50-95']:.4f}")
                log_callback(f"   Precision: {results_summary['precision']:.4f}")
                log_callback(f"   Recall: {results_summary['recall']:.4f}")
                log_callback(f"   F1-Score: {results_summary['f1_score']:.4f}")
            
            # 檢查並確保可視化文件被正確生成
            test_folder = main_folder / 'test'
            if log_callback:
                log_callback(f"🔍 檢查測試結果資料夾: {test_folder}")
            
            # 檢查測試資料夾中的文件
            if test_folder.exists():
                files = list(test_folder.glob('*'))
                if log_callback:
                    log_callback(f"📁 測試資料夾中找到 {len(files)} 個文件")
                    for file in files:
                        log_callback(f"   📄 {file.name}")
            
            self._ensure_visualization_files(test_folder)
            
            return results_summary
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ 測試驗證失敗: {e}")
                log_callback(f"🔍 錯誤類型: {type(e).__name__}")
                log_callback("💡 可能的原因:")
                log_callback("   1. 測試集標籤文件缺失或格式錯誤")
                log_callback("   2. 測試集圖像與標籤不匹配")
                log_callback("   3. 模型文件損壞或不兼容")
                log_callback("   4. GPU內存不足")
                log_callback("")
                log_callback("📋 檢查清單:")
                log_callback(f"   - 配置文件: {self.config_path}")
                log_callback("   - 測試集圖像路徑: images/test")
                log_callback("   - 測試集標籤路徑: labels/test")
                log_callback("   - 確認標籤文件格式: class x_center y_center width height")
                
            if progress_callback:
                progress_callback(f"❌ 測試驗證失敗: {e}")
                
            # 返回空結果但包含錯誤信息
            return {
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _ensure_visualization_files(self, test_folder):
        """確保可視化文件被正確生成"""
        try:
            # 檢查測試資料夾
            if not test_folder.exists():
                if self.progress_callback:
                    self.progress_callback("⚠️ 測試資料夾不存在，跳過可視化檢查...")
                return
            
            # 檢查是否有可視化文件
            visualization_files = list(test_folder.glob('*.jpg')) + list(test_folder.glob('*.png'))
            
            if not visualization_files:
                if self.progress_callback:
                    self.progress_callback("⚠️ 未找到可視化文件，嘗試手動生成...")
                
                # 嘗試手動生成可視化文件
                self._generate_manual_visualizations(test_folder)
            else:
                if self.progress_callback:
                    self.progress_callback(f"✅ 找到 {len(visualization_files)} 個可視化文件")
                    for file in visualization_files:
                        self.progress_callback(f"   📁 {file.name}")
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"⚠️ 檢查可視化文件時出錯: {e}")
    
    def _generate_manual_visualizations(self, output_dir):
        """手動生成可視化文件"""
        try:
            import cv2
            import numpy as np
            
            # 創建 visualizations 子資料夾
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            if self.progress_callback:
                self.progress_callback(f"📁 創建可視化資料夾: {viz_dir}")
            
            # 獲取測試集圖像路徑
            test_images_dir = Path(self.config_path).parent / 'images' / 'test'
            if not test_images_dir.exists():
                if self.progress_callback:
                    self.progress_callback(f"⚠️ 測試集圖像目錄不存在: {test_images_dir}")
                return
            
            # 獲取所有測試圖像
            image_files = []
            for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(test_images_dir.glob(f'*{ext}'))
                image_files.extend(test_images_dir.glob(f'*{ext.upper()}'))
            
            if not image_files:
                if self.progress_callback:
                    self.progress_callback("⚠️ 未找到測試圖像文件")
                return
            
            # 限制處理的圖像數量（避免過多）
            max_images = min(10, len(image_files))
            selected_images = image_files[:max_images]
            
            if self.progress_callback:
                self.progress_callback(f"🖼️ 處理 {len(selected_images)} 張測試圖像...")
            
            # 為每張圖像生成可視化
            for i, img_path in enumerate(selected_images):
                try:
                    # 載入圖像
                    if img_path.suffix.lower() == '.npy':
                        image = np.load(img_path)
                        # 如果是4通道，取前3通道用於可視化
                        if len(image.shape) == 3 and image.shape[2] == 4:
                            image = image[:, :, :3]
                    else:
                        image = cv2.imread(str(img_path))
                    
                    if image is None:
                        continue
                    
                    # 生成可視化文件名
                    output_name = f"test_visualization_{i+1:03d}.jpg"
                    output_path = viz_dir / output_name
                    
                    # 保存圖像
                    cv2.imwrite(str(output_path), image)
                    
                    if self.progress_callback:
                        self.progress_callback(f"   ✅ 生成: {output_name}")
                        
                except Exception as e:
                    if self.progress_callback:
                        self.progress_callback(f"   ❌ 處理 {img_path.name} 失敗: {e}")
            
            if self.progress_callback:
                self.progress_callback(f"📁 可視化文件已保存到: {viz_dir}")
                
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"⚠️ 手動生成可視化文件失敗: {e}")
    
    def _get_unique_folder_name(self, base_name):
        """生成唯一的資料夾名稱，如果重複則添加序號"""
        # 檢查 runs 目錄是否存在
        runs_dir = Path('runs')
        if not runs_dir.exists():
            return base_name
        
        # 檢查基礎名稱是否已存在
        if not (runs_dir / base_name).exists():
            return base_name
        
        # 如果存在，添加序號
        counter = 1
        while True:
            unique_name = f"{base_name}({counter})"
            if not (runs_dir / unique_name).exists():
                return unique_name
            counter += 1


class ConfigDetector:
    """配置文件偵測器"""
    
    @staticmethod
    def detect_configs(dataset_dir="Dataset"):
        """偵測Dataset目錄中的所有配置文件"""
        config_files = []
        dataset_path = Path(dataset_dir)
        
        if dataset_path.exists():
            for config_file in dataset_path.glob("*/data_config.yaml"):
                dataset_name = config_file.parent.name
                config_files.append((dataset_name, str(config_file)))
        
        return config_files
    
    @staticmethod
    def get_available_datasets(dataset_dir="Dataset"):
        """獲取所有可用的數據集名稱"""
        config_files = ConfigDetector.detect_configs(dataset_dir)
        return [dataset_name for dataset_name, _ in config_files]
    
    @staticmethod
    def validate_config(config_path):
        """驗證配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            return {
                'valid': True,
                'channels': config_data.get('channels', '未知'),
                'train_path': config_data.get('train', '未知'),
                'val_path': config_data.get('val', '未知'),
                'test_path': config_data.get('test', '未知'),
                'nc': config_data.get('nc', '未知'),
                'names': config_data.get('names', [])
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


def main():
    """命令行使用示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO 標準訓練器')
    parser.add_argument('--config', required=True, help='配置文件路徑')
    parser.add_argument('--model', required=True, help='模型文件路徑')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--imgsz', type=int, default=640, help='圖像大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='學習率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--scale', type=float, default=0.5, help='縮放比例')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic數據增強')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup數據增強')
    parser.add_argument('--copy_paste', type=float, default=0.1, help='Copy-paste數據增強')
    
    args = parser.parse_args()
    
    # 創建訓練器
    trainer = YOLOStandardTrainer(
        config_path=args.config,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        scale=args.scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste
    )
    
    # 定義回調函數
    def progress_callback(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_callback(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    # 開始訓練
    try:
        results = trainer.train(progress_callback=progress_callback, log_callback=log_callback)
        print("✅ 訓練完成!")
        return results
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        return None


if __name__ == '__main__':
    main()