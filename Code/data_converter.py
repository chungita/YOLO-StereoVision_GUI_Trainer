"""
數據集轉換器
支持兩種模式：
1. StereoPreprocessor：處理立體視覺數據（左視圖、右視圖、視差圖）
2. RGBPreprocessor：處理YOLO格式數據（RGB或RGBD）

統一輸出格式：Dataset/dataset_{type}_{timestamp}
"""

import os
import sys
import shutil
import yaml
import cv2
import numpy as np
import random
import struct
import re
from datetime import datetime
from pathlib import Path

# 添加根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from config import DATA_CONFIG, PREDEFINED_CLASSES

class StereoPreprocessor:
    """處理立體視覺數據的預處理器，支持左右視圖和視差圖處理"""
    
    def __init__(self, source_path=None, output_path=None, folder_count_limit=None, **kwargs):
        """
        初始化立體視覺預處理器
        
        Args:
            source_path (str): 源數據路徑
            output_path (str): 輸出路徑
            folder_count_limit (int): 限制處理的資料夾數量，None表示處理全部
        """
        if source_path is None:
            source_path = DATA_CONFIG['source_path']
        if output_path is None:
            # 使用統一格式：Dataset/dataset_{type}_{timestamp}
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"Dataset/dataset_Stereo_{timestamp}"
            
        # 設置基本屬性
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.folder_count_limit = folder_count_limit
        # 支持自定义分割比例，如果未提供则使用默认值
        self.train_ratio = kwargs.get('train_ratio', DATA_CONFIG['train_ratio'])
        self.val_ratio = kwargs.get('val_ratio', DATA_CONFIG['val_ratio'])
        self.test_ratio = kwargs.get('test_ratio', DATA_CONFIG['test_ratio'])
        self.left_pattern = "Img0_*"  # 左視圖
        self.right_pattern = "Img1_*"  # 右視圖
        self.disparity_pattern = "Disparity_*"  # 視差圖
        self.channels = 3  # 每個圖像3通道，分別儲存
        
        # 創建輸出目錄結構
        self._create_output_directories()
        
        print(f"配置為處理立體視覺數據 (分別儲存左視圖、右視圖、視差圖)")
        print(f"輸出路徑: {self.output_path}")
    
    def process_single_video(self, video_folder):
        """處理單個視頻文件夾的立體視覺數據"""
        img_folder = video_folder / 'Img'  # 圖像和視差圖都在Img文件夾內
        
        if not img_folder.exists():
            print(f"跳過 {video_folder.name}: 缺少Img文件夾")
            return []
        
        # 獲取左視圖文件
        left_files = list(img_folder.glob(f'{self.left_pattern}.png')) + list(img_folder.glob(f'{self.left_pattern}.jpg'))
        
        processed_data = []
        for left_file in left_files:
            # 構造對應的右視圖文件名
            left_name = left_file.stem  # Img0_1
            right_name = left_name.replace('Img0', 'Img1')  # Img1_1
            right_file = img_folder / f"{right_name}.png"
            if not right_file.exists():
                right_file = img_folder / f"{right_name}.jpg"
            
            # 構造對應的視差圖文件名
            disparity_name = left_name.replace('Img0', 'Disparity')  # Disparity_1
            disparity_file = img_folder / f"{disparity_name}.pfm"
            if not disparity_file.exists():
                disparity_file = img_folder / f"{disparity_name}.png"
            
            if not right_file.exists() or not disparity_file.exists():
                continue  # 跳過缺少文件的樣本
            
            processed_data.append({
                'left_image': left_file,
                'right_image': right_file,
                'disparity': disparity_file,
                'video': video_folder.name
            })
        
        return processed_data
    
    def process_single_folder(self, folder_path):
        """處理單一資料夾下的立體視覺圖片"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ 資料夾不存在或不是目錄: {folder_path}")
            return []
        
        print(f"📁 處理單一資料夾: {folder_path}")
        
        # 檢查必需的子資料夾
        img_folder = folder_path / 'Img'
        
        if not img_folder.exists() or not img_folder.is_dir():
            print(f"❌ 缺少必需的子資料夾: Img")
            return []
        
        # 支持的圖片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # 收集所有左視圖文件
        left_files = []
        for ext in image_extensions:
            left_files.extend(img_folder.glob(f'Img0_*{ext}'))
            left_files.extend(img_folder.glob(f'Img0_*{ext.upper()}'))
        
        # 去重複
        left_files = list(set(left_files))
        
        if not left_files:
            print(f"⚠️ 在Img資料夾中未找到任何左視圖文件: {img_folder}")
            return []
        
        print(f"📊 在Img資料夾中找到 {len(left_files)} 個左視圖文件")
        
        processed_data = []
        for left_file in left_files:
            # 構造對應的右視圖文件名
            left_name = left_file.stem  # Img0_1
            right_name = left_name.replace('Img0', 'Img1')  # Img1_1
            right_file = img_folder / f"{right_name}.png"
            if not right_file.exists():
                right_file = img_folder / f"{right_name}.jpg"
            
            # 構造對應的視差圖文件名
            disparity_name = left_name.replace('Img0', 'Disparity')  # Disparity_1
            disparity_file = img_folder / f"{disparity_name}.pfm"
            if not disparity_file.exists():
                disparity_file = img_folder / f"{disparity_name}.png"
            
            if right_file.exists() and disparity_file.exists():
                processed_data.append({
                    'left_image': left_file,
                    'right_image': right_file,
                    'disparity': disparity_file,
                    'video': folder_path.name
                })
            else:
                missing_files = []
                if not right_file.exists():
                    missing_files.append(f"右視圖({right_name})")
                if not disparity_file.exists():
                    missing_files.append(f"視差圖({disparity_name})")
                print(f"⚠️ 跳過 {left_file.name}: 缺少 {', '.join(missing_files)}")
        
        print(f"✅ 成功處理 {len(processed_data)} 個有效立體視覺樣本")
        return processed_data
    
    def update_config_file(self, mode='auto'):
        """更新立體視覺配置文件"""
        config_path = self.output_path / 'data_config.yaml'
        
        # 根據模式設置描述
        if mode == 'forest':
            description = '立體視覺數據集 - Forest格式 (分別儲存左視圖、右視圖、視差圖)'
        else:
            description = '立體視覺數據集 - 單一資料夾模式 (分別儲存左視圖、右視圖、視差圖)'
        
        config_data = {
            'path': str(self.output_path.absolute()),
            'train': 'Img0/train',  # 使用左視圖作為主要圖像路徑
            'val': 'Img0/val', 
            'test': 'Img0/test',
            'source_path': str(self.source_path),
            'channels': self.channels,
            'left_pattern': self.left_pattern,
            'right_pattern': self.right_pattern,
            'disparity_pattern': self.disparity_pattern,
            'description': description,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_path': str(self.output_path),
            'mode': mode,
            'folder_count_limit': self.folder_count_limit,
            'use_stereo': True,
            'stereo_folders': {
                'Img0': '左視圖資料夾',
                'Img1': '右視圖資料夾', 
                'Disparity': '視差圖資料夾'
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
        print(f"✅ 立體視覺配置文件已創建: {config_path}")
        
        return config_path
    
    def _create_output_directories(self):
        """創建立體視覺數據的輸出目錄結構"""
        directories = [
            # 左視圖資料夾
            self.output_path / 'Img0' / 'train',
            self.output_path / 'Img0' / 'val', 
            self.output_path / 'Img0' / 'test',
            # 右視圖資料夾
            self.output_path / 'Img1' / 'train',
            self.output_path / 'Img1' / 'val', 
            self.output_path / 'Img1' / 'test',
            # 視差圖資料夾
            self.output_path / 'Disparity' / 'train',
            self.output_path / 'Disparity' / 'val', 
            self.output_path / 'Disparity' / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_all_data(self, mode='auto'):
        """處理所有立體視覺數據並分割為訓練/驗證/測試集"""
        print(f"\n開始處理立體視覺數據...")
        
        # 收集所有數據
        all_data = []
        
        if mode == 'auto':
            # 自動檢測模式（排序確保一致性）
            video_folders = sorted([f for f in self.source_path.iterdir() if f.is_dir() and f.name.startswith('Forest_Video_')])
            
            if video_folders:
                print(f"🔍 檢測到Forest格式資料夾，使用Forest模式")
                mode = 'forest'
            else:
                print(f"🔍 未檢測到Forest格式資料夾，使用單一資料夾模式")
                mode = 'single'
        
        if mode == 'forest':
            # Forest格式處理（排序確保一致性）
            video_folders = sorted([f for f in self.source_path.iterdir() if f.is_dir() and f.name.startswith('Forest_Video_')])
            
            # 應用資料夾數量限制
            if self.folder_count_limit is not None and self.folder_count_limit > 0:
                original_count = len(video_folders)
                if self.folder_count_limit < original_count:
                    video_folders = video_folders[:self.folder_count_limit]
                    print(f"📊 資料夾數量限制: {original_count} -> {len(video_folders)} 個資料夾")
            
            for video_folder in sorted(video_folders):
                video_data = self.process_single_video(video_folder)
                all_data.extend(video_data)
            
            print(f"📊 Forest格式: 收集到 {len(all_data)} 個有效立體視覺樣本")
            
        elif mode == 'single':
            # 單一資料夾格式處理
            single_data = self.process_single_folder(self.source_path)
            all_data.extend(single_data)
            
            print(f"📊 單一資料夾格式: 收集到 {len(all_data)} 個有效立體視覺樣本")
        
        if len(all_data) == 0:
            print("❌ 沒有找到有效的立體視覺數據樣本")
            return
        
        # 分割數據集
        train_data, val_data, test_data = self._split_dataset(all_data)
        
        print(f"📊 數據分割: 訓練{len(train_data)} | 驗證{len(val_data)} | 測試{len(test_data)}")
        
        # 複製文件到對應目錄
        print(f"📁 開始複製立體視覺文件...")
        self._copy_files(train_data, 'train')
        self._copy_files(val_data, 'val')
        self._copy_files(test_data, 'test')
        
        # 更新配置文件
        self.update_config_file(mode)
        
        print(f"✅ 立體視覺數據處理完成! 總計: {len(train_data) + len(val_data) + len(test_data)} 個樣本")
        
        # 驗證實際生成的文件數量
        self._verify_generated_files()
    
    def _split_dataset(self, all_data):
        """將數據集分割為訓練/驗證/測試集"""
        random.shuffle(all_data)
        
        total = len(all_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_data = all_data[:train_end]
        val_data = all_data[train_end:val_end]
        test_data = all_data[val_end:]
        
        return train_data, val_data, test_data
    
    def _copy_files(self, data_list, split_name):
        """複製立體視覺文件到對應的目錄"""
        # 創建各類圖像的目標目錄
        img0_dir = self.output_path / 'Img0' / split_name
        img1_dir = self.output_path / 'Img1' / split_name
        disparity_dir = self.output_path / 'Disparity' / split_name
        
        copied_count = 0
        for item in data_list:
            try:
                # 添加視頻文件夾前綴避免文件名衝突
                video_prefix = item['video']
                base_name = item['left_image'].stem  # Img0_1
                
                # 複製左視圖 (Img0)
                img0_name = f"{video_prefix}_{base_name}.png"
                img0_dest = img0_dir / img0_name
                shutil.copy2(item['left_image'], img0_dest)
                
                # 複製右視圖 (Img1)
                img1_name = f"{video_prefix}_{base_name.replace('Img0', 'Img1')}.png"
                img1_dest = img1_dir / img1_name
                shutil.copy2(item['right_image'], img1_dest)
                
                # 複製視差圖 (Disparity) - 保持原始格式
                disparity_original_ext = item['disparity'].suffix  # 保持原始副檔名 (.pfm 或 .png)
                disparity_name = f"{video_prefix}_{base_name.replace('Img0', 'Disparity')}{disparity_original_ext}"
                disparity_dest = disparity_dir / disparity_name
                shutil.copy2(item['disparity'], disparity_dest)
                
                copied_count += 1
            except Exception as e:
                print(f"  ❌ 複製失敗 {item['left_image'].name}: {str(e)}")
        
        print(f"  {split_name}: {copied_count}/{len(data_list)} 個立體視覺樣本")
    
    def _verify_generated_files(self):
        """驗證實際生成的立體視覺文件數量"""
        splits = ['train', 'val', 'test']
        total_samples = 0
        total_files = 0
        
        for split in splits:
            # 檢查各個資料夾
            img0_dir = self.output_path / 'Img0' / split
            img1_dir = self.output_path / 'Img1' / split
            disparity_dir = self.output_path / 'Disparity' / split
            
            if all([img0_dir.exists(), img1_dir.exists(), disparity_dir.exists()]):
                # 檢查各類圖像文件
                img0_files = list(img0_dir.glob('*.png')) + list(img0_dir.glob('*.jpg'))
                img1_files = list(img1_dir.glob('*.png')) + list(img1_dir.glob('*.jpg'))
                disparity_files = list(disparity_dir.glob('*.pfm')) + list(disparity_dir.glob('*.png'))  # 優先檢查 .pfm 文件
                
                samples = len(img0_files)
                
                print(f"  {split.upper()}: {samples} 樣本")
                print(f"    Img0 (左視圖): {len(img0_files)} 文件")
                print(f"    Img1 (右視圖): {len(img1_files)} 文件")
                print(f"    Disparity (視差圖): {len(disparity_files)} 文件")
                
                total_samples += samples
                total_files += len(img0_files) + len(img1_files) + len(disparity_files)
            else:
                missing_dirs = []
                if not img0_dir.exists():
                    missing_dirs.append("Img0")
                if not img1_dir.exists():
                    missing_dirs.append("Img1")
                if not disparity_dir.exists():
                    missing_dirs.append("Disparity")
                print(f"  ❌ {split.upper()} 集缺少目錄: {', '.join(missing_dirs)}")
        
        print(f"📊 總計: {total_samples} 立體視覺樣本, {total_files} 文件")

class RGBPreprocessor:
    """處理RGB圖像的預處理器，支持RGB和RGBD NPY文件生成"""
    
    def __init__(self, source_path=None, output_path=None, folder_count_limit=None, use_depth=True, **kwargs):
        """
        初始化預處理器
        
        Args:
            source_path (str): 源數據路徑
            output_path (str): 輸出路徑
            folder_count_limit (int): 限制處理的資料夾數量，None表示處理全部
            use_depth (bool): 是否使用深度圖生成RGBD NPY文件
        """
        if source_path is None:
            source_path = DATA_CONFIG['source_path']
        if output_path is None:
            # 使用統一格式：Dataset/dataset_{type}_{timestamp}
            timestamp = datetime.now().strftime("%Y%m%d")
            dataset_type = "RGBD" if use_depth else "RGB"
            output_path = f"Dataset/dataset_{dataset_type}_{timestamp}"
            
        # 設置基本屬性
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.folder_count_limit = folder_count_limit
        self.use_depth = use_depth
        # 支持自定义分割比例，如果未提供则使用默认值
        self.train_ratio = kwargs.get('train_ratio', DATA_CONFIG['train_ratio'])
        self.val_ratio = kwargs.get('val_ratio', DATA_CONFIG['val_ratio'])
        self.test_ratio = kwargs.get('test_ratio', DATA_CONFIG['test_ratio'])
        self.image_pattern = DATA_CONFIG['image_pattern']
        self.depth_pattern = "DepthGT_*"
        self.channels = 4 if use_depth else 3
        
        # 創建輸出目錄結構
        self._create_output_directories()
        
        if use_depth:
            print(f"✅ 配置為處理4通道RGBD數據 (RGB + 深度圖)")
        else:
            print(f"✅ 配置為處理3通道RGB數據")
        print(f"✅ 輸出路徑: {self.output_path}")
    
    def process_single_video(self, video_folder):
        """處理單個視頻文件夾的數據 - 支持RGB和RGBD處理"""
        img_folder = video_folder / 'Img'  # 圖像和深度圖都在Img文件夾內
        label_folder = video_folder / 'YOLO_Label'
        
        if not img_folder.exists() or not label_folder.exists():
            print(f"跳過 {video_folder.name}: 缺少Img或YOLO_Label文件夾")
            return []
        
        # 獲取Img0開頭的圖像文件
        image_files = list(img_folder.glob(f'{self.image_pattern}.png')) + list(img_folder.glob(f'{self.image_pattern}.jpg'))
        
        processed_data = []
        for img_file in image_files:
            # 構造對應的標籤文件名
            img_name = img_file.stem  # Img0_1
            label_file = label_folder / f"{img_name}.txt"
            
            if not label_file.exists():
                continue  # 跳過缺少標籤文件的樣本
            
            # 根據是否使用深度圖來處理
            if self.use_depth:
                # 構造對應的深度圖文件名 (在Img文件夾內)
                depth_file = img_folder / f"DepthGT_{img_name.split('_')[1]}.pfm"  # DepthGT_1.pfm
                
                if depth_file.exists():
                    processed_data.append({
                        'image': img_file,
                        'depth': depth_file,
                        'label': label_file,
                        'video': video_folder.name
                    })
                else:
                    pass  # 靜默跳過缺少深度圖的樣本
            else:
                # 不使用深度圖，直接處理RGB圖像
                processed_data.append({
                    'image': img_file,
                    'depth': None,
                    'label': label_file,
                    'video': video_folder.name
                })
        return processed_data
    
    def process_single_folder(self, folder_path):
        """處理單一資料夾下的圖片 - 必須包含Img、YOLO_Label、MOT_Label子資料夾"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ 資料夾不存在或不是目錄: {folder_path}")
            return []
        
        print(f"📁 處理單一資料夾: {folder_path}")
        
        # 檢查必需的子資料夾
        required_folders = ['Img', 'YOLO_Label', 'MOT_Label']
        missing_folders = []
        
        for folder_name in required_folders:
            folder = folder_path / folder_name
            if not folder.exists() or not folder.is_dir():
                missing_folders.append(folder_name)
        
        if missing_folders:
            print(f"❌ 缺少必需的子資料夾: {', '.join(missing_folders)}")
            print(f"📋 單一資料夾模式需要包含以下子資料夾:")
            for folder_name in required_folders:
                status = "✅" if folder_name not in missing_folders else "❌"
                print(f"   {status} {folder_name}/")
            return []
        
        print(f"✅ 找到所有必需的子資料夾:")
        for folder_name in required_folders:
            folder = folder_path / folder_name
            file_count = len(list(folder.iterdir())) if folder.exists() else 0
            print(f"   📁 {folder_name}/ ({file_count} 個文件)")
        
        # 從Img資料夾收集圖片文件
        img_folder = folder_path / 'Img'
        yolo_label_folder = folder_path / 'YOLO_Label'
        
        # 支持的圖片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # 收集所有圖片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(img_folder.glob(f'*{ext}'))
            image_files.extend(img_folder.glob(f'*{ext.upper()}'))
        
        # 去重複（避免大小寫重複）
        image_files = list(set(image_files))
        
        if not image_files:
            print(f"⚠️ 在Img資料夾中未找到任何圖片文件: {img_folder}")
            return []
        
        print(f"📊 在Img資料夾中找到 {len(image_files)} 個圖片文件")
        
        processed_data = []
        for img_file in image_files:
            # 構造對應的YOLO標籤文件名
            yolo_label_file = yolo_label_folder / f"{img_file.stem}.txt"
            
            if yolo_label_file.exists():
                processed_data.append({
                    'image': img_file,
                    'depth': None,  # 單一資料夾模式沒有深度圖
                    'label': yolo_label_file,
                    'video': folder_path.name
                })
            else:
                print(f"⚠️ 跳過 {img_file.name}: 缺少對應的YOLO標籤文件 {yolo_label_file.name}")
        
        print(f"✅ 成功處理 {len(processed_data)} 個有效樣本")
        return processed_data
    
    def update_config_file(self, class_names, mode='auto'):
        """更新配置文件"""
        # 在dataset文件夾內創建配置文件
        config_path = self.output_path / 'data_config.yaml'
        
        # 根據模式和深度圖選項設置描述
        if mode == 'forest':
            if self.use_depth:
                description = '4通道RGBD數據集 - Forest格式 (RGB + 深度圖)'
                channels = 4
            else:
                description = '3通道RGB數據集 - Forest格式'
                channels = 3
        else:
            if self.use_depth:
                description = '4通道RGBD數據集 - 單一資料夾模式 (RGB + 深度圖)'
                channels = 4
            else:
                description = '3通道RGB數據集 - 單一資料夾模式'
                channels = 3
        
        config_data = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(class_names),
            'names': list(class_names.values()),
            'source_path': str(self.source_path),
            'channels': channels,
            'image_pattern': self.image_pattern,
            'depth_pattern': self.depth_pattern if self.use_depth else None,
            'description': description,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_path': str(self.output_path),
            'mode': mode,
            'folder_count_limit': self.folder_count_limit,
            'use_depth': self.use_depth
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
        print(f"✅ YOLO配置文件已創建: {config_path}")
        
        return config_path
    
    def _create_output_directories(self):
        """創建YOLO格式的輸出目錄結構"""
        directories = [
            self.output_path / 'images' / 'train',
            self.output_path / 'images' / 'val', 
            self.output_path / 'images' / 'test',
            self.output_path / 'labels' / 'train',
            self.output_path / 'labels' / 'val',
            self.output_path / 'labels' / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_class_names(self):
        """從config/predefined_classes.txt加載類別名稱"""
        class_file = Path('config/predefined_classes.txt')
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 使用配置文件中的預定義類別
            classes = PREDEFINED_CLASSES['classes']
        
        # 創建類別ID到名稱的映射
        class_names = {i: name for i, name in enumerate(classes)}
        return class_names
    
    def process_all_data(self, mode='auto'):
        """
        處理所有數據並分割為訓練/驗證/測試集
        
        Args:
            mode (str): 處理模式
                - 'auto': 自動檢測（Forest_Video_資料夾或單一資料夾）
                - 'forest': 強制使用Forest格式
                - 'single': 強制使用單一資料夾格式
        """
        print(f"\n開始處理數據...")
        
        # 加載類別名稱
        class_names = self.load_class_names()
        
        # 收集所有數據
        all_data = []
        
        if mode == 'auto':
            # 自動檢測模式（排序確保一致性）
            video_folders = sorted([f for f in self.source_path.iterdir() if f.is_dir() and f.name.startswith('Forest_Video_')])
            
            if video_folders:
                print(f"🔍 檢測到Forest格式資料夾，使用Forest模式")
                mode = 'forest'
            else:
                print(f"🔍 未檢測到Forest格式資料夾，使用單一資料夾模式")
                mode = 'single'
        
        if mode == 'forest':
            # Forest格式處理（排序確保一致性）
            video_folders = sorted([f for f in self.source_path.iterdir() if f.is_dir() and f.name.startswith('Forest_Video_')])
            
            # 應用資料夾數量限制
            if self.folder_count_limit is not None and self.folder_count_limit > 0:
                original_count = len(video_folders)
                if self.folder_count_limit < original_count:
                    video_folders = video_folders[:self.folder_count_limit]
                    print(f"📊 資料夾數量限制: {original_count} -> {len(video_folders)} 個資料夾")
                    print(f"⚠️ 已限制處理前 {len(video_folders)} 個資料夾")
                else:
                    print(f"📊 將處理全部 {len(video_folders)} 個資料夾")
            
            total_processed = 0
            total_skipped = 0
            
            for video_folder in sorted(video_folders):
                video_data = self.process_single_video(video_folder)
                all_data.extend(video_data)
                total_processed += len(video_data)
                
                # 計算跳過的樣本數量
                img_folder = video_folder / 'Img'
                if img_folder.exists():
                    image_files = list(img_folder.glob(f'{self.image_pattern}.png')) + list(img_folder.glob(f'{self.image_pattern}.jpg'))
                    skipped = len(image_files) - len(video_data)
                    total_skipped += skipped
            
            print(f"📊 Forest格式: 收集到 {len(all_data)} 個有效樣本")
            
        elif mode == 'single':
            # 單一資料夾格式處理
            single_data = self.process_single_folder(self.source_path)
            all_data.extend(single_data)
            
            print(f"📊 單一資料夾格式: 收集到 {len(all_data)} 個有效樣本")
        
        if len(all_data) == 0:
            print("❌ 沒有找到有效的數據樣本")
            return
        
        # 分割數據集
        train_data, val_data, test_data = self._split_dataset(all_data)
        
        print(f"📊 數據分割: 訓練{len(train_data)} | 驗證{len(val_data)} | 測試{len(test_data)}")
        
        # 複製文件到對應目錄
        print(f"📁 開始複製文件...")
        self._copy_files(train_data, 'train')
        self._copy_files(val_data, 'val')
        self._copy_files(test_data, 'test')
        
        # 更新配置文件
        self.update_config_file(class_names, mode)
        
        print(f"✅ 數據處理完成! 總計: {len(train_data) + len(val_data) + len(test_data)} 個樣本")
        
        # 驗證實際生成的文件數量
        self._verify_generated_files()
    
    def _split_dataset(self, all_data):
        """將數據集分割為訓練/驗證/測試集"""
        random.shuffle(all_data)
        
        total = len(all_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_data = all_data[:train_end]
        val_data = all_data[train_end:val_end]
        test_data = all_data[val_end:]
        
        return train_data, val_data, test_data
    
    def _copy_files(self, data_list, split_name):
        """複製文件到對應的目錄"""
        images_dir = self.output_path / 'images' / split_name
        labels_dir = self.output_path / 'labels' / split_name
        
        copied_count = 0
        for item in data_list:
            try:
                # 添加視頻文件夾前綴避免文件名衝突
                video_prefix = item['video']
                base_name = item['image'].stem  # Img0_1
                label_name = f"{video_prefix}_{item['label'].name}"
                
                # 檢查是否有深度圖
                if item['depth'] is not None and self.use_depth:
                    # 4通道RGBD模式：合併RGB圖像和深度圖為NPY文件
                    npy_name = f"{video_prefix}_{base_name}.npy"
                    npy_dest = images_dir / npy_name
                    
                    # 合併RGB圖像和深度圖為4通道NPY文件
                    np_file_path = self._create_four_channel_image(
                        rgb_path=item['image'],
                        depth_path=item['depth'],
                        output_path=npy_dest
                    )
                else:
                    # 3通道RGB模式：直接複製圖片
                    image_ext = item['image'].suffix
                    image_name = f"{video_prefix}_{base_name}{image_ext}"
                    image_dest = images_dir / image_name
                    
                    # 讀取並保持原始圖片尺寸
                    rgb_image = cv2.imread(str(item['image']))
                    if rgb_image is not None:
                        # 保持原始圖片尺寸，不進行任何調整
                        cv2.imwrite(str(image_dest), rgb_image)
                    else:
                        # 如果讀取失敗，直接複製
                        shutil.copy2(item['image'], image_dest)
                
                # 複製標籤文件
                label_dest = labels_dir / label_name
                shutil.copy2(item['label'], label_dest)
                
                copied_count += 1
            except Exception as e:
                print(f"  ❌ 複製失敗 {item['image'].name}: {str(e)}")
        
        print(f"  {split_name}: {copied_count}/{len(data_list)} 個樣本")
    
    def _create_four_channel_image(self, rgb_path, depth_path, output_path):
        """創建4通道RGBD NPY文件（RGB + 深度）"""
        # 讀取RGB圖像
        rgb_image = cv2.imread(str(rgb_path))
        if rgb_image is None:
            raise ValueError(f"無法讀取RGB圖像: {rgb_path}")
        
        # 讀取深度圖
        if depth_path.suffix.lower() == '.pfm':
            depth_image = self._read_pfm(str(depth_path))
        else:
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError(f"無法讀取深度圖: {depth_path}")
        
        # 確保深度圖是單通道
        if len(depth_image.shape) == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        
        # 確保RGB圖像和深度圖尺寸一致
        if rgb_image.shape[:2] != depth_image.shape[:2]:
            # 如果尺寸不一致，調整深度圖以匹配RGB圖像
            target_h, target_w = rgb_image.shape[:2]
            # OpenCV 幾何變換/重映射對 float16 不友好，先轉為 float32
            depth_image = depth_image.astype(np.float32)
            depth_image = cv2.resize(depth_image, (target_w, target_h))
        else:
            # 保證後續堆疊時 dtype 兼容 OpenCV/Ultralytics 增強
            depth_image = depth_image.astype(np.float32)
        
        # 創建4通道圖像：RGB + 深度（使用float32，避免OpenCV在增強時對float16不支援）
        rgb_float = rgb_image.astype(np.float32)
        depth_float = depth_image.astype(np.float32)
        four_channel = np.dstack([rgb_float, depth_float])
        
        # 保存為NumPy文件（float32精度）
        np.save(output_path, four_channel)
        
        return output_path
    
    def _read_pfm(self, file_path):
        """讀取PFM格式的深度圖"""
        try:
            with open(file_path, 'rb') as f:
                # 讀取PFM頭部
                header_line = f.readline()
                try:
                    header = header_line.decode('utf-8').rstrip()
                except UnicodeDecodeError:
                    # 如果UTF-8解碼失敗，嘗試直接比較bytes
                    header_bytes = header_line.rstrip()
                    if header_bytes == b'PF':
                        header = 'PF'
                    elif header_bytes == b'Pf':
                        header = 'Pf'
                    else:
                        raise ValueError(f"不是有效的PFM文件: {file_path}, 頭部: {header_bytes}")
                
                # 支持 "Pf" 和 "PF" 頭部
                if header not in ['PF', 'Pf']:
                    raise ValueError(f"不是有效的PFM文件: {file_path}, 頭部: {header}")
                
                color = (header == 'PF')
                
                # 讀取尺寸
                dim_line = f.readline()
                try:
                    dims = dim_line.decode('utf-8').rstrip().split()
                    width, height = int(dims[0]), int(dims[1])
                except (UnicodeDecodeError, ValueError, IndexError):
                    # 如果解碼失敗，嘗試使用正則表達式
                    dim_match = re.match(rb'^(\d+)\s(\d+)\s*$', dim_line)
                    if dim_match:
                        width, height = map(int, dim_match.groups())
                    else:
                        raise ValueError(f"PFM文件頭部格式錯誤: {file_path}, 尺寸行: {dim_line}")
                
                # 讀取比例因子和字節序
                scale_line = f.readline()
                try:
                    scale = float(scale_line.decode('utf-8').rstrip())
                except (UnicodeDecodeError, ValueError):
                    # 如果解碼失敗，嘗試直接轉換
                    scale_str = scale_line.rstrip()
                    if isinstance(scale_str, bytes):
                        try:
                            scale = float(scale_str.decode('utf-8', errors='ignore'))
                        except:
                            scale = float(scale_str)
                    else:
                        scale = float(scale_str)
                
                # 根據scale的符號確定字節序
                if scale < 0:
                    endian = '<'  # 小端
                    scale = -scale
                else:
                    endian = '>'  # 大端
                
                # 讀取數據（使用numpy更高效）
                data = np.fromfile(f, dtype=endian + 'f4')  # float32
                
                # 確定形狀
                expected_size = height * width * (3 if color else 1)
                
                # 檢查數據大小
                if len(data) < expected_size:
                    raise ValueError(f"PFM文件數據不完整: {file_path}, 期望 {expected_size} 個浮點數, 實際 {len(data)} 個")
                elif len(data) > expected_size:
                    # 如果數據過多，只取需要的部分
                    data = data[:expected_size]
                
                # 重塑數據
                if color:
                    depth_array = data.reshape((height, width, 3))
                    # 如果是彩色，通常只取第一個通道或轉換為灰度
                    if depth_array.shape[2] == 3:
                        depth_array = depth_array[:, :, 0]  # 取第一個通道
                else:
                    depth_array = data.reshape((height, width))
                
                # PFM格式中，scale的絕對值表示比例因子
                # 但通常scale已經被處理過了，這裡保持原樣
                # 如果需要應用scale，取消下面的註釋
                # if abs(scale) != 1.0:
                #     depth_array = depth_array * abs(scale)
                
                return depth_array
                
        except Exception as e:
            raise ValueError(f"讀取PFM文件失敗 {file_path}: {e}") from e
    
    def _verify_generated_files(self):
        """驗證實際生成的文件數量"""
        splits = ['train', 'val', 'test']
        total_samples = 0
        total_files = 0
        
        for split in splits:
            images_dir = self.output_path / 'images' / split
            labels_dir = self.output_path / 'labels' / split
            
            if images_dir.exists() and labels_dir.exists():
                # 計算實際文件數量
                label_files = list(labels_dir.glob('*.txt'))
                samples = len(label_files)
                
                # 檢查圖像文件類型
                npy_files = list(images_dir.glob('*.npy'))  # 4通道RGBD NPY文件
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpeg'))  # 標準圖片文件
                
                if npy_files:
                    # 4通道RGBD模式
                    actual_files = len(npy_files)
                    file_type = "RGBD NPY (4通道)"
                else:
                    # 3通道RGB模式
                    actual_files = len(image_files)
                    file_type = "標準RGB圖片"
                
                print(f"  {split.upper()}: {samples} 樣本, {actual_files} 圖像文件 ({file_type})")
                
                total_samples += samples
                total_files += actual_files + len(label_files)
            else:
                print(f"  ❌ {split.upper()} 集目錄不存在")
        
        print(f"📊 總計: {total_samples} 樣本, {total_files} 文件")

if __name__ == '__main__':
    print("=" * 80)
    print("⚠️  此模組僅供GUI調用，不支持命令行直接運行")
    print("⚠️  This module is for GUI use only and does not support direct command-line execution")
    print("=" * 80)
    print()
    print("📌 請使用GUI啟動器運行數據轉換功能:")
    print("📌 Please use the GUI launcher to run data conversion:")
    print()
    print("   python yolo_launcher_gui.py")
    print()
    print("=" * 80)
