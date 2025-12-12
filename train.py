#!/usr/bin/env python3
"""
Training script for RAFT-Stereo
RAFT-Stereo 訓練腳本
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_training():
    """Run the integrated_train.py script with specified parameters"""
    
    # 檢查整合訓練腳本是否存在
    train_script = "integrated_train.py"
    if not os.path.exists(train_script):
        print(f"錯誤: 找不到整合訓練腳本 {train_script}")
        print(f"Error: Integrated training script {train_script} not found")
        return False
    
    # 創建帶時間戳的輸出資料夾
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_folder = f"dataset_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"創建輸出資料夾: {output_folder}")
    print(f"Created output folder: {output_folder}")
    
    # 構建訓練命令 - 使用原始圖片大小進行訓練
    cmd = [
        sys.executable,  # 使用當前 Python 解釋器
        train_script,
        "--name", f"raft-stereo-test-{timestamp}",
        "--train_datasets", "drone",
        "--dataset_root", "Dataset/dataset_Stereo_20251028",
        "--batch_size", "1",  # 極小的批次大小，因為使用原始圖片大小且GPU記憶體不足
        "--train_iters", "4",  # 大幅減少訓練迭代次數
        "--valid_iters", "8",  # 減少驗證迭代次數
        "--spatial_scale", "-0.2", "0.4",
        "--saturation_range", "0.7", "1.3",
        "--n_downsample", "2",
        "--num_steps", "480",  # 大幅減少訓練步數 (約10分鐘)
        "--mixed_precision",
        "--output_dir", output_folder  # 將所有輸出保存到 dataset 資料夾中
    ]
    
    print("準備開始訓練...")
    print("Prepare to start training...")
    print(f"執行命令: {' '.join(cmd)}")
    print(f"Executing command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 執行訓練命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 50)
        print("訓練完成！")
        print("Training completed!")
        
        # 訓練完成後重新命名資料夾
        new_folder_name = f"raft_stereo_{timestamp}"
        if os.path.exists(output_folder):
            if os.path.exists(new_folder_name):
                print(f"警告: 目標資料夾 {new_folder_name} 已存在，將刪除舊資料夾")
                print(f"Warning: Target folder {new_folder_name} already exists, removing old folder")
                import shutil
                shutil.rmtree(new_folder_name)
            
            os.rename(output_folder, new_folder_name)
            print(f"資料夾已重新命名: {output_folder} -> {new_folder_name}")
            print(f"Folder renamed: {output_folder} -> {new_folder_name}")
            output_folder = new_folder_name
        
        return True, output_folder
        
    except subprocess.CalledProcessError as e:
        print(f"訓練失敗: {e}")
        print(f"Training failed: {e}")
        return False, output_folder
    except FileNotFoundError:
        print("找不到 integrated_train.py 文件")
        print("integrated_train.py file not found")
        return False, output_folder
    except KeyboardInterrupt:
        print("\n訓練被用戶中斷")
        print("Training interrupted by user")
        return False, output_folder


if __name__ == "__main__":
    print("RAFT-Stereo 訓練腳本")
    print("RAFT-Stereo Training Script")
    print("=" * 50)
    
    print("\n開始訓練...")
    print("Starting training...")
    print("=" * 50)
    
    success, output_folder = run_training()
    
    if success:
        print("\n[SUCCESS] 訓練成功完成！")
        print("[SUCCESS] Training completed successfully!")
        print("模型已保存到 checkpoints/ 目錄")
        print("Model saved to checkpoints/ directory")
        print(f"輸出資料夾: {output_folder}")
        print(f"Output folder: {output_folder}")
    else:
        print("\n[ERROR] 訓練失敗")
        print("[ERROR] Training failed")
        sys.exit(1)
