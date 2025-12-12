#!/usr/bin/env python3
"""
快速測試腳本 - 10分鐘內完成訓練測試
Quick Test Script - Complete training test within 10 minutes
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_quick_test():
    """運行快速測試訓練"""
    
    # 檢查整合訓練腳本是否存在
    train_script = "integrated_train.py"
    if not os.path.exists(train_script):
        print(f"錯誤: 找不到整合訓練腳本 {train_script}")
        print(f"Error: Integrated training script {train_script} not found")
        return False
    
    # 創建帶時間戳的輸出資料夾
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_folder = f"quick_test_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"創建快速測試輸出資料夾: {output_folder}")
    print(f"Created quick test output folder: {output_folder}")
    
    # 構建快速測試命令 - 使用原始圖片大小，優化為10分鐘內完成
    cmd = [
        sys.executable,  # 使用當前 Python 解釋器
        train_script,
        "--name", f"raft-stereo-quick-test-{timestamp}",
        "--train_datasets", "drone",
        "--dataset_root", "Dataset/dataset_Stereo_20251028",
        "--batch_size", "2",  # 極小的批次大小，因為使用原始圖片大小且GPU記憶體不足
        "--train_iters", "2",  # 最少的訓練迭代次數
        "--valid_iters", "4",  # 最少的驗證迭代次數
        "--spatial_scale", "-0.1", "0.2",  # 減少空間變換範圍
        "--saturation_range", "0.8", "1.2",  # 減少顏色變換範圍
        "--n_downsample", "2",
        "--num_steps", "20",  # 極少的訓練步數 (約5-10分鐘)
        "--mixed_precision",
        "--output_dir", output_folder
    ]
    
    print("準備開始快速測試訓練...")
    print("Prepare to start quick test training...")
    print(f"執行命令: {' '.join(cmd)}")
    print(f"Executing command: {' '.join(cmd)}")
    print("-" * 50)
    print("預估完成時間: 5-10分鐘")
    print("Estimated completion time: 5-10 minutes")
    print("-" * 50)
    
    try:
        # 執行快速測試命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 50)
        print("快速測試完成！")
        print("Quick test completed!")
        
        # 測試完成後重新命名資料夾
        new_folder_name = f"raft_stereo_quick_test_{timestamp}"
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
        print(f"快速測試失敗: {e}")
        print(f"Quick test failed: {e}")
        return False, output_folder
    except FileNotFoundError:
        print("找不到 integrated_train.py 文件")
        print("integrated_train.py file not found")
        return False, output_folder
    except KeyboardInterrupt:
        print("\n快速測試被用戶中斷")
        print("Quick test interrupted by user")
        return False, output_folder


if __name__ == "__main__":
    print("RAFT-Stereo 快速測試腳本")
    print("RAFT-Stereo Quick Test Script")
    print("=" * 50)
    
    print("\n開始快速測試...")
    print("Starting quick test...")
    print("=" * 50)
    
    success, output_folder = run_quick_test()
    
    if success:
        print("\n[SUCCESS] 快速測試成功完成！")
        print("[SUCCESS] Quick test completed successfully!")
        print("模型已保存到 checkpoints/ 目錄")
        print("Model saved to checkpoints/ directory")
        print(f"輸出資料夾: {output_folder}")
        print(f"Output folder: {output_folder}")
    else:
        print("\n[ERROR] 快速測試失敗")
        print("[ERROR] Quick test failed")
        sys.exit(1)
