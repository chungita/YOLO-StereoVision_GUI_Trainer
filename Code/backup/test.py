import torch
from ultralytics.models import YOLO
import os
import numpy as np
import cv2

# 設置設備 / Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用設備: {device} / Using device: {device}")

# 載入模型 / Load model
model_path = "runs/train/yolo12n_RGBD_5epochs_20251017(2)/weights/best.pt"
if not os.path.exists(model_path):
    print(f"錯誤: 模型文件不存在 / Error: Model file not found: {model_path}")
    exit(1)

model = YOLO(model_path)
print("模型載入成功 / Model loaded successfully")

# 設置路徑 / Set paths
predict_data_dir = "Predict/Data"
result_dir = "Predict/Result"

# 檢查預測數據目錄 / Check predict data directory
if not os.path.exists(predict_data_dir):
    print(f"錯誤: 預測數據目錄不存在 / Error: Predict data directory not found: {predict_data_dir}")
    exit(1)

# 找到可用的圖片文件 / Find available image files
image_files = []
for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
    files = [f for f in os.listdir(predict_data_dir) if f.lower().endswith(ext)]
    image_files.extend(files)

if not image_files:
    print(f"錯誤: 在 {predict_data_dir} 中未找到圖片文件 / Error: No image files found in {predict_data_dir}")
    exit(1)

print(f"找到 {len(image_files)} 個圖片文件 / Found {len(image_files)} image files")

# 確保結果目錄存在 / Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)

# 處理每張圖片 / Process each image
for i, image_file in enumerate(image_files):
    try:
        print(f"處理圖片 {i+1}/{len(image_files)}: {image_file} / Processing image {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(predict_data_dir, image_file)
        
        # 如果是.npy文件，直接使用numpy數組 / If it's a .npy file, use numpy array directly
        if image_file.lower().endswith('.npy'):
            # 載入numpy數組 (保持float16格式) / Load numpy array (keep float16 format)
            img_array = np.load(image_path)
            print(f"載入numpy數組，形狀: {img_array.shape}, 數據類型: {img_array.dtype} / Loaded numpy array, shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # 保持RGBD數據的4個通道 / Keep all 4 channels for RGBD data
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                print("使用RGBD 4通道數據 / Using RGBD 4-channel data")
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                print("使用RGB 3通道數據 / Using RGB 3-channel data")
            
            # 直接使用numpy數組作為source / Use numpy array directly as source
            image_path = img_array
        
        # 進行推理 / Perform inference
        result = model.predict(
            source=image_path,
            device=device,
            project=os.path.dirname(result_dir),
            name=os.path.basename(result_dir),
            exist_ok=True,  # 如果目錄已存在則使用現有目錄，不創建新的 / Use existing directory if it exists
            save=True,  # 保存結果圖片 / Save result images
            save_txt=True,  # 保存文本結果 / Save text results
            save_conf=True,  # 保存置信度 / Save confidence
            show=False,  # 不顯示圖片 / Don't show images
            verbose=False,  # 減少輸出 / Reduce output
            conf=0.25,  # 置信度閾值 / Confidence threshold
            iou=0.45,  # NMS IoU閾值 / NMS IoU threshold
            max_det=300,  # 最大檢測數量 / Maximum detections
            line_width=3,  # 邊框線寬 / Bounding box line width
            show_labels=True,  # 顯示標籤 / Show labels
            show_conf=True,  # 顯示置信度 / Show confidence
            save_crop=False,  # 不保存裁剪 / Don't save crops
            visualize=True,  # 啟用可視化 / Enable visualization
            augment=False,  # 不使用數據增強 / Don't use data augmentation
            agnostic_nms=False,  # 不使用類別無關NMS / Don't use agnostic NMS
            retina_masks=False,  # 不使用視網膜遮罩 / Don't use retina masks
            show_boxes=True,  # 顯示邊框 / Show boxes
            format='torch'  # 返回torch格式 / Return torch format
        )
        
        print(f"✅ 成功處理 {image_file} / ✅ Successfully processed {image_file}")
        
    except Exception as e:
        print(f"❌ 處理 {image_file} 失敗: {e} / ❌ Failed to process {image_file}: {e}")
        continue

print("推理完成 / Inference completed")
print(f"結果保存在: {result_dir} / Results saved to: {result_dir}")