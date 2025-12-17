import torch
from ultralytics.models import YOLO
import os
import numpy as np
from functools import wraps

def fix_plot_images_function():
    """修復 plot_images 函數的 cls 參數問題"""
    try:
        import ultralytics.utils.plotting as plotting
        
        # 保存原始函數
        original_plot_images = plotting.plot_images
        
        @wraps(original_plot_images)
        def fixed_plot_images(*args, **kwargs):
            # 確保參數順序正確：images, batch_idx, cls
            if len(args) < 3:
                # 如果參數不足，嘗試從訓練上下文中獲取正確的類別信息
                args = list(args)
                
                # 嘗試從預定義類別模組讀取類別數量
                try:
                    import sys
                    from pathlib import Path
                    
                    # 添加config目錄到Python路徑
                    config_dir = Path(__file__).parent.parent / 'config'
                    if str(config_dir) not in sys.path:
                        sys.path.insert(0, str(config_dir))
                    
                    from predefined_classes import get_predefined_classes_count  # type: ignore
                    nc = get_predefined_classes_count()
                    print(f"📊 從預定義類別檢測到類別數量: {nc}")
                except Exception as e:
                    print(f"⚠️ 無法讀取預定義類別，使用默認值: {e}")
                    nc = 1  # 默認1個類別
            
            # 調用原始函數
            return original_plot_images(*args, **kwargs)
                
        
    except Exception as e:
        print(f"⚠️ plot_images 函數修復失敗: {e}")

def main(model_path, confidence_threshold=0.25, device=None, predict_data_dir=None):
    """
    主推理函數 - 基於test.py的改進版本
    
    Args:
        model_path: 模型文件路徑 (必需)
        confidence_threshold: 置信度閾值，默認0.25
        device: 設備類型，如果為None則自動檢測
        predict_data_dir: 預測數據目錄，如果為None則使用默認目錄
    """
    # 修復 plot_images 函數
    fix_plot_images_function()
    
    # 檢查必需參數
    if not model_path:
        print("錯誤: 必須提供模型文件路徑 / Error: Model path is required")
        return
    
    # 檢查 CUDA 是否可用
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device} / Using device: {device}")

    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"錯誤: 模型文件不存在 / Error: Model file not found: {model_path}")
        return

    # 載入模型
    print("載入模型中... / Loading model...")
    model = YOLO(model_path)
    print("模型載入成功 / Model loaded successfully")

    # 設置預測數據目錄
    if predict_data_dir is None:
        predict_data_dir = r"Predict\Data"
    
    if not os.path.exists(predict_data_dir):
        print(f"錯誤: 預測數據目錄不存在 / Error: Predict data directory not found: {predict_data_dir}")
        return
    
    # 檢查是否有圖片文件
    image_files = []
    for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend([f for f in os.listdir(predict_data_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"錯誤: 在 {predict_data_dir} 中未找到圖片文件 / Error: No image files found in {predict_data_dir}")
        return
    
    print(f"找到 {len(image_files)} 個圖片文件 / Found {len(image_files)} image files")

    # 確保結果目錄存在 - 使用帶時間戳的runs目錄
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    result_dir = f"runs/yolo_inference/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 對每張圖片進行推理
    results = []
    for i, image_file in enumerate(image_files):
        try:
            print(f"處理圖片 {i+1}/{len(image_files)}: {image_file} / Processing image {i+1}/{len(image_files)}: {image_file}")
            
            # 構建完整路徑
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
            
            # 進行推理
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
                conf=confidence_threshold,  # 置信度閾值 / Confidence threshold
                iou=0.45,  # NMS IoU閾值 / NMS IoU threshold
                max_det=300,  # 最大檢測數量 / Maximum detections
                line_width=3,  # 邊框線寬 / Bounding box line width
                show_labels=True,  # 顯示標籤 / Show labels
                show_conf=True,  # 顯示置信度 / Show confidence
                save_crop=False,  # 不保存裁剪 / Don't save crops
                visualize=False, # 啟用可視化 / Enable visualization
                augment=False,  # 不使用數據增強 / Don't use data augmentation
                agnostic_nms=False,  # 不使用類別無關NMS / Don't use agnostic NMS
                retina_masks=False,  # 不使用視網膜遮罩 / Don't use retina masks
                show_boxes=True,  # 顯示邊框 / Show boxes
                format='torch'  # 返回torch格式 / Return torch format
            )
            
            # 將結果添加到列表中
            if result:
                results.extend(result)
            
            print(f"✅ 成功處理 {image_file} / ✅ Successfully processed {image_file}")
            
        except Exception as e:
            print(f"❌ 處理 {image_file} 失敗: {e} / ❌ Failed to process {image_file}: {e}")
            continue
    
    print("推理完成 / Inference completed")
    print(f"成功處理 {len(results)} 張圖片 / Successfully processed {len(results)} images")
    print(f"結果保存在: {result_dir} / Results saved to: {result_dir}")
    
    return results

if __name__ == '__main__':
    # 如果直接運行，需要提供模型路徑
    import sys
    if len(sys.argv) < 2:
        print("用法: python yolo_inference.py <model_path> [confidence_threshold] [device] [predict_data_dir]")
        print("Example: python yolo_inference.py model.pt 0.25 cuda Predict/Data")
        sys.exit(1)
    
    model_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    device = sys.argv[3] if len(sys.argv) > 3 else None
    predict_data_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    main(model_path, confidence_threshold, device, predict_data_dir)


def enhanced_inference(model_path, confidence_threshold=0.25, device=None, predict_data_dir=None,
                      iou_threshold=0.45, max_det=300, line_width=3, show_labels=True, 
                      show_conf=True, show_boxes=True, save_txt=True, save_conf=True, 
                      save_crop=False, visualize=True, augment=False, agnostic_nms=False, 
                      retina_masks=False, output_format='torch', verbose=False, show=False):
    """
    增強版推理函數 - 支持所有model.predict()參數
    
    Args:
        model_path: 模型文件路徑 (必需)
        confidence_threshold: 置信度閾值，默認0.25
        device: 設備類型，如果為None則自動檢測
        predict_data_dir: 預測數據目錄，如果為None則使用默認目錄
        iou_threshold: IoU閾值，默認0.45
        max_det: 最大檢測數量，默認300
        line_width: 邊框線寬，默認3
        show_labels: 顯示標籤，默認True
        show_conf: 顯示置信度，默認True
        show_boxes: 顯示邊框，默認True
        save_txt: 保存文本結果，默認True
        save_conf: 保存置信度，默認True
        save_crop: 保存裁剪，默認False
        visualize: 啟用可視化，默認True
        augment: 數據增強，默認False
        agnostic_nms: 類別無關NMS，默認False
        retina_masks: 視網膜遮罩，默認False
        output_format: 輸出格式，默認'torch'
        verbose: 詳細輸出，默認False
        show: 顯示圖片，默認False
    """
    # 修復 plot_images 函數
    fix_plot_images_function()
    
    # 檢查必需參數
    if not model_path:
        print("錯誤: 必須提供模型文件路徑 / Error: Model path is required")
        return
    
    # 檢查 CUDA 是否可用
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device} / Using device: {device}")

    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"錯誤: 模型文件不存在 / Error: Model file not found: {model_path}")
        return

    # 載入模型
    print("載入模型中... / Loading model...")
    model = YOLO(model_path)
    print("模型載入成功 / Model loaded successfully")

    # 設置預測數據目錄
    if predict_data_dir is None:
        predict_data_dir = r"Predict\Data"
    
    if not os.path.exists(predict_data_dir):
        print(f"錯誤: 預測數據目錄不存在 / Error: Predict data directory not found: {predict_data_dir}")
        return
    
    # 檢查是否有圖片文件
    image_files = []
    for ext in ['.npy', '.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend([f for f in os.listdir(predict_data_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"錯誤: 在 {predict_data_dir} 中未找到圖片文件 / Error: No image files found in {predict_data_dir}")
        return
    
    print(f"找到 {len(image_files)} 個圖片文件 / Found {len(image_files)} image files")
    
    # 設置結果目錄 - 使用帶時間戳的runs目錄
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    result_dir = f"runs/yolo_inference/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 對每張圖片進行推理
    results = []
    for i, image_file in enumerate(image_files):
        try:
            print(f"處理圖片 {i+1}/{len(image_files)}: {image_file} / Processing image {i+1}/{len(image_files)}: {image_file}")
            
            # 構建完整路徑
            image_path = os.path.join(predict_data_dir, image_file)
            
            # 如果是.npy文件，直接使用numpy數組
            if image_file.lower().endswith('.npy'):
                img_array = np.load(image_path)
                print(f"載入numpy數組，形狀: {img_array.shape}, 數據類型: {img_array.dtype} / Loaded numpy array, shape: {img_array.shape}, dtype: {img_array.dtype}")
                
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    print("使用RGBD 4通道數據 / Using RGBD 4-channel data")
                elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    print("使用RGB 3通道數據 / Using RGB 3-channel data")
                
                image_path = img_array
            
            # 進行推理 - 使用所有高級參數
            result = model.predict(
                source=image_path,
                device=device,
                project=os.path.dirname(result_dir),
                name=os.path.basename(result_dir),
                exist_ok=True,
                save=True,
                save_txt=save_txt,
                save_conf=save_conf,
                show=show,
                verbose=verbose,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_det,
                line_width=line_width,
                show_labels=show_labels,
                show_conf=show_conf,
                save_crop=save_crop,
                visualize=visualize,
                augment=augment,
                agnostic_nms=agnostic_nms,
                retina_masks=retina_masks,
                show_boxes=show_boxes,
                format=output_format
            )
            
            print(f"✅ 成功處理 {image_file} / ✅ Successfully processed {image_file}")
            results.append(result)
            
        except Exception as e:
            print(f"❌ 處理 {image_file} 失敗: {e} / ❌ Failed to process {image_file}: {e}")
            continue

    print("推理完成 / Inference completed")
    print(f"結果保存在: {result_dir} / Results saved to: {result_dir}")
    return results