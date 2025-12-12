import os
import torch
import tkinter as tk
from tkinter import messagebox, simpledialog
import glob
from datetime import datetime
import yaml
import sys
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any

def find_pt_files(search_dir: Optional[str] = None) -> List[str]:
    """Find all .pt files in the specified directory or parent directory"""
    if search_dir is None:
        search_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Search in multiple common locations
    search_paths = [
        search_dir,
        os.path.join(search_dir, "Model_file", "PT_File"),
        os.path.join(search_dir, "runs", "train"),
        os.path.join(search_dir, "weights")
    ]
    
    pt_files = []
    for path in search_paths:
        if os.path.exists(path):
            pt_files.extend(glob.glob(os.path.join(path, "*.pt")))
            pt_files.extend(glob.glob(os.path.join(path, "**", "*.pt"), recursive=True))
    
    # Remove duplicates and sort
    pt_files = sorted(list(set(pt_files)))
    return pt_files

def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get comprehensive model information"""
    info = {
        'file_path': model_path,
        'file_size_mb': 0,
        'model_type': 'Unknown',
        'input_channels': None,
        'num_classes': None,
        'total_parameters': 0,
        'trainable_parameters': 0,
        'precision': {},
        'training_info': {},
        'architecture': None,
        'yaml_config': None,
        'file_type': 'Unknown'
    }
    
    try:
        # File size
        info['file_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
        
        # Determine file type
        file_ext = os.path.splitext(model_path)[1].lower()
        info['file_type'] = file_ext
        
        # Load model
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            if 'model' in model_data:
                # Complete model with metadata (typical .pt format)
                model = model_data['model']
                info['model_type'] = type(model).__name__
                
                # Get YAML configuration
                if hasattr(model, 'yaml'):
                    info['yaml_config'] = model.yaml
                    if isinstance(model.yaml, str):
                        try:
                            yaml_dict = yaml.safe_load(model.yaml)
                            info['input_channels'] = yaml_dict.get('ch')
                            info['num_classes'] = yaml_dict.get('nc')
                        except:
                            pass
                
                # Parameter statistics
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    info['total_parameters'] = total_params
                    info['trainable_parameters'] = trainable_params
                    
                    # Precision analysis
                    param_dtypes = {}
                    for p in model.parameters():
                        dtype = str(p.dtype)
                        param_dtypes[dtype] = param_dtypes.get(dtype, 0) + p.numel()
                    info['precision'] = param_dtypes
                
                # Training information
                training_keys = ['epoch', 'best_fitness', 'training_time', 'date', 'version']
                for key in training_keys:
                    if key in model_data:
                        info['training_info'][key] = model_data[key]
                
                # Architecture
                if hasattr(model, 'model'):
                    info['architecture'] = str(model.model)
                elif hasattr(model, '__str__'):
                    info['architecture'] = str(model)
            
            elif all(isinstance(v, torch.Tensor) for v in model_data.values()):
                # State dictionary format (typical .pth format)
                info['model_type'] = 'StateDict'
                info['total_parameters'] = sum(tensor.numel() for tensor in model_data.values())
                
                # Analyze parameter types and shapes
                param_info = {}
                for key, tensor in model_data.items():
                    param_info[key] = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'numel': tensor.numel()
                    }
                info['state_dict_info'] = param_info
                
                # Precision analysis
                param_dtypes = {}
                for tensor in model_data.values():
                    dtype = str(tensor.dtype)
                    param_dtypes[dtype] = param_dtypes.get(dtype, 0) + tensor.numel()
                info['precision'] = param_dtypes
                
                # Try to extract model architecture info from layer names
                layer_types = {}
                for key in model_data.keys():
                    if '.' in key:
                        layer_name = key.split('.')[0]
                        layer_types[layer_name] = layer_types.get(layer_name, 0) + 1
                info['layer_info'] = layer_types
                
            else:
                # Mixed content dictionary
                info['model_type'] = 'MixedDict'
                info['dict_keys'] = list(model_data.keys())
                
                # Check for common training metadata
                training_keys = ['epoch', 'best_fitness', 'training_time', 'date', 'version', 'optimizer', 'scheduler']
                for key in training_keys:
                    if key in model_data:
                        info['training_info'][key] = model_data[key]
        
        else:
            # Direct model object
            info['model_type'] = type(model_data).__name__
            if hasattr(model_data, 'parameters'):
                total_params = sum(p.numel() for p in model_data.parameters())
                trainable_params = sum(p.numel() for p in model_data.parameters() if p.requires_grad)
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
                
                # Precision analysis
                param_dtypes = {}
                for p in model_data.parameters():
                    dtype = str(p.dtype)
                    param_dtypes[dtype] = param_dtypes.get(dtype, 0) + p.numel()
                info['precision'] = param_dtypes
            
            info['architecture'] = str(model_data)
        
    except Exception as e:
        info['error'] = str(e)
    
    return info

def display_yaml_content(model, output_file):
    """Display and save YAML content if available"""
    yaml_content = None
    
    # Check if model has yaml attribute
    if hasattr(model, 'yaml'):
        try:
            yaml_content = model.yaml
            print(f"\n=== YAML 配置 (YAML Configuration) ===")
            print(f"YAML 類型: {type(yaml_content)}")
            
            if isinstance(yaml_content, str):
                print("YAML 內容 (YAML Content):")
                print(yaml_content)
                
                # Parse YAML for better display
                try:
                    yaml_dict = yaml.safe_load(yaml_content)
                    print(f"\n=== 解析後的 YAML 配置 (Parsed YAML Configuration) ===")
                    
                    # Display key information
                    if 'ch' in yaml_dict:
                        print(f"輸入通道數 (Input Channels): {yaml_dict['ch']}")
                    
                    if 'nc' in yaml_dict:
                        print(f"類別數量 (Number of Classes): {yaml_dict['nc']}")
                    
                    if 'backbone' in yaml_dict:
                        print(f"Backbone 層數: {len(yaml_dict['backbone'])}")
                        print("前3層 Backbone 配置:")
                        for i, layer in enumerate(yaml_dict['backbone'][:3]):
                            print(f"  層 {i+1}: {layer}")
                    
                    if 'head' in yaml_dict:
                        print(f"Head 層數: {len(yaml_dict['head'])}")
                    
                    # Save YAML to file
                    with open(output_file, 'a', encoding='utf-8-sig') as f:
                        f.write(f"\n=== YAML 配置 (YAML Configuration) ===\n")
                        f.write(f"YAML 類型: {type(yaml_content)}\n")
                        f.write("YAML 內容 (YAML Content):\n")
                        f.write(yaml_content + "\n")
                        f.write(f"\n=== 解析後的 YAML 配置 (Parsed YAML Configuration) ===\n")
                        f.write(f"輸入通道數 (Input Channels): {yaml_dict.get('ch', 'N/A')}\n")
                        f.write(f"類別數量 (Number of Classes): {yaml_dict.get('nc', 'N/A')}\n")
                        f.write(f"Backbone 層數: {len(yaml_dict.get('backbone', []))}\n")
                        f.write(f"Head 層數: {len(yaml_dict.get('head', []))}\n")
                        
                except Exception as e:
                    print(f"解析 YAML 時發生錯誤: {e}")
                    
            elif isinstance(yaml_content, dict):
                print("YAML 字典內容:")
                print(yaml_content)
                
                # Save YAML to file
                with open(output_file, 'a', encoding='utf-8-sig') as f:
                    f.write(f"\n=== YAML 配置 (YAML Configuration) ===\n")
                    f.write(f"YAML 類型: {type(yaml_content)}\n")
                    f.write("YAML 字典內容:\n")
                    f.write(str(yaml_content) + "\n")
            else:
                print(f"未知的 YAML 格式: {type(yaml_content)}")
                
        except Exception as e:
            print(f"讀取 YAML 時發生錯誤: {e}")
    else:
        print(f"\n=== YAML 配置 (YAML Configuration) ===")
        print("模型沒有 YAML 屬性 (Model has no YAML attribute)")
        
        # Save to file
        with open(output_file, 'a', encoding='utf-8-sig') as f:
            f.write(f"\n=== YAML 配置 (YAML Configuration) ===\n")
            f.write("模型沒有 YAML 屬性 (Model has no YAML attribute)\n")

def display_model_architecture(pt_file_path: str, output_file: Optional[str] = None) -> bool:
    """Load and display the architecture of a PyTorch model"""
    try:
        print(f"\n正在讀取模型檔案: {os.path.basename(pt_file_path)}")
        print("Loading model file...")
        
        # Get comprehensive model information
        model_info = get_model_info(pt_file_path)
        
        if 'error' in model_info:
            print(f"❌ 讀取模型時發生錯誤: {model_info['error']}")
            return False
        
        # Load the model with weights_only=False for YOLO models
        try:
            model_data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
        except Exception as e:
            if "weights_only" in str(e):
                print("嘗試使用 weights_only=False 載入模型 (Trying to load model with weights_only=False)...")
                model_data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
            else:
                raise e
        
        print(f"\n檔案路徑: {pt_file_path}")
        print(f"檔案大小: {model_info['file_size_mb']:.2f} MB")
        print(f"模型類型: {model_info['model_type']}")
        
        # Display key information
        if model_info['input_channels']:
            print(f"輸入通道數: {model_info['input_channels']}")
        if model_info['num_classes']:
            print(f"類別數量: {model_info['num_classes']}")
        if model_info['total_parameters'] > 0:
            print(f"總參數數量: {model_info['total_parameters']:,}")
            print(f"可訓練參數: {model_info['trainable_parameters']:,}")
        
        # Display precision information
        if model_info['precision']:
            print("\n參數精度分析:")
            total_params = model_info['total_parameters']
            for dtype, count in model_info['precision'].items():
                percentage = (count / total_params) * 100 if total_params > 0 else 0
                print(f"  {dtype}: {count:,} 參數 ({percentage:.2f}%)")
        
        # Display training information
        if model_info['training_info']:
            print("\n訓練資訊:")
            for key, value in model_info['training_info'].items():
                print(f"  {key}: {value}")
        
        # Create output file for detailed architecture
        if output_file is None:
            output_file = f"model_architecture_{os.path.splitext(os.path.basename(pt_file_path))[0]}.txt"
        
        # Check if it's a state dict or a complete model
        if isinstance(model_data, dict):
            print("\n模型資料結構 (Model Data Structure):")
            for key, value in model_data.items():
                if hasattr(value, 'shape'):
                    info = f"  {key}: {type(value).__name__} - Shape: {value.shape}"
                    print(info)
                else:
                    info = f"  {key}: {type(value).__name__}"
                    print(info)
            
            # Try to get model architecture if available
            if 'model' in model_data:
                model = model_data['model']
                print(f"\n模型類型: {type(model).__name__}")
                
                # Display YAML content if available
                display_yaml_content(model, output_file)
                
                # Display model architecture
                print("模型架構 (Model Architecture):")
                if hasattr(model, 'model'):
                    arch_str = str(model.model)
                    print(arch_str)
                elif hasattr(model, '__str__'):
                    arch_str = str(model)
                    print(arch_str)
                else:
                    arch_str = f"模型物件: {model}"
                    print(arch_str)
                
                # Display model parameters count and precision info
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"\n參數統計 (Parameter Statistics):")
                    print(f"  總參數數量 (Total parameters): {total_params:,}")
                    print(f"  可訓練參數 (Trainable parameters): {trainable_params:,}")
                    
                    # Check parameter precision
                    param_dtypes = {}
                    for p in model.parameters():
                        dtype = str(p.dtype)
                        param_dtypes[dtype] = param_dtypes.get(dtype, 0) + p.numel()
                    
                    print(f"\n參數精度分析 (Parameter Precision Analysis):")
                    for dtype, count in param_dtypes.items():
                        percentage = (count / total_params) * 100
                        print(f"  {dtype}: {count:,} 參數 ({percentage:.2f}%)")
            
            # Display training information if available
            if 'epoch' in model_data:
                print(f"\n訓練資訊 (Training Info):")
                print(f"  Epoch: {model_data.get('epoch', 'N/A')}")
                print(f"  Best mAP50: {model_data.get('best_fitness', 'N/A')}")
                print(f"  Training time: {model_data.get('training_time', 'N/A')}")
                
        else:
            # It's a complete model
            print(f"\n模型類型: {type(model_data).__name__}")
            print("模型架構 (Model Architecture):")
            print(model_data)
        
        # Write detailed architecture to file
        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write(f"=== 模型架構詳細資訊 (Model Architecture Details) ===\n")
            f.write(f"檔案路徑: {pt_file_path}\n")
            f.write(f"檔案大小: {model_info['file_size_mb']:.2f} MB\n")
            f.write(f"模型類型: {model_info['model_type']}\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write comprehensive model information
            f.write("=== 模型基本信息 ===\n")
            if model_info['input_channels']:
                f.write(f"輸入通道數: {model_info['input_channels']}\n")
            if model_info['num_classes']:
                f.write(f"類別數量: {model_info['num_classes']}\n")
            if model_info['total_parameters'] > 0:
                f.write(f"總參數數量: {model_info['total_parameters']:,}\n")
                f.write(f"可訓練參數: {model_info['trainable_parameters']:,}\n")
            
            # Write precision information
            if model_info['precision']:
                f.write("\n=== 參數精度分析 ===\n")
                total_params = model_info['total_parameters']
                for dtype, count in model_info['precision'].items():
                    percentage = (count / total_params) * 100 if total_params > 0 else 0
                    f.write(f"{dtype}: {count:,} 參數 ({percentage:.2f}%)\n")
            
            # Write training information
            if model_info['training_info']:
                f.write("\n=== 訓練資訊 ===\n")
                for key, value in model_info['training_info'].items():
                    f.write(f"{key}: {value}\n")
            
            # Write YAML configuration
            if model_info['yaml_config']:
                f.write("\n=== YAML 配置 ===\n")
                f.write(str(model_info['yaml_config']) + "\n")
            
            if isinstance(model_data, dict):
                f.write("\n模型資料結構 (Model Data Structure):\n")
                for key, value in model_data.items():
                    if hasattr(value, 'shape'):
                        f.write(f"  {key}: {type(value).__name__} - Shape: {value.shape}\n")
                    else:
                        f.write(f"  {key}: {type(value).__name__}\n")
                
                if 'model' in model_data:
                    model = model_data['model']
                    f.write(f"\n模型類型: {type(model).__name__}\n")
                    
                    # Add YAML content to file
                    if hasattr(model, 'yaml'):
                        try:
                            yaml_content = model.yaml
                            f.write(f"\n=== YAML 配置 (YAML Configuration) ===\n")
                            f.write(f"YAML 類型: {type(yaml_content)}\n")
                            f.write("YAML 內容 (YAML Content):\n")
                            f.write(str(yaml_content) + "\n")
                            
                            # Parse and add structured YAML info
                            if isinstance(yaml_content, str):
                                try:
                                    yaml_dict = yaml.safe_load(yaml_content)
                                    f.write(f"\n=== 解析後的 YAML 配置 (Parsed YAML Configuration) ===\n")
                                    f.write(f"輸入通道數 (Input Channels): {yaml_dict.get('ch', 'N/A')}\n")
                                    f.write(f"類別數量 (Number of Classes): {yaml_dict.get('nc', 'N/A')}\n")
                                    f.write(f"Backbone 層數: {len(yaml_dict.get('backbone', []))}\n")
                                    f.write(f"Head 層數: {len(yaml_dict.get('head', []))}\n")
                                except:
                                    pass
                        except:
                            pass
                    
                    f.write("模型架構 (Model Architecture):\n")
                    if hasattr(model, 'model'):
                        f.write(str(model.model) + "\n")
                    elif hasattr(model, '__str__'):
                        f.write(str(model) + "\n")
                    else:
                        f.write(f"模型物件: {model}\n")
                    
                    if hasattr(model, 'parameters'):
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        f.write(f"\n參數統計 (Parameter Statistics):\n")
                        f.write(f"  總參數數量 (Total parameters): {total_params:,}\n")
                        f.write(f"  可訓練參數 (Trainable parameters): {trainable_params:,}\n")
                        
                        # Check parameter precision
                        param_dtypes = {}
                        for p in model.parameters():
                            dtype = str(p.dtype)
                            param_dtypes[dtype] = param_dtypes.get(dtype, 0) + p.numel()
                        
                        f.write(f"\n參數精度分析 (Parameter Precision Analysis):\n")
                        for dtype, count in param_dtypes.items():
                            percentage = (count / total_params) * 100
                            f.write(f"  {dtype}: {count:,} 參數 ({percentage:.2f}%)\n")
                
                if 'epoch' in model_data:
                    f.write(f"\n訓練資訊 (Training Info):\n")
                    f.write(f"  Epoch: {model_data.get('epoch', 'N/A')}\n")
                    f.write(f"  Best mAP50: {model_data.get('best_fitness', 'N/A')}\n")
                    f.write(f"  Training time: {model_data.get('training_time', 'N/A')}\n")
            else:
                f.write(f"\n模型類型: {type(model_data).__name__}\n")
                f.write("模型架構 (Model Architecture):\n")
                f.write(str(model_data) + "\n")
        
        print(f"\n詳細架構已儲存至: {output_file}")
        print("Complete architecture saved to:", output_file)
        
        # Also save as JSON for programmatic access
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
        print(f"模型信息JSON已儲存至: {json_file}")
            
    except Exception as e:
        print(f"讀取模型時發生錯誤 (Error reading model): {str(e)}")
        return False
    
    return True

def select_pt_file(pt_files):
    """Show a selection dialog for multiple PT files"""
    if len(pt_files) == 1:
        return pt_files[0]
    
    # Create a simple selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Create selection dialog
    choice = simpledialog.askstring(
        "選擇模型檔案 (Select Model File)",
        f"找到 {len(pt_files)} 個 .pt 檔案 (Found {len(pt_files)} .pt files):\n\n" +
        "\n".join([f"{i+1}. {os.path.basename(f)}" for i, f in enumerate(pt_files)]) +
        f"\n\n請輸入檔案編號 (1-{len(pt_files)}) (Please enter file number):",
        initialvalue="1"
    )
    
    root.destroy()
    
    if choice and choice.isdigit():
        file_index = int(choice) - 1
        if 0 <= file_index < len(pt_files):
            return pt_files[file_index]
    
    return None

def analyze_model_batch(model_paths: List[str]) -> Dict[str, Any]:
    """Analyze multiple models and return comprehensive results"""
    results = {
        'total_models': len(model_paths),
        'successful_analyses': 0,
        'failed_analyses': 0,
        'models': {},
        'summary': {}
    }
    
    for model_path in model_paths:
        try:
            model_info = get_model_info(model_path)
            results['models'][model_path] = model_info
            if 'error' not in model_info:
                results['successful_analyses'] += 1
            else:
                results['failed_analyses'] += 1
        except Exception as e:
            results['models'][model_path] = {'error': str(e)}
            results['failed_analyses'] += 1
    
    # Generate summary statistics
    if results['successful_analyses'] > 0:
        param_counts = [info.get('total_parameters', 0) for info in results['models'].values() 
                       if 'error' not in info and info.get('total_parameters', 0) > 0]
        if param_counts:
            results['summary']['avg_parameters'] = sum(param_counts) / len(param_counts)
            results['summary']['min_parameters'] = min(param_counts)
            results['summary']['max_parameters'] = max(param_counts)
    
    return results

def main():
    """Main function to read and display PT file architecture"""
    print("=== 模型讀取器 (Model Reader) ===")
    print("Searching for .pt files in parent directory...")
    
    # Find PT files
    pt_files = find_pt_files()
    
    if not pt_files:
        print("在父目錄中未找到 .pt 檔案 (No .pt files found in parent directory)")
        return
    
    print(f"找到 {len(pt_files)} 個 .pt 檔案 (Found {len(pt_files)} .pt files):")
    for i, file_path in enumerate(pt_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Select file if multiple found
    selected_file = select_pt_file(pt_files)
    
    if selected_file:
        print(f"\n已選擇檔案 (Selected file): {os.path.basename(selected_file)}")
        display_model_architecture(selected_file)
    else:
        print("未選擇檔案 (No file selected)")

def get_model_summary(model_path: str) -> str:
    """Get a brief summary of model information"""
    try:
        model_info = get_model_info(model_path)
        if 'error' in model_info:
            return f"❌ 錯誤: {model_info['error']}"
        
        summary_parts = [f"📁 {os.path.basename(model_path)}"]
        summary_parts.append(f"📊 {model_info['file_size_mb']:.1f} MB")
        
        if model_info['input_channels']:
            summary_parts.append(f"🔢 {model_info['input_channels']} 通道")
        if model_info['num_classes']:
            summary_parts.append(f"🏷️ {model_info['num_classes']} 類別")
        if model_info['total_parameters'] > 0:
            summary_parts.append(f"⚙️ {model_info['total_parameters']:,} 參數")
        
        return " | ".join(summary_parts)
    except Exception as e:
        return f"❌ 錯誤: {str(e)}"

if __name__ == "__main__":
    main()
