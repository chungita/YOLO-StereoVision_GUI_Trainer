"""
模型修改器模組
用於修改 PyTorch 模型的輸入通道數，支持新增通道功能
保持原始精度，不強制轉換為float16
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelModifier:
    """模型修改器 - 調整 PyTorch 模型的輸入通道數，支持新增通道功能"""
    
    def __init__(self):
        """初始化模型修改器"""
        self.supported_formats = ['.pt', '.pth']
        self.weight_methods = {
            'copy_avg': '複製原始權重 + 平均值',
            'copy_zero': '複製原始權重 + 零初始化',
            'copy_random': '複製原始權重 + 隨機初始化',
            'full_random': '完全隨機初始化'
        }
    
    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """
        分析模型結構
        
        Args:
            model_path: 模型文件路徑
            
        Returns:
            模型分析結果字典
        """
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            logger.info("🔍 分析模型結構...")
            
            # 載入模型
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 處理不同的模型格式
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    return {
                        'error': 'state_dict 格式不支持直接分析',
                        'model_type': 'state_dict',
                        'file_name': Path(model_path).name
                    }
            
            # 分析第一層卷積層
            first_conv = None
            conv_layers = []
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_info = {
                        'name': name,
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                        'bias': module.bias is not None
                    }
                    conv_layers.append(conv_info)
                    
                    if first_conv is None:
                        first_conv = conv_info
            
            if first_conv is None:
                return {
                    'error': '未找到卷積層',
                    'model_type': type(model).__name__,
                    'file_name': Path(model_path).name
                }
            
            # 生成建議
            suggestions = self._generate_suggestions(first_conv['in_channels'])
            
            return {
                'success': True,
                'file_name': Path(model_path).name,
                'model_type': type(model).__name__,
                'first_conv': first_conv,
                'all_conv_layers': conv_layers,
                'suggestions': suggestions,
                'total_conv_layers': len(conv_layers)
            }
            
        except Exception as e:
            logger.error(f"❌ 模型分析失敗: {e}")
            return {
                'error': f"模型分析失敗: {e}",
                'file_name': Path(model_path).name if model_path else "未知"
            }
    
    
    def modify_model_channels(
        self, 
        input_path: str, 
        output_path: str = None, 
        original_channels: int = None, 
        target_channels: int = 4, 
        weight_method: str = 'copy_avg'
    ) -> Dict[str, Any]:
        """
        修改模型通道數
        
        Args:
            input_path: 輸入模型路徑
            output_path: 輸出模型路徑 (可選，預設為Model_file/4_channel/目錄)
            original_channels: 原始通道數 (可選，自動檢測)
            target_channels: 目標通道數 (預設為4)
            weight_method: 權重初始化方法
            
        Returns:
            修改結果字典
        """
        try:
            if not Path(input_path).exists():
                raise FileNotFoundError(f"輸入模型文件不存在: {input_path}")
            
            # 載入模型以獲取原始通道數（如果未指定）
            model = torch.load(input_path, map_location='cpu', weights_only=False)
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            
            # 自動檢測原始通道數
            if original_channels is None:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        original_channels = module.in_channels
                        break
                if original_channels is None:
                    raise Exception("未找到卷積層")
            
            if original_channels == target_channels:
                return {
                    'success': True,
                    'message': '原始通道數與目標通道數相同，無需修改',
                    'output_path': output_path
                }
            
            # 如果沒有指定輸出路徑，自動生成
            if output_path is None:
                input_file = Path(input_path)
                # 生成預設輸出路徑：Model_file/4_channel/原本檔名_目標通道數channel.pt
                base_name = input_file.stem  # 原本檔名（不含副檔名）
                output_filename = f"{base_name}_{target_channels}channel.pt"
                output_path = f"Model_file/4_channel/{output_filename}"
                
                # 確保輸出目錄存在
                Path("Model_file/4_channel").mkdir(parents=True, exist_ok=True)
            
            logger.info("🔧 開始修改模型通道數...")
            logger.info(f"   原始通道數: {original_channels}")
            logger.info(f"   目標通道數: {target_channels}")
            logger.info(f"   權重初始化: {self.weight_methods.get(weight_method, weight_method)}")
            logger.info(f"   輸出路徑: {output_path}")
            
            # 載入模型
            model = torch.load(input_path, map_location='cpu', weights_only=False)
            
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    raise Exception("state_dict 格式不支持直接修改")
            
            # 找到第一層卷積層
            first_conv = None
            first_conv_name = None
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    first_conv = module
                    first_conv_name = name
                    break
            
            if first_conv is None:
                raise Exception("未找到卷積層")
            
            if first_conv.in_channels != original_channels:
                raise Exception(f"模型實際通道數 ({first_conv.in_channels}) 與設置的原始通道數 ({original_channels}) 不匹配")
            
            # 創建新的第一層卷積層
            new_first_conv = nn.Conv2d(
                in_channels=target_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # 保持原始精度，不強制轉換為float16
            if first_conv.weight.dtype == torch.float16:
                new_first_conv = new_first_conv.half()
            else:
                new_first_conv = new_first_conv.float()
            
            # 權重初始化
            self._initialize_weights(new_first_conv, first_conv, original_channels, target_channels, weight_method)
            
            # 替換模型中的第一層
            self._replace_first_conv(model, first_conv_name, new_first_conv)
            
            # 更新模型的yaml配置
            self._update_model_yaml(model, target_channels)
            
            # 保存修改後的模型
            torch.save(model, output_path)
            
            # 驗證修改結果
            verification_result = self._verify_modification(output_path, target_channels)
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            
            return {
                'success': True,
                'message': '模型修改成功',
                'output_path': output_path,
                'original_channels': original_channels,
                'target_channels': target_channels,
                'actual_channels': verification_result['actual_channels'],
                'weight_method': self.weight_methods.get(weight_method, weight_method),
                'file_size_mb': round(file_size_mb, 2),
                'yaml_updated': hasattr(model, 'yaml'),
                'verification': verification_result
            }
            
        except Exception as e:
            logger.error(f"❌ 模型修改失敗: {e}")
            return {
                'success': False,
                'error': f"模型修改失敗: {e}"
            }
    
    def _generate_suggestions(self, current_channels: int) -> Dict[str, Any]:
        """生成通道數修改建議"""
        suggestions = {
            'current_channels': current_channels,
            'recommended_target': None,
            'reason': None
        }
        
        if current_channels == 3:
            suggestions['recommended_target'] = 4
            suggestions['reason'] = '檢測到3通道模型，建議修改為4通道以支持RGBD數據'
        elif current_channels == 4:
            suggestions['recommended_target'] = 3
            suggestions['reason'] = '檢測到4通道模型，建議修改為3通道以支持標準RGB數據'
        else:
            suggestions['reason'] = f'當前{current_channels}通道，請手動設置目標通道數'
        
        return suggestions
    
    def _initialize_weights(
        self, 
        new_conv: nn.Conv2d, 
        original_conv: nn.Conv2d, 
        original_channels: int, 
        target_channels: int, 
        weight_method: str
    ):
        """初始化新卷積層的權重"""
        with torch.no_grad():
            # 保持原始精度，不強制轉換
            original_weight = original_conv.weight
            original_bias = original_conv.bias if original_conv.bias is not None else None
            
            if target_channels > original_channels:
                # 增加通道數
                if weight_method.startswith('copy'):
                    # 複製原始權重（保持原始精度）
                    new_conv.weight[:, :original_channels, :, :] = original_weight
                    
                    if weight_method == 'copy_avg':
                        # 新通道使用平均值
                        avg_weight = original_weight.mean(dim=1, keepdim=True)
                        new_conv.weight[:, original_channels:, :, :] = avg_weight
                    elif weight_method == 'copy_zero':
                        # 新通道設為零
                        new_conv.weight[:, original_channels:, :, :] = 0
                    elif weight_method == 'copy_random':
                        # 新通道使用隨機初始化
                        nn.init.xavier_uniform_(new_conv.weight[:, original_channels:, :, :])
                else:
                    # 完全隨機初始化
                    nn.init.xavier_uniform_(new_conv.weight)
            else:
                # 減少通道數
                if weight_method.startswith('copy'):
                    # 只保留前 target_channels 個通道
                    new_conv.weight[:, :, :, :] = original_weight[:, :target_channels, :, :]
                else:
                    # 完全隨機初始化
                    nn.init.xavier_uniform_(new_conv.weight)
            
            # 處理偏置（保持原始精度）
            if original_bias is not None:
                new_conv.bias = original_bias.clone()
    
    def _replace_first_conv(self, model: nn.Module, first_conv_name: str, new_conv: nn.Conv2d):
        """替換模型中的第一層卷積層"""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and name == first_conv_name:
                # 找到父模塊並替換
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent_module = model
                    for attr in parent_name.split('.'):
                        parent_module = getattr(parent_module, attr)
                    setattr(parent_module, name.split('.')[-1], new_conv)
                else:
                    # 如果第一層是根模塊
                    setattr(model, name, new_conv)
                break
    
    def _update_model_yaml(self, model, target_channels: int):
        """更新模型的yaml配置以反映新的通道數"""
        try:
            import yaml
            
            # 檢查模型是否有yaml屬性
            if not hasattr(model, 'yaml'):
                logger.info("模型沒有yaml屬性，跳過yaml更新（這是正常的，因為標準PyTorch模型不包含yaml配置）")
                return
            
            # 檢查yaml屬性是否可訪問
            try:
                yaml_attr = getattr(model, 'yaml', None)
                if yaml_attr is None:
                    logger.info("模型yaml屬性為None，跳過yaml更新")
                    return
            except AttributeError as e:
                logger.info(f"無法訪問模型yaml屬性: {e}，跳過yaml更新")
                return
            
            # 安全地訪問yaml屬性
            try:
                yaml_content = model.yaml
            except AttributeError as e:
                logger.warning(f"無法訪問模型yaml屬性: {e}，跳過yaml更新")
                return
            
            # 如果yaml是字符串，解析為字典
            if isinstance(yaml_content, str):
                try:
                    yaml_dict = yaml.safe_load(yaml_content)
                except Exception as e:
                    logger.warning(f"無法解析模型yaml字符串: {e}")
                    return
            elif isinstance(yaml_content, dict):
                yaml_dict = yaml_content
            else:
                logger.warning(f"未知的yaml格式: {type(yaml_content)}")
                return
            
            # 更新通道數相關配置
            if 'ch' in yaml_dict:
                yaml_dict['ch'] = target_channels
                logger.info(f"更新yaml配置: ch = {target_channels}")
            
            # 更新backbone配置中的第一層通道數
            if 'backbone' in yaml_dict and isinstance(yaml_dict['backbone'], list):
                for layer in yaml_dict['backbone']:
                    if isinstance(layer, list) and len(layer) >= 4:
                        # 檢查是否是第一層卷積層
                        if layer[2] == 'Conv' and len(layer[3]) >= 2:
                            # 更新第一層的輸入通道數
                            layer[3][0] = target_channels
                            logger.info(f"更新backbone第一層通道數: {target_channels}")
                            break
            
            # 將更新後的配置轉換回字符串格式
            updated_yaml_str = yaml.dump(yaml_dict, default_flow_style=False)
            
            # 安全地設置yaml屬性
            try:
                model.yaml = updated_yaml_str
                logger.info("模型yaml配置已更新")
            except AttributeError as e:
                logger.warning(f"無法設置模型yaml屬性: {e}")
            
        except Exception as e:
            logger.error(f"更新模型yaml配置失敗: {e}")
            # 不拋出異常，因為yaml更新失敗不應該阻止模型修改
    
    
    def _verify_modification(self, output_path: str, expected_channels: int) -> Dict[str, Any]:
        """驗證修改結果"""
        try:
            modified_model = torch.load(output_path, map_location='cpu', weights_only=False)
            if isinstance(modified_model, dict) and 'model' in modified_model:
                modified_model = modified_model['model']
            
            for name, module in modified_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    actual_channels = module.in_channels
                    return {
                        'success': True,
                        'actual_channels': actual_channels,
                        'expected_channels': expected_channels,
                        'match': actual_channels == expected_channels
                    }
            
            return {
                'success': False,
                'error': '未找到卷積層進行驗證'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"驗證失敗: {e}"
            }
    
    def get_weight_methods(self) -> Dict[str, str]:
        """獲取可用的權重初始化方法"""
        return self.weight_methods.copy()
    
    def validate_model_file(self, model_path: str) -> bool:
        """驗證模型文件是否有效"""
        try:
            if not Path(model_path).exists():
                return False
            
            # 嘗試載入模型
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 檢查是否包含卷積層
            has_conv = False
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    return False  # state_dict 格式不支持
            
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    has_conv = True
                    break
            
            return has_conv
            
        except Exception:
            return False


def create_model_modifier() -> ModelModifier:
    """創建模型修改器實例"""
    return ModelModifier()


# 便捷函數
def analyze_model_structure(model_path: str) -> Dict[str, Any]:
    """分析模型結構的便捷函數"""
    modifier = ModelModifier()
    return modifier.analyze_model(model_path)




def modify_model_channels(
    input_path: str, 
    output_path: str = None, 
    original_channels: int = None, 
    target_channels: int = 4, 
    weight_method: str = 'copy_avg'
) -> Dict[str, Any]:
    """修改模型通道數的便捷函數"""
    modifier = ModelModifier()
    return modifier.modify_model_channels(
        input_path, output_path, original_channels, target_channels, weight_method
    )


if __name__ == "__main__":
    # 測試代碼
    modifier = ModelModifier()
    
    # 測試分析功能
    test_model = "Model_file/standard/yolov12n.pt"
    if Path(test_model).exists():
        print("=== 測試模型分析 ===")
        result = modifier.analyze_model(test_model)
        print("分析結果:", result)
        
        
        print("\n=== 測試通道數修改（使用預設路徑）===")
        channel_result = modifier.modify_model_channels(test_model, target_channels=4)
        print("通道修改結果:", channel_result)
    
    print("模型修改器模組載入成功")
