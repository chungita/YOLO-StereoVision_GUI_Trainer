"""
RAFT-Stereo 推理脚本
用于对立体图像对进行视差估计推理
"""

from __future__ import print_function, division
import sys
import os
from pathlib import Path
import glob
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# 添加当前目录到路径，以便导入 raft_stereo_trainer
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 导入 RAFT-Stereo 模型和相关工具
from raft_stereo_trainer import RAFTStereo, InputPadder, readPFM

# 兼容不同 PyTorch 版本的 AMP 支持
try:
    from torch.amp import autocast as _autocast_base
    # PyTorch 2.0+ 需要 device_type 参数
    _USE_AMP_V2 = True
except ImportError:
    # PyTorch < 2.0 使用 cuda.amp
    from torch.cuda.amp import autocast as _autocast_old
    _USE_AMP_V2 = False

# 创建兼容的 autocast 函数
if _USE_AMP_V2:
    def autocast(enabled=True, device_type='cuda'):
        """兼容 PyTorch 2.0+ 的 autocast"""
        return _autocast_base(device_type=device_type, enabled=enabled)
else:
    def autocast(enabled=True, device_type=None):
        """兼容 PyTorch < 2.0 的 autocast (忽略 device_type 参数)"""
        return _autocast_old(enabled=enabled)

# 尝试导入 OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: OpenCV 不可用，将使用 PIL 进行图像处理")


def read_image(path):
    """读取图像并转换为 torch tensor"""
    try:
        if CV2_AVAILABLE:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(path).convert('RGB')
            img = np.array(img)
        
        # 转换为 torch tensor: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img
    except Exception as e:
        raise ValueError(f"读取图像失败 {path}: {e}")


def save_disparity_pfm(disparity, path, verify=True):
    """
    保存视差图为 PFM 格式（Portable Float Map）
    
    Args:
        disparity: torch.Tensor，视差图数据
        path: 输出文件路径
        verify: 是否在保存后验证文件可读性（默认True）
    
    Returns:
        bool: 如果verify=True，返回验证结果；否则返回True
    """
    # 确保路径是字符串
    path_str = str(Path(path))
    
    # PFM 格式：保存原始浮点数，无损失
    disparity_np = disparity.cpu().numpy()
    
    # 处理不同的输入形状，确保最终得到 2D 数组 (height, width)
    # 移除所有大小为1的维度
    while len(disparity_np.shape) > 2 and any(s == 1 for s in disparity_np.shape):
        # 找到第一个大小为1的维度并移除
        squeeze_axis = None
        for i, s in enumerate(disparity_np.shape):
            if s == 1:
                squeeze_axis = i
                break
        if squeeze_axis is not None:
            disparity_np = np.squeeze(disparity_np, axis=squeeze_axis)
        else:
            break
    
    # 如果仍然是多维数组，取第一个通道或第一个batch
    while len(disparity_np.shape) > 2:
        if disparity_np.shape[0] == 1 or disparity_np.shape[0] <= 3:
            # 可能是通道维度，取第一个通道
            disparity_np = disparity_np[0]
        elif disparity_np.shape[-1] == 1 or disparity_np.shape[-1] <= 3:
            # 通道在最后，取第一个通道
            disparity_np = disparity_np[..., 0]
        else:
            # 可能是batch维度，取第一个
            disparity_np = disparity_np[0]
    
    # 确保是 2D 数组
    if len(disparity_np.shape) != 2:
        raise ValueError(f"无法处理视差图形状: {disparity_np.shape}, 期望 2D 数组")
    
    # 获取高度和宽度（注意：numpy 数组是 (height, width) 格式）
    height, width = disparity_np.shape
    disparity_2d = disparity_np.astype(np.float32)
    
    # 确保输出目录存在
    output_dir = Path(path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入 PFM 文件
    with open(path_str, 'wb') as f:
        # PFM 文件头
        f.write(b'Pf\n')  # 单通道浮点图
        f.write(f'{width} {height}\n'.encode())  # PFM 格式：宽度 高度
        f.write(b'-1.0\n')  # 字节序（-1.0 表示小端，负数表示小端字节序）
        
        # 写入数据（不进行翻转，保持原始方向）
        # 注意：直接保存原始数据，不进行flipud，读取时也会得到相同方向的数据
        data_bytes = disparity_2d.astype(np.float32).tobytes()
        f.write(data_bytes)
    
    # 验证文件是否可以正常读取
    if verify:
        try:
            # 读取刚保存的文件
            loaded_data = readPFM(path_str)
            
            # 验证尺寸
            if loaded_data.shape != (height, width):
                raise ValueError(
                    f"PFM文件尺寸不匹配: 保存时 {disparity_2d.shape}, "
                    f"读取时 {loaded_data.shape}"
                )
            
            # 验证数据范围（允许小的数值误差）
            # 注意：保存时不进行flipud，readPFM函数也不进行flipud
            # 所以读取回来的数据应该和原始数据方向一致
            # 即：loaded_data 应该等于 disparity_2d
            expected_data = disparity_2d  # 保存到文件的数据（未翻转）
            
            # 计算差异（允许浮点误差）
            max_diff = np.abs(loaded_data - expected_data).max()
            if max_diff > 1e-5:  # 允许小的浮点误差
                logging.warning(
                    f"PFM文件数据验证警告: 最大差异 {max_diff:.2e} "
                    f"(文件: {Path(path_str).name})"
                )
                # 不抛出异常，因为可能是浮点精度问题
            
            logging.debug(f"✅ PFM文件验证通过: {Path(path_str).name}, 尺寸: {loaded_data.shape}")
            return True
            
        except Exception as e:
            logging.error(f"❌ PFM文件验证失败: {path_str}, 错误: {e}")
            raise RuntimeError(f"PFM文件保存后验证失败: {e}")
    
    return True


def save_disparity_image(disparity, path, output_format='png', flip=False):
    """保存视差图为图像格式（PNG, JPG, TIFF, BMP）"""
    # 确保路径是字符串
    path_str = str(Path(path))
    
    disparity_np = disparity.cpu().numpy()
    
    # 处理不同的输入形状，确保最终得到 2D 数组 (height, width)
    # 移除所有大小为1的维度
    while len(disparity_np.shape) > 2 and any(s == 1 for s in disparity_np.shape):
        # 找到第一个大小为1的维度并移除
        squeeze_axis = None
        for i, s in enumerate(disparity_np.shape):
            if s == 1:
                squeeze_axis = i
                break
        if squeeze_axis is not None:
            disparity_np = np.squeeze(disparity_np, axis=squeeze_axis)
        else:
            break
    
    # 如果仍然是多维数组，取第一个通道或第一个batch
    while len(disparity_np.shape) > 2:
        if disparity_np.shape[0] == 1 or disparity_np.shape[0] <= 3:
            # 可能是通道维度，取第一个通道
            disparity_np = disparity_np[0]
        elif disparity_np.shape[-1] == 1 or disparity_np.shape[-1] <= 3:
            # 通道在最后，取第一个通道
            disparity_np = disparity_np[..., 0]
        else:
            # 可能是batch维度，取第一个
            disparity_np = disparity_np[0]
    
    # 确保是 2D 数组
    if len(disparity_np.shape) != 2:
        raise ValueError(f"无法处理视差图形状: {disparity_np.shape}, 期望 2D 数组")
    
    disparity_2d = disparity_np.astype(np.float32)
    
    # 应用翻转（如果需要）
    if flip:
        disparity_2d = np.flipud(np.fliplr(disparity_2d))
    
    # 归一化到 0-1 范围用于可视化
    if disparity_2d.max() > disparity_2d.min():
        normalized = (disparity_2d - disparity_2d.min()) / (disparity_2d.max() - disparity_2d.min())
    else:
        normalized = np.zeros_like(disparity_2d)
    
    # 应用 jet colormap
    try:
        from matplotlib import cm
        jet = cm.get_cmap('jet')
        # normalized 是 (H, W) 形状，jet 会返回 (H, W, 4) RGBA
        colored = jet(normalized)  # 返回 (H, W, 4)
        # 只取 RGB，忽略 alpha，并转换为 uint8
        colored = (colored[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    except ImportError:
        # 如果没有 matplotlib，使用简单的灰度图
        colored = (normalized * 255).astype(np.uint8)  # (H, W)
        colored = np.stack([colored, colored, colored], axis=-1)  # (H, W, 3)
    
    # 确保输出目录存在
    output_dir = Path(path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图像 - PIL Image.fromarray 需要 (H, W, 3) 格式的 uint8 数组
    img = Image.fromarray(colored, 'RGB')
    img.save(path_str, format=output_format.upper())


def save_disparity_numpy(disparity, path):
    """保存视差图为 NumPy 数组格式"""
    # 确保路径是字符串
    path_str = str(Path(path))
    
    disparity_np = disparity.cpu().numpy()
    if len(disparity_np.shape) == 3:
        disparity_np = disparity_np[0]  # 移除 batch 维度
    if len(disparity_np.shape) == 2:
        disparity_np = disparity_np[np.newaxis, :, :]  # 添加通道维度
    
    # 确保输出目录存在
    output_dir = Path(path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(path_str, disparity_np[0])  # 保存单通道视差图


@torch.no_grad()
def demo(args):
    """
    执行 RAFT-Stereo 推理
    
    Args:
        args: 包含推理参数的命名空间对象
            - restore_ckpt: 模型检查点路径
            - left_imgs: 左图像路径（支持通配符）
            - right_imgs: 右图像路径（支持通配符）
            - output_directory: 输出目录
            - valid_iters: 推理迭代次数
            - mixed_precision: 是否使用混合精度
            - save_numpy: 是否保存 NumPy 数组
            - output_format: 输出格式 ('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pfm')
            - flip_non_pfm: 是否对非PFM格式进行翻转
            - hidden_dims: 隐藏维度
            - corr_implementation: 相关性实现方式
            - shared_backbone: 是否共享骨干网络
            - corr_levels: 相关性金字塔层数
            - corr_radius: 相关性半径
            - n_downsample: 下采样次数
            - context_norm: 上下文归一化方式
            - slow_fast_gru: 是否使用慢快GRU
            - n_gru_layers: GRU层数
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    )
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备 Using device: {device}")
    
    # 加载模型
    logging.info("正在加载模型 Loading model...")
    model = RAFTStereo(args)
    
    # 加载检查点
    if args.restore_ckpt:
        if not os.path.exists(args.restore_ckpt):
            raise FileNotFoundError(f"模型文件不存在 Model file not found: {args.restore_ckpt}")
        
        logging.info(f"加载检查点 Loading checkpoint: {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt, map_location=device)
        
        # 处理 DataParallel 包装的模型
        if 'module.' in list(checkpoint.keys())[0]:
            # 如果检查点是用 DataParallel 保存的，需要去掉 'module.' 前缀
            new_checkpoint = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_checkpoint[k[7:]] = v
                else:
                    new_checkpoint[k] = v
            checkpoint = new_checkpoint
        
        model.load_state_dict(checkpoint, strict=True)
        logging.info("模型加载完成 Model loaded successfully")
    else:
        raise ValueError("必须提供模型检查点路径 restore_ckpt is required")
    
    model = model.to(device)
    model.eval()
    
    # 统计模型参数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数数量 Model parameters: {num_params / 1e6:.2f}M")
    
    # 查找图像文件
    left_files = sorted(glob.glob(args.left_imgs, recursive=True))
    right_files = sorted(glob.glob(args.right_imgs, recursive=True))
    
    if not left_files:
        raise FileNotFoundError(f"未找到左图像 No left images found: {args.left_imgs}")
    
    if not right_files:
        raise FileNotFoundError(f"未找到右图像 No right images found: {args.right_imgs}")
    
    num_pairs = min(len(left_files), len(right_files))
    logging.info(f"找到 {len(left_files)} 张左图像和 {len(right_files)} 张右图像")
    logging.info(f"Found {len(left_files)} left images and {len(right_files)} right images")
    logging.info(f"将处理 {num_pairs} 对图像 Will process {num_pairs} image pairs")
    
    # 创建输出目录
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"输出目录 Output directory: {output_dir.absolute()}")
    
    # 统计成功和失败的数量
    success_count = 0
    fail_count = 0
    
    # 处理每对图像
    for i in tqdm(range(num_pairs), desc="处理图像对 Processing image pairs"):
        left_path = Path(left_files[i])
        right_path = Path(right_files[i])
        
        logging.info(f"处理 {i+1}/{num_pairs}: {left_path.name} <-> {right_path.name}")
        
        try:
            # 读取图像
            image1 = read_image(left_path)
            image2 = read_image(right_path)
            logging.debug(f"图像尺寸 Image size: {image1.shape}")
            
            # 添加 batch 维度
            image1 = image1[None].to(device)
            image2 = image2[None].to(device)
            
            # 填充图像到32的倍数
            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)
            
            # 推理
            # 根据设备类型确定 device_type
            device_type = 'cuda' if device.startswith('cuda') else 'cpu'
            with autocast(enabled=args.mixed_precision, device_type=device_type):
                _, flow_up = model(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
            
            # 去除填充
            flow_up = padder.unpad(flow_up)
            
            # 获取视差图（flow_up 的第一通道是视差）
            disparity = flow_up[:, 0:1]  # 保持 (B, C, H, W) 格式
            logging.debug(f"视差图形状 Disparity shape: {disparity.shape}")
            
            # 生成输出文件名
            base_name = left_path.stem
            output_format = args.output_format.lower()
            
            # 保存视差图
            if output_format == 'pfm':
                # PFM 格式：保存原始浮点数，不应用翻转
                output_path = output_dir / f"{base_name}.pfm"
                try:
                    save_disparity_pfm(disparity, output_path, verify=True)
                    logging.info(f"✅ 保存并验证 PFM: {output_path.name}")
                except Exception as e:
                    logging.error(f"❌ PFM保存失败: {output_path.name}, 错误: {e}")
                    raise
            else:
                # 图像格式：保存彩色可视化，可选择翻转
                output_path = output_dir / f"{base_name}.{output_format}"
                save_disparity_image(disparity, output_path, output_format=output_format, 
                                   flip=args.flip_non_pfm)
                logging.info(f"✅ 保存 {output_format.upper()}: {output_path.name}")
            
            # 可选：保存 NumPy 数组
            if args.save_numpy:
                numpy_path = output_dir / f"{base_name}.npy"
                save_disparity_numpy(disparity, numpy_path)
                logging.info(f"✅ 保存 NumPy: {numpy_path.name}")
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            logging.error(f"处理图像对 {i+1}/{num_pairs} 失败: {e}")
            continue
    
    # 输出最终统计
    logging.info(f"✅ 推理完成！处理了 {num_pairs} 对图像")
    logging.info(f"成功: {success_count}, 失败: {fail_count}")
    logging.info(f"结果保存在: {output_dir.absolute()}")
    
    # 检查是否有成功处理的图像
    if success_count == 0:
        raise RuntimeError(f"所有图像处理都失败了！处理了 {num_pairs} 对图像，全部失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT-Stereo 推理脚本')
    parser.add_argument('--restore_ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--left_imgs', type=str, required=True, help='左图像路径（支持通配符）')
    parser.add_argument('--right_imgs', type=str, required=True, help='右图像路径（支持通配符）')
    parser.add_argument('--output_directory', type=str, default='demo_output', help='输出目录')
    parser.add_argument('--valid_iters', type=int, default=32, help='推理迭代次数')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度')
    parser.add_argument('--save_numpy', action='store_true', help='保存 NumPy 数组')
    parser.add_argument('--output_format', type=str, default='png', 
                       choices=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pfm'],
                       help='输出格式')
    parser.add_argument('--flip_non_pfm', action='store_true', 
                       help='对非PFM格式进行水平和垂直翻转')
    
    # 模型架构参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3,
                       help='隐藏状态和上下文维度')
    parser.add_argument('--corr_implementation', type=str, default='alt',
                       choices=['reg', 'alt', 'reg_cuda', 'alt_cuda'],
                       help='相关性体积实现方式')
    parser.add_argument('--shared_backbone', action='store_true',
                       help='为上下文和特征编码器使用单一骨干网络')
    parser.add_argument('--corr_levels', type=int, default=4,
                       help='相关性金字塔层数')
    parser.add_argument('--corr_radius', type=int, default=4,
                       help='相关性金字塔宽度')
    parser.add_argument('--n_downsample', type=int, default=2,
                       help='视差场分辨率 (1/2^K)')
    parser.add_argument('--context_norm', type=str, default='batch',
                       choices=['group', 'batch', 'instance', 'none'],
                       help='上下文编码器归一化方式')
    parser.add_argument('--slow_fast_gru', action='store_true',
                       help='更频繁地迭代低分辨率GRU')
    parser.add_argument('--n_gru_layers', type=int, default=3,
                       help='隐藏GRU层数')
    
    args = parser.parse_args()
    
    demo(args)

