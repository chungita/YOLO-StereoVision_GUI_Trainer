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
from raft_stereo_trainer import RAFTStereo, InputPadder

# 兼容不同 PyTorch 版本的 AMP 支持
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# 尝试导入 OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    try:
        print("Warning: OpenCV not available, will use PIL for image processing")
    except:
        pass  # 如果打印也失败，就跳过


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


def save_disparity_pfm(disparity, path):
    """保存视差图为 PFM 格式（Portable Float Map）"""
    # 确保路径是字符串
    path_str = str(Path(path))
    
    # PFM 格式：保存原始浮点数，无损失
    disparity_np = disparity.cpu().numpy()
    if len(disparity_np.shape) == 3:
        disparity_np = disparity_np[0]  # 移除 batch 维度
    if len(disparity_np.shape) == 2:
        disparity_np = disparity_np[np.newaxis, :, :]  # 添加通道维度
    
    # PFM 格式要求：高度、宽度、通道数（1）
    h, w = disparity_np.shape[1], disparity_np.shape[2]
    disparity_2d = disparity_np[0]  # 获取单通道视差图
    
    # 确保输出目录存在
    output_dir = Path(path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入 PFM 文件
    with open(path_str, 'wb') as f:
        # PFM 文件头
        f.write(b'Pf\n')  # 单通道浮点图
        f.write(f'{w} {h}\n'.encode())
        f.write(b'-1.0\n')  # 字节序（-1.0 表示小端）
        # 写入数据（按行倒序，PFM 格式要求）
        disparity_2d_flipped = np.flipud(disparity_2d)
        f.write(disparity_2d_flipped.astype(np.float32).tobytes())
    
    # 验证文件是否成功创建
    if not Path(path_str).exists():
        raise IOError(f"PFM 文件保存失败 File save failed: {path_str}")


def save_disparity_image(disparity, path, output_format='png', flip=False):
    """保存视差图为图像格式（PNG, JPG, TIFF, BMP）"""
    # 确保路径是字符串
    path_str = str(Path(path))
    
    disparity_np = disparity.cpu().numpy()
    
    # 处理不同的输入形状
    # 可能的形状: (B, C, H, W), (C, H, W), (H, W)
    while len(disparity_np.shape) > 2:
        if disparity_np.shape[0] == 1:
            disparity_np = disparity_np[0]  # 移除 batch 或通道维度
        else:
            break
    
    # 确保是 2D 数组 (H, W)
    if len(disparity_np.shape) == 2:
        disparity_2d = disparity_np
    elif len(disparity_np.shape) == 1:
        # 如果是一维，尝试重塑
        raise ValueError(f"Cannot handle 1D disparity array with shape: {disparity_np.shape}")
    else:
        # 取第一个通道
        disparity_2d = disparity_np[0] if disparity_np.shape[0] == 1 else disparity_np
    
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
        # normalized 应该是 (H, W) 形状
        colored = jet(normalized)  # 返回 (H, W, 4) RGBA
        colored = (colored[:, :, :3] * 255).astype(np.uint8)  # 只取 RGB，转换为 uint8
    except ImportError:
        # 如果没有 matplotlib，使用简单的灰度图
        colored = (normalized * 255).astype(np.uint8)
        colored = np.stack([colored, colored, colored], axis=-1)  # (H, W, 3)
    
    # 确保 colored 是 (H, W, 3) 格式
    if len(colored.shape) != 3 or colored.shape[2] != 3:
        raise ValueError(f"Invalid color array shape: {colored.shape}, expected (H, W, 3)")
    
    # 确保输出目录存在
    output_dir = Path(path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图像
    img = Image.fromarray(colored, 'RGB')
    img.save(path_str, format=output_format.upper())
    
    # 验证文件是否成功创建
    if not Path(path_str).exists():
        raise IOError(f"图像文件保存失败 Image save failed: {path_str}")


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
    
    # 验证文件是否成功创建
    if not Path(path_str).exists():
        raise IOError(f"NumPy 文件保存失败 NumPy save failed: {path_str}")


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
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # 验证目录是否真的存在
        if not output_dir.exists():
            raise IOError(f"无法创建输出目录 Cannot create output directory: {output_dir}")
        logging.info(f"输出目录 Output directory: {output_dir.absolute()}")
    except Exception as e:
        logging.error(f"创建输出目录失败 Failed to create output directory: {e}")
        raise
    
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
            # 兼容不同版本的 autocast
            try:
                # PyTorch 2.0+ 需要 device_type 参数
                with autocast(device_type=device.split(':')[0], enabled=args.mixed_precision):
                    _, flow_up = model(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
            except TypeError:
                # 旧版本 PyTorch
                with autocast(enabled=args.mixed_precision):
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
                save_disparity_pfm(disparity, output_path)
                logging.info(f"✅ 保存 PFM: {output_path.absolute()}")
            else:
                # 图像格式：保存彩色可视化，可选择翻转
                output_path = output_dir / f"{base_name}.{output_format}"
                save_disparity_image(disparity, output_path, output_format=output_format, 
                                   flip=args.flip_non_pfm)
                logging.info(f"✅ 保存 {output_format.upper()}: {output_path.absolute()}")
            
            # 可选：保存 NumPy 数组
            if args.save_numpy:
                numpy_path = output_dir / f"{base_name}.npy"
                save_disparity_numpy(disparity, numpy_path)
                logging.info(f"✅ 保存 NumPy: {numpy_path.absolute()}")
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            error_msg = f"❌ 处理图像对失败 Failed to process image pair {i+1}: {e}"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            # 继续处理下一张图片，但记录错误
            continue
    
    # 输出最终统计
    logging.info(f"✅ 推理完成！处理了 {num_pairs} 对图像")
    logging.info(f"✅ Inference completed! Processed {num_pairs} image pairs")
    logging.info(f"成功 Success: {success_count}, 失败 Failed: {fail_count}")
    logging.info(f"结果保存在 Results saved to: {output_dir.absolute()}")
    
    # 列出输出目录中的所有文件
    try:
        output_files = list(output_dir.glob("*"))
        if output_files:
            logging.info(f"输出文件列表 Output files ({len(output_files)} files):")
            for f in sorted(output_files):
                logging.info(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")
        else:
            logging.warning(f"⚠️  输出目录为空！Output directory is empty: {output_dir.absolute()}")
    except Exception as e:
        logging.warning(f"无法列出输出文件 Cannot list output files: {e}")


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

