#!/usr/bin/env python3

from __future__ import print_function, division

import logging
import numpy as np
import os
import re
import copy
import math
import random
import warnings
import time
import json
import imageio
from pathlib import Path
from glob import glob
import os.path as osp
from datetime import datetime
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ColorJitter, functional, Compose

from PIL import Image

# 嘗試導入 OpenCV，如果失敗則設置為不可用
try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    CV2_AVAILABLE = True
    print("OpenCV 成功載入，版本:", cv2.__version__)
except ImportError as e:
    print(f"警告: 無法載入 OpenCV: {e}")
    print("將使用 PIL 進行圖像處理")
    CV2_AVAILABLE = False
    # 創建一個虛擬的 cv2 模組
    class DummyCV2:
        INTER_LINEAR = 1
        def resize(self, *args, **kwargs):
            raise NotImplementedError("OpenCV 不可用，請安裝 OpenCV")
        def setNumThreads(self, *args, **kwargs):
            pass
        def ocl(self):
            return type('ocl', (), {'setUseOpenCL': lambda *args: None})()
    cv2 = DummyCV2()

from scipy import interpolate
from skimage import color, io

from opt_einsum import contract

# 兼容不同 PyTorch 版本的 AMP 支持
try:
    from torch.amp import GradScaler
    autocast = torch.amp.autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class InputPadder:
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    """读取PFM格式文件（Portable Float Map）
    
    Args:
        file: 文件路径（字符串）或文件对象
        
    Returns:
        numpy数组，形状为 (height, width) 或 (height, width, 3)
    """
    # 如果是字符串路径，打开文件；否则假设是文件对象
    file_path = file
    if isinstance(file, str):
        file = open(file, 'rb')
        should_close = True
    else:
        should_close = False
    
    try:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        # 读取文件头
        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception(f'Not a PFM file. Header: {header}')

        # 读取尺寸
        dim_line = file.readline()
        dim_match = re.match(rb'^(\d+)\s(\d+)\s*$', dim_line)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception(f'Malformed PFM header. Dimension line: {dim_line}')

        # 读取比例因子和字节序
        scale_line = file.readline().rstrip()
        try:
            scale = float(scale_line)
        except (ValueError, TypeError):
            # 如果无法转换为float，尝试解码
            if isinstance(scale_line, bytes):
                scale = float(scale_line.decode('utf-8', errors='ignore'))
            else:
                raise Exception(f'Malformed PFM header. Scale line: {scale_line}')
        
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        # 读取数据
        data = np.fromfile(file, dtype=endian + 'f4')  # 明确指定float32
        shape = (height, width, 3) if color else (height, width)
        
        # 检查数据大小是否匹配
        expected_size = height * width * (3 if color else 1)
        if len(data) < expected_size:
            raise Exception(f'PFM file data incomplete. Expected {expected_size} floats, got {len(data)}')
        elif len(data) > expected_size:
            # 如果数据过多，只取需要的部分
            data = data[:expected_size]

        data = np.reshape(data, shape)
        # 注意：PFM 格式通常從底部存儲，但為了與推論保持一致，
        # 我們保持原始方向，不進行 flipud
        # data = np.flipud(data)
        return data
    finally:
        if should_close:
            file.close()

def read_gen(file_name, pil=False):
    ext = osp.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

class AdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

class FlowAugmentor:
    def __init__(self, crop_size=None, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1]):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        ht, wd = img1.shape[:2]
        
        # 如果沒有指定 crop_size，則不進行縮放限制
        if self.crop_size is not None:
            min_scale = np.maximum(
                (self.crop_size[0] + 1) / float(ht), 
                (self.crop_size[1] + 1) / float(wd))
        else:
            min_scale = 0.1  # 設置一個最小的縮放限制

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            if CV2_AVAILABLE:
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            else:
                h, w = img1.shape[:2]
                new_h, new_w = int(h * scale_y), int(w * scale_x)
                img1_pil = Image.fromarray(img1)
                img2_pil = Image.fromarray(img2)
                img1_pil = img1_pil.resize((new_w, new_h), Image.BILINEAR)
                img2_pil = img2_pil.resize((new_w, new_h), Image.BILINEAR)
                img1 = np.array(img1_pil)
                img2 = np.array(img2_pil)
                
                flow_pil = Image.fromarray(flow.astype(np.uint8))
                flow_pil = flow_pil.resize((new_w, new_h), Image.BILINEAR)
                flow = np.array(flow_pil).astype(np.float32)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # 如果沒有指定 crop_size，則不進行裁剪
        if self.crop_size is not None:
            margin_y = 20
            margin_x = 50

            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
            x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

            y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
            x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    def __init__(self, crop_size=None, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1]):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        ht, wd = img1.shape[:2]
        
        # 如果沒有指定 crop_size，則不進行縮放限制
        if self.crop_size is not None:
            min_scale = np.maximum(
                (self.crop_size[0] + 1) / float(ht), 
                (self.crop_size[1] + 1) / float(wd))
        else:
            min_scale = 0.1  # 設置一個最小的縮放限制

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            if CV2_AVAILABLE:
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            else:
                h, w = img1.shape[:2]
                new_h, new_w = int(h * scale_y), int(w * scale_x)
                img1_pil = Image.fromarray(img1)
                img2_pil = Image.fromarray(img2)
                img1_pil = img1_pil.resize((new_w, new_h), Image.BILINEAR)
                img2_pil = img2_pil.resize((new_w, new_h), Image.BILINEAR)
                img1 = np.array(img1_pil)
                img2 = np.array(img2_pil)
                flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # 如果沒有指定 crop_size，則不進行裁剪
        if self.crop_size is not None:
            margin_y = 20
            margin_x = 50

            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
            x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

            y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
            x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        h, w = img1.shape[:2]
        # 完全移除 resize 功能，保持原始圖片大小
        logging.debug(f"保持原始圖片大小: {h}x{w}")

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class DroneDataset(StereoDataset):
    def __init__(self, aug_params=None, root='Dataset/dataset_Stereo_20251028', split='train'):
        super(DroneDataset, self).__init__(aug_params, sparse=True, reader=self._read_drone_disp)
        assert os.path.exists(root)
        assert split in ["train", "val", "test"]
        
        img0_path = osp.join(root, 'Img0', split)
        img1_path = osp.join(root, 'Img1', split)
        disp_path = osp.join(root, 'Disparity', split)
        
        logging.info(f"Looking for images in: {img0_path}")
        logging.info(f"Looking for images in: {img1_path}")
        logging.info(f"Looking for disparity in: {disp_path}")
        
        img0_files = sorted(glob(osp.join(img0_path, '*.png')))
        logging.info(f"Found {len(img0_files)} files in Img0/{split}")
        
        for img0_file in img0_files:
            filename = osp.basename(img0_file)
            
            # 正確的文件名替換邏輯
            img1_filename = filename.replace('Img0', 'Img1')
            img1_file = osp.join(img1_path, img1_filename)
            
            # 修復 Disparity 文件名生成邏輯
            disp_filename = filename.replace('Img0', 'Disparity').replace('.png', '.pfm')
            disp_file = osp.join(disp_path, disp_filename)
            
            if osp.exists(img1_file) and osp.exists(disp_file):
                self.image_list += [ [img0_file, img1_file] ]
                self.disparity_list += [ disp_file ]
            else:
                logging.warning(f"Missing files for {filename}: img1={osp.exists(img1_file)}, disp={osp.exists(disp_file)}")
                logging.warning(f"  Expected img1: {img1_file}")
                logging.warning(f"  Expected disp: {disp_file}")
                logging.warning(f"  This sample will be skipped")
        
        logging.info(f"Added {len(self.disparity_list)} samples from Drone dataset ({split} split)")
    
    def _read_drone_disp(self, file_name):
        return readPFM(file_name).astype(np.float32)

def fetch_dataloader(args):
    # 構建增廣參數
    aug_params = {'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    # 如果指定了 crop_size，添加到增廣參數中
    if hasattr(args, "crop_size") and args.crop_size is not None:
        aug_params["crop_size"] = args.crop_size
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name == 'drone':
            new_dataset = DroneDataset(aug_params, root=args.dataset_root, split='train')
            logging.info(f"Adding {len(new_dataset)} samples from Drone dataset")
        else:
            logging.warning(f"Unknown dataset: {dataset_name}")
            continue
        
        if train_dataset is None:
            train_dataset = new_dataset
        else:
            train_dataset = train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=0, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)

class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast('cuda', enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        corr_block = CorrBlock1D
        fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast('cuda', enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            delta_flow[:,1] = 0.0

            coords1 = coords1 + delta_flow

            if test_mode and itr < iters-1:
                continue

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    if valid.dim() == 3:
        valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
        valid = valid.expand_as(flow_gt)
    elif valid.dim() == 4:
        valid = ((valid >= 0.5) & (mag < max_flow))
        valid = valid.expand_as(flow_gt)
    else:
        raise ValueError(f"Unexpected valid shape: {valid.shape}")
    
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    valid_mask = valid.bool()
    # 修復維度處理：valid 是 [batch, 2, height, width]，epe 是 [batch, height, width]
    # 需要將 valid 降維到 [batch, height, width] 來匹配 epe
    valid_mask_2d = valid_mask[:, 0, :, :]  # 取第一個通道
    
    # 確保有有效的像素點
    if valid_mask_2d.sum() == 0:
        # 如果沒有有效像素，返回零指標
        metrics = {
            'epe': 0.0,
            '1px': 0.0,
            '3px': 0.0,
            '5px': 0.0,
        }
        return flow_loss, metrics
    
    epe = epe[valid_mask_2d]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # 創建 OneCycleLR 調度器
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, output_dir='.'):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.output_dir = output_dir
        # 修改：TensorBoard日志保存到output_dir/tensorboard目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        tensorboard_dir = Path(output_dir) / 'tensorboard' / f'stereo_training_{timestamp}'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%dT%H%M")
            tensorboard_dir = Path(self.output_dir) / 'tensorboard' / f'stereo_training_{timestamp}'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%dT%H%M")
            tensorboard_dir = Path(self.output_dir) / 'tensorboard' / f'stereo_training_{timestamp}'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args, progress_callback=None):
    """
    訓練函數
    
    Args:
        args: 訓練參數
        progress_callback: 可選的進度回調函數，簽名：callback(current_step, total_steps, message)
    """
    model = nn.DataParallel(RAFTStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, args.output_dir)
    
    # 如果提供了進度回調，發送初始狀態
    if progress_callback:
        progress_callback(0, args.num_steps, "初始化訓練 Initializing training...")

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint: {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn()

    validation_frequency = 10000

    # 兼容不同 PyTorch 版本的 GradScaler 初始化
    try:
        scaler = GradScaler('cuda', enabled=args.mixed_precision)
    except TypeError:
        # 舊版本 PyTorch 不支持 device 參數
        scaler = GradScaler(enabled=args.mixed_precision)

    # 為 OneCycleLR 進行初始化：先執行一次虛擬的 optimizer.step()
    # 這樣可以避免 "scheduler.step() before optimizer.step()" 的警告
    optimizer.zero_grad()
    dummy_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
    dummy_loss.backward()
    optimizer.step()

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            file_paths, image1, image2, flow, valid = data_blob
            image1, image2, flow, valid = [x.cuda() for x in [image1, image2, flow, valid]]

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(flow_predictions, flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 確保正確的調用順序：先 optimizer.step()，再 scheduler.step()
            scaler.step(optimizer)  # 這會調用 optimizer.step()
            scaler.update()
            scheduler.step()  # 在 optimizer.step() 之後調用

            logger.push(metrics)
            
            # 定期更新進度（每100步或每10%更新一次）
            if progress_callback and (total_steps % 100 == 0 or total_steps % max(1, args.num_steps // 10) == 0):
                progress_callback(total_steps, args.num_steps, f"訓練中 Training... Step {total_steps}/{args.num_steps}")

            if total_steps % validation_frequency == validation_frequency - 1:
                # 修改：checkpoints保存到output_dir/checkpoints目录
                checkpoint_dir = Path(args.output_dir) / 'checkpoints' / 'stereo_training'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_path = checkpoint_dir / f'{total_steps + 1}_{args.name}.pth'
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            # 修改：checkpoints保存到output_dir/checkpoints目录
            checkpoint_dir = Path(args.output_dir) / 'checkpoints' / 'stereo_training'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = checkpoint_dir / f'{total_steps + 1}_epoch_{args.name}.pth.gz'
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    # 修改：最终模型保存到output_dir/checkpoints目录
    checkpoint_dir = Path(args.output_dir) / 'checkpoints' / 'stereo_training'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    PATH = checkpoint_dir / f'{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    
    # 發送最終進度更新
    if progress_callback:
        progress_callback(args.num_steps, args.num_steps, "訓練完成 Training completed!")

    return str(PATH)

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['drone'], help="training datasets.")
    parser.add_argument('--dataset_root', default='Dataset/dataset_Stereo_20251028', help="root directory for datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    parser.add_argument('--output_dir', default='.', help='output directory for checkpoints and logs')
    
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    train(args)

class RAFTStereoTrainer:
    """
    RAFT-Stereo 訓練器類
    包裝 train 函數，使其可以通過 TrainingConfig 對象進行訓練
    """
    def __init__(self, config):
        """
        初始化訓練器
        
        Args:
            config: TrainingConfig 對象，包含所有訓練參數
        """
        self.config = config
        self._stop_requested = False
        
    def stop(self):
        """請求停止訓練"""
        self._stop_requested = True
    
    def train(self, progress_callback=None):
        """
        執行訓練
        
        Args:
            progress_callback: 可選的進度回調函數，簽名：callback(current_step, total_steps, message)
        
        Returns:
            str: 訓練完成後模型保存路徑
        """
        # 將 TrainingConfig 轉換為 argparse.Namespace 格式
        # 創建一個簡單的類來模擬 argparse.Namespace
        class Args:
            def __init__(self, config):
                self.name = config.name
                self.restore_ckpt = config.restore_ckpt
                self.mixed_precision = config.mixed_precision
                self.batch_size = config.batch_size
                self.train_datasets = config.train_datasets
                self.dataset_root = config.dataset_root
                self.lr = config.lr
                self.num_steps = config.num_steps
                self.train_iters = config.train_iters
                self.wdecay = config.wdecay
                self.valid_iters = config.valid_iters
                self.corr_implementation = config.corr_implementation
                self.shared_backbone = config.shared_backbone
                self.corr_levels = config.corr_levels
                self.corr_radius = config.corr_radius
                self.n_downsample = config.n_downsample
                self.context_norm = config.context_norm
                self.slow_fast_gru = config.slow_fast_gru
                self.n_gru_layers = config.n_gru_layers
                self.hidden_dims = config.hidden_dims
                self.img_gamma = config.img_gamma
                self.saturation_range = config.saturation_range
                self.do_flip = config.do_flip
                self.spatial_scale = config.spatial_scale
                self.noyjitter = config.noyjitter
                self.output_dir = config.output_dir
                
                # 處理 image_size
                # 注意：不使用 crop_size，以保持原始圖像尺寸進行訓練
                # 這避免了裁剪尺寸大於圖像尺寸的問題
                if hasattr(config, 'image_size') and config.image_size:
                    if isinstance(config.image_size, (list, tuple)) and len(config.image_size) == 2:
                        self.image_size = config.image_size
                        # 不使用 crop_size，讓圖像保持原始尺寸
                        self.crop_size = None
                    else:
                        self.image_size = config.image_size
                        self.crop_size = None
                else:
                    self.image_size = None
                    self.crop_size = None
        
        # 創建 args 對象
        args = Args(self.config)
        
        # 確保輸出目錄存在
        output_dir = Path(self.config.output_dir)
        checkpoints_dir = output_dir / "checkpoints"
        
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        
        # 設置日誌
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
        
        # 設置隨機種子
        torch.manual_seed(1234)
        np.random.seed(1234)
        
        # 調用原始的 train 函數，傳遞進度回調
        result_path = train(args, progress_callback=progress_callback)
        
        return result_path

if __name__ == '__main__':
    main()
