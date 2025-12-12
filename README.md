# YOLO 统一启动器
**YOLO Unified Launcher - Professional GUI Application**

一个基于PyQt5的专业YOLO模型训练、推理和管理工具。

---

## 🚀 快速开始 Quick Start

### 启动GUI / Launch GUI

**方法1：双击启动**
```
双击 run_gui.bat
```

**方法2：命令行**
```bash
python yolo_launcher_gui_modular.py
```

---

## ✨ 主要功能 Features

| 功能模块 | 说明 |
|---------|------|
| 🔄 **数据转换** | Forest数据集转换（RGB/RGBD/立体视觉） |
| 🚀 **模型训练** | YOLO模型训练（支持预训练和从头训练） |
| 🔍 **模型推理** | 单张/批量/视频推理 |
| 📊 **模型分析** | 模型结构和参数分析 |
| 🔧 **模型修改** | 修改模型输入通道数（3↔4通道） |
| 👁️ **立体视觉** | RAFT-Stereo立体视觉训练 |

---

## 📁 项目结构 Structure

```
YOLO/
├── gui/                          # GUI包
│   ├── modules/                  # 功能模块
│   ├── utils/                    # 工具函数
│   ├── config/                   # 配置管理
│   └── workers/                  # 后台任务
│
├── Code/                         # 核心代码
├── Dataset/                      # 数据集
├── Model_file/                   # 模型文件
│   ├── PT_File/                  # PyTorch模型
│   ├── Stereo_Vision/            # 立體視覺預訓練權重（.pth）
│   └── YAML/                     # YAML配置
│
├── yolo_launcher_gui_modular.py  # 主程序
└── run_gui.bat                   # 启动脚本
```

---

## 🛠️ 环境要求 Requirements

- Python 3.8+
- PyQt5
- PyTorch
- Ultralytics YOLO
- CUDA（推荐用于GPU加速）

---

## 📖 使用文档 Documentation

### GUI包使用
详见 [`gui/README.md`](gui/README.md) - 完整的GUI包使用文档

### 历史文档
如需查看详细的架构演进和重构历史，请查看 [`docs/archive/`](docs/archive/) 目录

---

## 💡 开发示例 Development

### 导入功能模块
```python
from gui.modules import TrainingModule, InferenceModule
```

### 使用工具函数
```python
from gui.utils import ensure_dir, get_file_size, log_message
```

### 使用配置
```python
from gui.config import SettingsManager
from gui.config.constants import DEFAULT_EPOCHS
```

---

## 🎯 架构特点 Architecture

- ✅ **模块化设计** - 清晰的职责分离
- ✅ **专业结构** - 符合软件工程规范
- ✅ **易于维护** - 高内聚低耦合
- ✅ **工具丰富** - 完整的工具函数库
- ✅ **配置管理** - 统一的配置系统

---

## 📝 更新日志 Changelog

### v2.0.0 (2025-10-24)
- ✅ 完成GUI目录合并
- ✅ 采用专业模块化架构
- ✅ 添加完整的utils工具库
- ✅ 统一配置管理系统

### v1.0.0
- ✅ 基础模块化重构
- ✅ 7大功能模块实现

---

**Version**: 2.0.0  
**Last Update**: 2025-10-24  
**Status**: ✅ Production Ready

"# YOLO-StereoVision_GUI_Trainer" 
