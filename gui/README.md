# GUI包 - 符合软件工程规范的目录结构

## 📦 包结构

```
gui/
├── __init__.py                 # 包初始化
├── main_window.py             # 主窗口类（未来）
│
├── modules/                    # 功能模块
│   ├── __init__.py
│   ├── base_module.py         # 基础模块类
│   └── ...                    # 其他功能模块
│
├── utils/                      # 工具函数 ⭐
│   ├── __init__.py
│   ├── logger.py              # 日志工具
│   ├── file_utils.py          # 文件操作
│   ├── model_utils.py         # 模型工具
│   └── ui_utils.py            # UI辅助
│
├── workers/                    # 后台工作线程 ⭐
│   ├── __init__.py
│   └── worker_thread.py       # 工作线程
│
├── config/                     # 配置管理 ⭐
│   ├── __init__.py
│   ├── constants.py           # 常量定义
│   └── settings.py            # 设置管理器
│
└── README.md                   # 本文档
```

## 🎯 设计理念

这个结构遵循**软件工程最佳实践**：

### 1. 分层架构
```
UI层 (modules/)     ← 用户界面和交互
├─ 业务逻辑层 (modules/)  ← 功能实现
├─ 工具层 (utils/)       ← 通用工具
├─ 配置层 (config/)      ← 配置管理
└─ 任务层 (workers/)     ← 后台任务
```

### 2. 职责单一
每个目录只负责一类功能：
- **modules/** - 业务逻辑（做什么）
- **utils/** - 工具函数（怎么做）
- **config/** - 配置管理（用什么配置）
- **workers/** - 后台任务（后台怎么做）

### 3. 高内聚低耦合
- 相关功能聚合在一起
- 模块间通过接口通信
- 避免循环依赖

## 🔧 核心模块说明

### utils/ - 工具函数目录

**logger.py** - 日志工具
```python
from gui.utils import setup_logger, log_message

logger = setup_logger('MyModule')
message = log_message("处理完成", level='INFO')
```

**file_utils.py** - 文件操作
```python
from gui.utils import ensure_dir, get_file_size, find_files

# 确保目录存在
output_dir = ensure_dir("output")

# 获取文件大小
size = get_file_size("model.pt", unit='MB')

# 查找文件
files = find_files(".", pattern='*.pt')
```

**model_utils.py** - 模型工具
```python
from gui.utils.model_utils import validate_model, get_model_channels

# 验证模型
result = validate_model('model.pt')

# 获取通道数
channels = get_model_channels('model.pt')
```

**ui_utils.py** - UI辅助
```python
from gui.utils import show_error, show_question, format_time

# 显示消息
show_error(self, "操作失败")

# 显示确认对话框
if show_question(self, "确认删除?"):
    # 执行删除
    pass

# 格式化时间
time_str = format_time(3665)  # "1.0小时"
```

### config/ - 配置管理

**constants.py** - 常量定义
```python
from gui.config.constants import DEFAULT_EPOCHS, COLOR_SUCCESS

epochs = DEFAULT_EPOCHS  # 100
label.setStyleSheet(f"color: {COLOR_SUCCESS}")
```

**settings.py** - 设置管理器
```python
from gui.config import SettingsManager

settings = SettingsManager()

# 读取配置
last_model = settings.get('training.last_model')

# 保存配置
settings.set('training.epochs', 200)
settings.save()

# 重置配置
settings.reset()
```

### workers/ - 后台工作线程

```python
from gui.workers import WorkerThread

# 创建工作线程
worker = WorkerThread('train', **params)
worker.progress.connect(self.on_progress)
worker.finished.connect(self.on_finished)
worker.start()
```

## 📚 使用示例

### 示例1: 创建新的功能模块

```python
from gui.modules import BaseModule
from gui.utils import log_message, show_error
from gui.config.constants import DEFAULT_BATCH_SIZE

class MyNewModule(BaseModule):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def create_tab(self):
        # 创建UI
        tab = QWidget()
        # ...
        return tab
        
    def process_data(self):
        # 使用工具函数
        self.log(log_message("开始处理..."))
        
        try:
            # 业务逻辑
            result = self.do_something()
            self.log(log_message("处理完成"))
        except Exception as e:
            show_error(self.parent, f"处理失败: {e}")
```

### 示例2: 使用配置和工具

```python
from gui.config import SettingsManager
from gui.config.constants import DEFAULT_EPOCHS
from gui.utils import ensure_dir, get_file_size

class Trainer:
    def __init__(self):
        # 加载配置
        self.settings = SettingsManager()
        self.epochs = self.settings.get('training.epochs', DEFAULT_EPOCHS)
        
        # 使用工具函数
        self.output_dir = ensure_dir('output/training')
        
    def train(self):
        # 训练逻辑
        for epoch in range(self.epochs):
            # ...
            pass
        
        # 保存结果
        model_size = get_file_size('model.pt')
        print(f"模型大小: {model_size:.2f} MB")
```

## 🎨 代码风格

### 导入顺序
```python
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
from PyQt5.QtWidgets import QWidget
import torch

# 3. 项目内部 - gui包
from gui.utils import log_message
from gui.config import SettingsManager

# 4. 项目内部 - 其他包
from Code.data_converter import RGBPreprocessor
```

### 命名规范
```python
# 模块/包: 小写+下划线
from gui.utils import file_utils

# 类: 大驼峰
class DataConversionModule:

# 函数/变量: 小写+下划线
def get_file_size():
    model_path = "model.pt"

# 常量: 大写+下划线
DEFAULT_EPOCHS = 100
```

## 🚀 开发进度

### Phase 1 ✅ 完成
- [x] 创建utils/目录及工具函数
- [x] 创建config/目录及配置管理
- [x] 创建workers/目录结构

### Phase 2 ✅ 完成
- [x] 将功能模块迁移到modules/ ⭐
- [x] 合并 gui_modules/ 到 gui/ ⭐
- [x] 更新所有导入路径 ⭐

### Phase 3 (进行中)
- [ ] 将WorkerThread迁移到workers/
- [ ] 增强工具函数功能
- [ ] 完善配置管理
- [ ] 创建统一的main_window.py

### Phase 4 (未来)
- [ ] 添加插件系统
- [ ] 支持主题切换
- [ ] 国际化支持

## 📖 相关文档

- `GUI_MERGE_SUMMARY.md` - ⭐ 目录合并总结（最新）
- `PROJECT_STRUCTURE.md` - 完整项目结构文档
- `SOFTWARE_ENGINEERING_UPGRADE.md` - 软件工程升级报告
- `MODULAR_REFACTORING_SUMMARY.md` - 重构总结
- `gui_modules_backup/README.md` - 旧模块说明（备份）

## 💡 为什么要这样设计？

### 传统方式的问题
```python
# ❌ 问题1: 工具函数散落各处
class ModuleA:
    def get_file_size(self, path):
        return os.path.getsize(path) / 1024**2

class ModuleB:
    def get_file_size(self, path):  # 重复代码！
        return os.path.getsize(path) / 1024**2

# ❌ 问题2: 配置硬编码
epochs = 100  # 魔法数字
learning_rate = 0.01  # 魔法数字
```

### 新方式的优势
```python
# ✅ 优势1: 工具函数统一管理
from gui.utils import get_file_size

class ModuleA:
    def process(self):
        size = get_file_size('file.txt')  # 复用

class ModuleB:
    def process(self):
        size = get_file_size('data.dat')  # 复用

# ✅ 优势2: 配置集中管理
from gui.config.constants import DEFAULT_EPOCHS, DEFAULT_LR

epochs = DEFAULT_EPOCHS
learning_rate = DEFAULT_LR
```

## 🎯 总结

这个目录结构符合**软件工程最佳实践**：

1. ✅ **清晰的职责分离** - 每个目录都有明确用途
2. ✅ **高度可维护** - 易于查找和修改代码
3. ✅ **便于扩展** - 添加新功能很简单
4. ✅ **易于测试** - 模块独立可测
5. ✅ **代码复用** - 工具函数集中管理
6. ✅ **专业规范** - 符合行业标准

---

**创建时间**: 2025-10-24  
**维护者**: YOLO Team  
**版本**: 2.0.0

