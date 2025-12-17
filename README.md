# YOLO-StereoVision GUI Trainer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11%2Fv12-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Modern PyQt5-based Unified Platform for YOLO Training and Inference**

[Features](#-features) •
[Quick Start](#-quick-start) •
[Project Structure](#-project-structure) •
[User Guide](#-user-guide) •
[Development](#-development)

</div>

---

## 📖 Introduction

YOLO-StereoVision GUI Trainer is a powerful deep learning training platform designed for YOLO object detection and stereo vision tasks. With a modular architecture and intuitive graphical interface, users can easily complete the entire workflow from data processing to model training and inference.

### ✨ Core Highlights

- 🎨 **Modern Interface** - Beautiful and user-friendly GUI based on PyQt5
- 🧩 **Modular Design** - Architecture following software engineering best practices
- 🚀 **Full-Featured Integration** - Complete workflow from data conversion to training, inference, and analysis
- 👁️ **Stereo Vision Support** - Built-in RAFT-Stereo and other stereo vision models
- 💾 **Intelligent Settings Management** - Automatic save and restore of user settings
- 🖥️ **GPU Acceleration** - Full CUDA acceleration support

---

## 🎯 Features

### 1. 🔄 Data Conversion
- **RGB Image Preprocessing** - Convert color images to grayscale
- **Format Conversion** - Support for multiple data format conversions
- **Batch Processing** - Efficiently process large datasets

### 2. 🚀 Model Training
- **YOLO Standard Training** - Support for YOLOv8/v11/v12 and more
- **Custom Training Parameters** - Epochs, Batch Size, Learning Rate, etc.
- **Real-time Monitoring** - Training process visualization and logging
- **Resume Training** - Resume from checkpoints

### 3. 🔍 Model Inference
- **Single/Batch Inference** - Flexible inference modes
- **Real-time Results Display** - Instant visualization of detection results
- **Multi-format Support** - Support for images, videos, and more

### 4. 📊 Model Analysis
- **Model Architecture Viewing** - Detailed network architecture information
- **Parameter Statistics** - Model size, layer count, parameter count, etc.
- **Performance Analysis** - Inference speed, memory usage metrics

### 5. 🔧 Model Modification
- **Channel Adjustment** - Modify input channel count (RGB ↔ Grayscale)
- **Model Conversion** - Convert between different model formats
- **Structure Optimization** - Model pruning and optimization

### 6. 👁️ Stereo Vision
- **RAFT-Stereo Training** - Professional stereo matching training
- **Depth Estimation** - Generate depth maps from binocular images
- **Stereo Inference** - Real-time stereo vision inference

---

## 🛠️ Tech Stack

### Core Frameworks
- **Python** 3.8+
- **PyTorch** 2.0+ - Deep learning framework
- **Ultralytics** 8.0+ - YOLO implementation
- **PyQt5** 5.15+ - GUI framework

### Vision Processing
- **OpenCV** 4.8+ - Image processing
- **Pillow** 10.0+ - Image manipulation

### Scientific Computing
- **NumPy** 1.24+ - Numerical computing
- **SciPy** 1.10+ - Scientific computing
- **Matplotlib** 3.7+ - Data visualization

---

## 🚀 Quick Start

### Requirements

```bash
# System Requirements
- Python 3.8 or higher
- CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 4GB+ GPU VRAM (for training)
```

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/YOLO-StereoVision_GUI_Trainer.git
cd YOLO-StereoVision_GUI_Trainer
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch Application
```bash
python yolo_launcher_gui_modular.py
```

---

## 📁 Project Structure

```
YOLO-StereoVision_GUI_Trainer/
│
├── yolo_launcher_gui_modular.py   # Main entry point
├── requirements.txt                # Dependencies list
├── README.md                       # This document
│
├── Code/                          # Core code
│   ├── data_converter.py          # Data conversion
│   ├── YOLO_standard_trainer.py   # YOLO trainer
│   ├── yolo_inference.py          # YOLO inference
│   ├── raft_stereo_trainer.py     # Stereo vision training
│   ├── evaluate_stereo.py         # Stereo vision evaluation
│   ├── model_modifier.py          # Model modification tool
│   ├── Read_Model.py              # Model analysis tool
│   └── backup/                    # Backup code
│
├── gui/                           # GUI modules
│   ├── modules/                   # Feature modules
│   │   ├── base_module.py         # Base module class
│   │   ├── data_converter.py     # Data conversion module
│   │   ├── yolo_training.py      # YOLO training module
│   │   ├── yolo_inference.py     # YOLO inference module
│   │   ├── model_analyzer.py     # Model analyzer module
│   │   ├── model_modifier.py     # Model modifier module
│   │   ├── stereo_training.py    # Stereo vision training
│   │   └── stereo_inference.py   # Stereo vision inference
│   │
│   ├── utils/                     # Utility functions
│   │   ├── logger.py              # Logging utilities
│   │   ├── file_utils.py          # File operations
│   │   ├── model_utils.py         # Model utilities
│   │   ├── ui_utils.py            # UI helpers
│   │   └── gpu_utils.py           # GPU utilities
│   │
│   ├── config/                    # Configuration management
│   │   ├── constants.py           # Constants definition
│   │   └── settings.py            # Settings manager
│   │
│   ├── workers/                   # Background worker threads
│   │   └── worker_thread.py       # Worker thread
│   │
│   └── README.md                  # GUI module documentation
│
├── config/                        # Application configuration
│   ├── config.py                  # Configuration class
│   ├── gui_settings.yaml          # GUI settings
│   └── predefined_classes.txt     # Predefined classes
│
├── Model_file/                    # Model files
│   ├── PT_File/                   # PyTorch models
│   ├── YAML/                      # YOLO config files
│   └── Stereo_Vision/             # Stereo vision models
│
├── runs/                          # Training outputs
│   └── (training outputs)
│
└── training_logs/                 # Training logs
    └── (training logs)
```

---

## 📚 User Guide

### 1️⃣ Data Preparation

#### YOLO Data Format
```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       ├── img001.jpg
│       └── img002.jpg
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── img002.txt
    └── val/
        ├── img001.txt
        └── img002.txt
```

#### Annotation Format (YOLO)
```
# Each line: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.3 0.4
1 0.3 0.7 0.2 0.2
```

### 2️⃣ Model Training

1. **Select Training Module** - Click the "🚀 Model Training" tab
2. **Choose Model** - Select a pretrained model from the dropdown
3. **Configure Parameters**:
   - Epochs: Number of training epochs (recommended: 100-300)
   - Batch Size: Batch size (recommended: 16-32)
   - Image Size: Image dimensions (recommended: 640)
   - Learning Rate: Learning rate (default: 0.01)
4. **Select Dataset** - Specify training data path
5. **Start Training** - Click "Start Training" button

### 3️⃣ Model Inference

1. **Select Inference Module** - Click the "🔍 Model Inference" tab
2. **Load Model** - Select a trained model file
3. **Choose Input** - Select image or video file
4. **Configure Parameters**:
   - Confidence: Confidence threshold (0-1)
   - IoU Threshold: IoU threshold (0-1)
5. **Run Inference** - Click "Start Inference" to view results

### 4️⃣ Model Analysis

1. **Load Model** - Select the model to analyze
2. **View Information**:
   - Model architecture
   - Parameter count
   - Input/output dimensions
   - FLOPs (computational complexity)

### 5️⃣ Stereo Vision

1. **Prepare Binocular Data** - Left and right view image pairs
2. **Configure Training Parameters** - Select stereo vision training module
3. **Execute Training** - Train depth estimation model
4. **Depth Inference** - Generate depth maps using trained model

---

## ⚙️ Configuration

### GUI Settings File
`config/gui_settings.yaml` - Stores application settings

```yaml
window:
  geometry:
    x: 100
    y: 100
    width: 1400
    height: 900
  last_tab_index: 0

training:
  last_model: "yolo11n.pt"
  default_epochs: 100
  default_batch_size: 16
```

### Predefined Classes
`config/predefined_classes.txt` - Custom detection classes

```
person
car
bicycle
dog
cat
...
```

---

## 🔧 Development

### Adding New Feature Modules

1. **Inherit Base Module**
```python
from gui.modules import BaseModule

class MyNewModule(BaseModule):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def create_tab(self):
        # Create UI
        tab = QWidget()
        # ... Implement UI
        return tab
```

2. **Register in Main Window**
```python
# yolo_launcher_gui_modular.py
self.my_module = MyNewModule(self)
self.tab_widget.addTab(self.my_module.create_tab(), "🆕 New Feature")
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add necessary comments and docstrings
- Modular design with high cohesion and low coupling

### Testing

```bash
# Run unit tests
pytest tests/

# Test specific module
pytest tests/test_data_converter.py
```

---

## 📝 Changelog

### Version 2.0.0 (Current)
- ✨ Complete modular refactoring
- 🎨 Modern UI design
- 💾 Intelligent settings management
- 👁️ Added stereo vision features
- 🔧 Enhanced model modification tools
- 📊 Improved model analysis features

### Version 1.0.0
- 🎉 Initial release
- 🚀 Basic YOLO training functionality
- 🔍 Basic inference functionality

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

## 👥 Author

- **ITA CHUNG** - *Initial work*

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) - Stereo vision model

---

## 📮 Contact

For questions or suggestions, please contact:

- 📧 Email: joe.chungita@gmail.com

---

## 📸 Screenshots

### Main Interface
![Main Interface](docs/images/Main%20Interface.png)

### Training Interface
![Training](docs/images/Training%20Interface.png)

### Inference Results
![Inference](docs/images/Inference%20Results.png)

---

<div align="center">

**⭐ If this project helps you, please give it a star!**

Made with ❤️ by YOLO Team

</div>
