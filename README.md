# Contour-Dewarp | 文档去畸变工具

[![Paper](https://img.shields.io/badge/arXiv-2212.13489-b31b1b.svg)](https://doi.org/10.48550/arXiv.2212.13489)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于轮廓估计曲面细分的文档图像平整化工具。通过分析文档轮廓的曲率，智能生成网格并进行透视逆变换，实现弯曲文档的自动展平。

**A document dewarping tool based on contour estimation and surface subdivision.** Automatically flattens curved documents by analyzing contour curvature, generating adaptive mesh grids, and applying perspective transformations.

![Demo](https://user-images.githubusercontent.com/126972674/222947144-a5e8f03e-5950-4318-a2af-67a3fe6c6a6c.png)

## ✨ 主要特性 | Features

- 📄 **多种输入格式** - 支持单张图片（JPG, PNG, BMP等）和PDF文件
- 📖 **双页/单页模式** - 自动处理书籍跨页或单页文档
- 🎯 **智能网格生成** - 基于轮廓曲率的自适应网格细分
- ⚡ **多种质量预设** - 快速模式、默认模式、高质量模式可选
- 🔧 **高度可配置** - 丰富的参数配置，满足不同场景需求
- 🐍 **纯Python实现** - 易于集成和扩展
- 💻 **命令行工具** - 提供便捷的CLI接口
- 📚 **批量处理** - 支持批量处理多个文件

## 📖 算法原理 | Algorithm

本项目实现了论文 [Flattening Surface Based On Using Contour Estimating Subdivision Surface](https://doi.org/10.48550/arXiv.2212.13489) 中描述的方法：

1. **轮廓预处理** - 提取文档轮廓，分割为光滑曲面
2. **轮廓函数拟合** - 使用多项式拟合边界曲线
3. **网格化曲面** - 根据曲率自适应细分网格
4. **透视逆变换** - 将每个网格块透视变换并拼接

**The implementation follows the method described in the paper:**
- Contour preprocessing and segmentation
- Polynomial fitting of boundary curves
- Curvature-based adaptive mesh generation
- Perspective transformation and reconstruction

## 🚀 快速开始 | Quick Start

### 安装依赖 | Installation

```bash
# 基础依赖
pip install opencv-python numpy

# PDF支持（可选）
pip install PyMuPDF Pillow

# 完整安装
pip install -r requirements.txt
```

### 基本用法 | Basic Usage

#### 方式1: 使用命令行工具 | CLI

```bash
# 处理单张图片
python cli.py image input.jpg output.jpg

# 处理PDF
python cli.py pdf input.pdf output.pdf

# 单页模式
python cli.py image input.jpg output.jpg --page-mode single

# 批量处理
python cli.py batch input_dir/ output_dir/

# 高质量模式
python cli.py image input.jpg output.jpg --preset high_quality
```

#### 方式2: 作为Python库使用 | As Library

```python
from document_dewarp import DocumentDewarp

# 创建去畸变器
dewarp = DocumentDewarp()

# 处理图片
result = dewarp.process_image('input.jpg')

# 保存结果
import cv2
cv2.imwrite('output.jpg', result)
```

## 📘 详细使用 | Detailed Usage

### 单页模式 | Single Page Mode

```python
from document_dewarp import DocumentDewarp
from config import SINGLE_PAGE_CONFIG

# 使用单页配置
dewarp = DocumentDewarp(config=SINGLE_PAGE_CONFIG)
result = dewarp.process_image('single_page.jpg')
```

### 双页模式（书籍） | Double Page Mode

```python
from document_dewarp import DocumentDewarp
from config import DEFAULT_CONFIG

# 默认配置即为双页模式
dewarp = DocumentDewarp(config=DEFAULT_CONFIG)
result = dewarp.process_image('book_spread.jpg')
```

### 处理PDF | PDF Processing

```python
from document_dewarp import DocumentDewarp

dewarp = DocumentDewarp()

# 带进度回调
def progress(current, total):
    print(f"Processing page {current}/{total}")

dewarp.process_pdf('input.pdf', 'output.pdf', progress_callback=progress)
```

### 批量处理 | Batch Processing

```python
from document_dewarp import DocumentDewarp

dewarp = DocumentDewarp()

# 批量处理图片
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
output_paths = dewarp.process_images_batch(
    image_paths,
    output_dir='output/',
    output_format='png'
)
```

### 自定义配置 | Custom Configuration

```python
from document_dewarp import DocumentDewarp
from config import DewarpConfig

# 创建自定义配置
config = DewarpConfig(
    page_mode='single',          # 单页模式
    target_width=5000,            # 目标宽度
    target_height=3500,           # 目标高度
    poly_degree=8,                # 多项式阶数
    horizontal_samples=60,        # 横向采样数
    vertical_samples=25,          # 纵向采样数
    debug=True                    # 调试模式
)

dewarp = DocumentDewarp(config=config)
result = dewarp.process_image('input.jpg')
```

## ⚙️ 配置参数 | Configuration

### 预设配置 | Presets

| 预设 | 说明 | 适用场景 |
|-----|------|---------|
| `DEFAULT_CONFIG` | 默认配置（双页） | 一般书籍扫描 |
| `SINGLE_PAGE_CONFIG` | 单页配置 | 单页文档 |
| `HIGH_QUALITY_CONFIG` | 高质量配置 | 高分辨率文档 |
| `FAST_CONFIG` | 快速配置 | 快速预览 |

### 主要参数 | Key Parameters

```python
DewarpConfig(
    # 页面模式
    page_mode='double',           # 'single' 或 'double'

    # 图像尺寸
    target_width=7016,            # 目标图像宽度
    target_height=4960,           # 目标图像高度

    # 多项式拟合
    poly_degree=7,                # 多项式阶数（建议5-9）

    # 网格采样
    horizontal_samples=51,        # 横向采样点数
    vertical_samples=20,          # 纵向采样点数

    # 对比度增强
    clahe_clip_limit=1.2,         # CLAHE对比度限制

    # PDF处理
    pdf_zoom=7.0,                 # PDF转图像的缩放因子

    # 调试选项
    debug=False,                  # 是否输出调试信息
    save_intermediate=False       # 是否保存中间结果
)
```

## 📁 项目结构 | Project Structure

```
contour-dewarp/
├── config.py                 # 配置文件
├── utils.py                  # 工具函数
├── image_processor.py        # 图像预处理
├── contour_extractor.py      # 轮廓提取
├── surface_mesh.py           # 曲面网格化
├── dewarp_transformer.py     # 去畸变变换
├── document_dewarp.py        # 主类
├── cli.py                    # 命令行接口
├── example.py                # 使用示例
├── requirements.txt          # 依赖列表
├── README.md                 # 项目说明
└── scan-draft.py            # 原始代码（已废弃）
```

## 💡 使用示例 | Examples

完整示例请参考 `example.py` 文件，包含10个不同场景的使用示例：

1. 基础图片处理
2. 单页模式
3. 双页模式
4. 自定义配置
5. PDF处理
6. 批量处理
7. NumPy数组输入
8. 高质量模式
9. 快速模式
10. 灰度输出

运行示例：
```bash
python example.py
```

## 🔬 技术细节 | Technical Details

### 算法流程 | Algorithm Flow

```
输入图像
  ↓
图像预处理（锐化、二值化）
  ↓
轮廓提取与分割
  ↓
边界曲线多项式拟合
  ↓
基于曲率的网格生成
  ↓
透视逆变换
  ↓
图像重组与拼接
  ↓
输出展平图像
```

### 核心技术 | Core Technologies

- **形态学操作** - 用于文档区域分离
- **多项式拟合** - 7阶多项式拟合边界曲线
- **曲率计算** - κ(x) = P''(x) / (1 + P'(x)²)^(3/2)
- **自适应采样** - 根据曲率动态调整网格密度
- **透视变换** - OpenCV的getPerspectiveTransform

## ⚠️ 限制与注意事项 | Limitations

1. **轮廓清晰度** - 需要文档与背景有明显对比
2. **弯曲类型** - 适用于单调弯曲，不适用于褶皱
3. **拍摄角度** - 建议尽量垂直拍摄
4. **计算资源** - 高质量模式需要较多内存和计算时间

## 📊 性能参考 | Performance

| 模式 | 处理时间 | 输出质量 | 内存占用 |
|-----|---------|---------|---------|
| 快速 | ~2s | 中 | ~500MB |
| 默认 | ~5s | 高 | ~1GB |
| 高质量 | ~10s | 极高 | ~2GB |

*测试环境: Intel i7, 16GB RAM, 3000x2000像素输入*

## 🤝 贡献 | Contributing

欢迎提交Issue和Pull Request！

如果这个项目对你有帮助，请给个Star ⭐

## 📄 许可证 | License

MIT License

## 📚 引用 | Citation

如果在学术研究中使用本项目，请引用：

```bibtex
@article{xu2022flattening,
  title={Flattening Surface Based On Using Contour Estimating Subdivision Surface},
  author={Xu, Yuhan and Luo, Renqing},
  journal={arXiv preprint arXiv:2212.13489},
  year={2022}
}
```

## 📮 联系方式 | Contact

- Email: yuhanx5@uw.edu
- GitHub: [yuhanxu01/contour-dewarp](https://github.com/yuhanxu01/contour-dewarp)

## 🙏 致谢 | Acknowledgments

- OpenCV团队提供的优秀图像处理库
- 所有贡献者和用户的支持

---

**Made with ❤️ by Yuhan Xu**
