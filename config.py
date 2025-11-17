"""
配置文件 - 定义文档去畸变的所有可配置参数

Configuration file for document dewarp parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DewarpConfig:
    """文档去畸变配置类

    Document dewarp configuration class containing all adjustable parameters.
    """

    # 图像处理参数 / Image processing parameters
    target_width: int = 7016  # 目标图像宽度 / Target image width
    target_height: int = 4960  # 目标图像高度 / Target image height
    pdf_zoom: float = 7.0  # PDF转图像时的缩放因子 / PDF to image zoom factor

    # 形态学操作参数 / Morphological operation parameters
    morph_kernel_size: Tuple[int, int] = (3, 3)  # 形态学核大小 / Morphological kernel size
    dilate_iterations: int = 3  # 膨胀迭代次数 / Dilation iterations
    erode_iterations: int = 5  # 腐蚀迭代次数 / Erosion iterations

    # 文本轮廓检测参数 / Text contour detection parameters
    text_kernel_size: Tuple[int, int] = (5, 5)
    text_erode_iterations: int = 3
    text_dilate_iterations: int = 4

    # 对比度增强参数 / Contrast enhancement parameters
    clahe_clip_limit: float = 1.2  # CLAHE对比度限制 / CLAHE clip limit
    clahe_tile_size: Tuple[int, int] = (8, 8)  # CLAHE分块大小 / CLAHE tile grid size
    high_contrast_limit: float = 2.0  # 高对比度增强限制 / High contrast limit

    # 轮廓提取参数 / Contour extraction parameters
    min_area_ratio: float = 0.0002  # 最小区域面积比例 / Minimum area ratio
    max_area_ratio: float = 0.95  # 最大区域面积比例 / Maximum area ratio
    page_detect_min_ratio: float = 0.3  # 页面检测最小比例 / Page detection min ratio

    # 多项式拟合参数 / Polynomial fitting parameters
    poly_degree: int = 7  # 多项式阶数 / Polynomial degree
    boundary_sample_points: int = 20  # 边界采样点数 / Boundary sample points

    # 曲面网格化参数 / Surface meshing parameters
    horizontal_samples: int = 51  # 横向采样数 / Horizontal samples
    vertical_samples: int = 20  # 纵向采样数 / Vertical samples

    # 页面类型 / Page type
    page_mode: str = 'double'  # 'single' 或 'double' / 'single' or 'double'

    # 调试选项 / Debug options
    debug: bool = False  # 是否输出调试信息 / Whether to output debug info
    save_intermediate: bool = False  # 是否保存中间结果 / Whether to save intermediate results

    # 高斯模糊参数 / Gaussian blur parameters
    gaussian_kernel_size: Tuple[int, int] = (0, 0)
    gaussian_sigma: float = 7.0
    sharpen_weight: float = 1.5  # 锐化权重 / Sharpen weight
    blur_weight: float = -0.5  # 模糊权重 / Blur weight

    # 边界偏移 / Boundary offset
    boundary_offset_y: int = 50  # Y方向边界偏移 / Y-direction boundary offset
    boundary_offset_x: int = 10  # X方向边界偏移 / X-direction boundary offset


# 默认配置实例 / Default configuration instance
DEFAULT_CONFIG = DewarpConfig()


# 单页模式配置 / Single page mode configuration
SINGLE_PAGE_CONFIG = DewarpConfig(
    page_mode='single',
    target_width=3508,  # A4纸宽度的一半 / Half of A4 width
    horizontal_samples=40,
)


# 高质量配置 / High quality configuration
HIGH_QUALITY_CONFIG = DewarpConfig(
    target_width=10000,
    target_height=7000,
    pdf_zoom=10.0,
    poly_degree=9,
    horizontal_samples=71,
    vertical_samples=30,
)


# 快速模式配置 / Fast mode configuration
FAST_CONFIG = DewarpConfig(
    target_width=3508,
    target_height=2480,
    pdf_zoom=3.0,
    poly_degree=5,
    horizontal_samples=31,
    vertical_samples=15,
)
