"""
轮廓提取模块 - 提取文档轮廓并进行多项式拟合

Contour extraction module for extracting document contours and polynomial fitting.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import DewarpConfig
from utils import (
    to_np_array, get_xy, xy_to_points, sort_by_x, sort_by_y,
    divide_by_y, divide_by_x_inclusive, filter_by_x_range,
    get_edge_points
)


class ContourExtractor:
    """轮廓提取器类

    Contour extractor for finding and processing document boundaries.
    """

    def __init__(self, config: Optional[DewarpConfig] = None):
        """
        初始化轮廓提取器

        Args:
            config: 去畸变配置对象
        """
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG

    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        查找图像中的轮廓

        Find contours in binary image.

        Args:
            binary_image: 二值图像

        Returns:
            轮廓列表
        """
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def get_largest_contour(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        获取最大的轮廓

        Get the largest contour by area.

        Args:
            contours: 轮廓列表

        Returns:
            最大的轮廓
        """
        if not contours:
            raise ValueError("No contours found")

        max_contour = contours[0]
        max_area = cv2.contourArea(max_contour)

        for contour in contours[1:]:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_contour = contour
                max_area = area

        return max_contour

    def fit_polynomial(
        self,
        points: List,
        degree: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对点进行多项式拟合

        Fit polynomial to points.

        Args:
            points: 点列表
            degree: 多项式阶数，None则使用配置值

        Returns:
            (x_fitted, y_fitted, coefficients) 的元组
        """
        if degree is None:
            degree = self.config.poly_degree

        x, y = get_xy(points)
        coeffs = np.polyfit(x, y, degree)

        # 生成拟合点
        x_new = np.linspace(x[0], x[-1], self.config.boundary_sample_points)
        poly = np.poly1d(coeffs)
        y_new = poly(x_new)

        return x_new, y_new, coeffs

    def get_fitted_points(
        self,
        x_start: float,
        x_end: float,
        coeffs: np.ndarray,
        num_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据多项式系数生成拟合点

        Generate fitted points from polynomial coefficients.

        Args:
            x_start: 起始x坐标
            x_end: 结束x坐标
            coeffs: 多项式系数
            num_points: 点的数量，None则使用配置值

        Returns:
            (x_array, y_array) 的元组
        """
        if num_points is None:
            num_points = self.config.boundary_sample_points

        x_new = np.linspace(x_start, x_end, num_points)
        poly = np.poly1d(coeffs)
        y_new = poly(x_new)

        return x_new, y_new

    def fit_smooth_curve(
        self,
        points: List,
        x_mid: float
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        对点进行平滑曲线拟合（分左右两部分）

        Fit smooth curve to points (split into left and right parts).

        Args:
            points: 点列表
            x_mid: 中点x坐标

        Returns:
            (拟合点列表, 左侧系数, 右侧系数) 的元组
        """
        # 分为左右两部分
        left, right = divide_by_x_inclusive(points, x_mid)

        # 分别拟合
        xl, yl, left_coeffs = self.fit_polynomial(left)
        xr, yr, right_coeffs = self.fit_polynomial(right)

        # 合并拟合点
        fitted_points = xy_to_points(xl, yl) + xy_to_points(xr, yr)

        return fitted_points, left_coeffs, right_coeffs

    def interpolate_curves(
        self,
        x_start: float,
        x_end: float,
        coeffs_start: np.ndarray,
        coeffs_end: np.ndarray,
        num_steps: Optional[int] = None
    ) -> List[List[float]]:
        """
        在两条曲线之间进行插值

        Interpolate between two curves.

        Args:
            x_start: 起始x坐标
            x_end: 结束x坐标
            coeffs_start: 起始曲线系数
            coeffs_end: 结束曲线系数
            num_steps: 插值步数，None则使用配置值

        Returns:
            插值点列表
        """
        if num_steps is None:
            num_steps = self.config.horizontal_samples

        # 计算系数差值
        coeffs_diff = (coeffs_end - coeffs_start) / (num_steps - 1)

        points = []
        current_coeffs = coeffs_start.copy()

        for i in range(num_steps):
            x_new, y_new = self.get_fitted_points(
                x_start,
                x_end,
                current_coeffs,
                self.config.vertical_samples
            )

            # 添加点（跳过第一个点避免重复）
            for j in range(1, len(x_new)):
                points.append([x_new[j], y_new[j]])

            current_coeffs += coeffs_diff

        return points

    def extract_boundary_curves(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[List, List]:
        """
        提取轮廓的上下边界曲线

        Extract upper and lower boundary curves from contour.

        Args:
            contour: 输入轮廓
            image_shape: 图像形状 (height, width)

        Returns:
            (上边界点列表, 下边界点列表) 的元组
        """
        height, width = image_shape
        points = to_np_array(contour)

        # 根据y坐标的中值分为上下两部分
        y_mid = height // 2

        # 获取x范围
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()

        # 获取在x范围内的点并加上边缘点
        x_range = (x_min + 100, x_max - 100)  # 留一些边距
        interval_points = filter_by_x_range(points.tolist(), x_range)
        all_points = interval_points + get_edge_points(points, width, height)

        # 分为上下两组
        upper, lower = divide_by_y(all_points, y_mid)

        # 按x坐标排序并去重
        upper = sort_by_x(np.unique(upper, axis=0).tolist())
        lower = sort_by_x(np.unique(lower, axis=0).tolist())

        return upper, lower

    def split_double_page(
        self,
        upper: List,
        lower: List
    ) -> Tuple[float, float]:
        """
        分割双页（找到书脊位置）

        Split double page by finding the spine location.

        Args:
            upper: 上边界点列表
            lower: 下边界点列表

        Returns:
            (上边界中点x坐标, 下边界中点x坐标) 的元组
        """
        from operator import itemgetter

        # 找到上边界的最低点（书脊处）
        upper_mid_point = min(upper[1:-1], key=itemgetter(1))
        upper_mid_x = upper_mid_point[0]

        # 找到下边界的最高点（书脊处）
        lower_mid_point = max(lower[1:-1], key=itemgetter(1))
        lower_mid_x = lower_mid_point[0]

        return upper_mid_x, lower_mid_x
