"""
曲面网格化模块 - 根据轮廓曲率生成网格

Surface meshing module for generating mesh grids based on contour curvature.
"""

import numpy as np
from typing import List, Tuple, Optional
from config import DewarpConfig
from utils import sort_by_x, min_dist_x, get_next_point


class SurfaceMesh:
    """曲面网格生成器

    Surface mesh generator for creating subdivision grids based on
    boundary curvatures.
    """

    def __init__(self, config: Optional[DewarpConfig] = None):
        """
        初始化曲面网格生成器

        Args:
            config: 去畸变配置对象
        """
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG

    def calculate_curvature(
        self,
        x: float,
        poly_coeffs: np.ndarray
    ) -> float:
        """
        计算多项式曲线在指定点的曲率

        Calculate curvature of polynomial curve at given point.

        公式: κ(x) = P''(x) / (1 + P'(x)^2)^(3/2)

        Args:
            x: x坐标
            poly_coeffs: 多项式系数

        Returns:
            曲率值
        """
        # 创建多项式对象
        poly = np.poly1d(poly_coeffs)

        # 一阶导数
        poly_first = np.polyder(poly, 1)
        first_deriv = poly_first(x)

        # 二阶导数
        poly_second = np.polyder(poly, 2)
        second_deriv = poly_second(x)

        # 计算曲率
        curvature = second_deriv / ((1 + first_deriv ** 2) ** 1.5)

        return curvature

    def calculate_scaling_factors(
        self,
        upper_coeffs: np.ndarray,
        lower_coeffs: np.ndarray,
        x_points: np.ndarray
    ) -> np.ndarray:
        """
        计算采样点的缩放因子

        Calculate scaling factors for sampling points based on curvature.

        Args:
            upper_coeffs: 上边界多项式系数
            lower_coeffs: 下边界多项式系数
            x_points: x坐标点数组

        Returns:
            缩放因子数组
        """
        scaling_factors = []

        for x in x_points:
            # 计算上下边界的曲率
            kappa_upper = self.calculate_curvature(x, upper_coeffs)
            kappa_lower = self.calculate_curvature(x, lower_coeffs)

            # 计算平均曲率差
            k = (kappa_lower - kappa_upper) / 2

            # 计算缩放因子
            # γ = |k|^(-k/|k|)
            if abs(k) < 1e-6:  # 避免除零
                gamma = 1.0
            else:
                gamma = abs(k) ** (-k / abs(k))

            scaling_factors.append(gamma)

        return np.array(scaling_factors)

    def normalize_scaling_factors(
        self,
        scaling_factors: np.ndarray
    ) -> np.ndarray:
        """
        归一化缩放因子

        Normalize scaling factors.

        Args:
            scaling_factors: 原始缩放因子数组

        Returns:
            归一化后的缩放因子数组
        """
        # 跳过第一个点（起始点）
        factors = scaling_factors[1:]
        total = np.sum(factors)

        if total == 0:
            return np.ones_like(factors) / len(factors)

        normalized = factors / total
        return normalized

    def calculate_sample_points(
        self,
        x_start: float,
        x_end: float,
        scaling_factors: np.ndarray
    ) -> np.ndarray:
        """
        根据缩放因子计算采样点位置

        Calculate sample point positions based on scaling factors.

        Args:
            x_start: 起始x坐标
            x_end: 结束x坐标
            scaling_factors: 归一化的缩放因子数组

        Returns:
            采样点x坐标数组
        """
        total_range = x_end - x_start
        sample_points = [x_start]

        current_x = x_start
        for factor in scaling_factors:
            current_x += factor * total_range
            sample_points.append(current_x)

        return np.array(sample_points)

    def generate_mesh_grid(
        self,
        upper_boundary: List,
        lower_boundary: List,
        upper_left_coeffs: np.ndarray,
        upper_right_coeffs: np.ndarray,
        lower_left_coeffs: np.ndarray,
        lower_right_coeffs: np.ndarray,
        x_mid: float
    ) -> np.ndarray:
        """
        生成网格点阵

        Generate mesh grid points.

        Args:
            upper_boundary: 上边界点列表
            lower_boundary: 下边界点列表
            upper_left_coeffs: 上边界左侧系数
            upper_right_coeffs: 上边界右侧系数
            lower_left_coeffs: 下边界左侧系数
            lower_right_coeffs: 下边界右侧系数
            x_mid: 中间分割线x坐标

        Returns:
            网格块数组，形状为 (n, 4, 2)，每个块有4个角点
        """
        # 确定起始和结束坐标
        x_start = upper_boundary[0][0]
        x_end = upper_boundary[-1][0]

        # 生成初始采样点用于计算曲率
        num_samples = self.config.vertical_samples
        x_samples = np.linspace(x_start, x_end, num_samples)

        # 计算缩放因子（基于曲率）
        # 这里使用上边界的左右系数
        scaling_factors = self.calculate_scaling_factors(
            upper_left_coeffs,
            lower_left_coeffs,
            x_samples
        )

        # 归一化
        normalized_factors = self.normalize_scaling_factors(scaling_factors)

        # 计算实际采样点位置
        x_points = self.calculate_sample_points(
            x_start,
            x_end,
            normalized_factors
        )

        # 生成网格块
        blocks = self._create_blocks(
            upper_boundary,
            lower_boundary,
            x_points
        )

        return blocks

    def _create_blocks(
        self,
        upper: List,
        lower: List,
        x_points: np.ndarray
    ) -> np.ndarray:
        """
        根据上下边界和x采样点创建网格块

        Create mesh blocks from upper/lower boundaries and x sample points.

        Args:
            upper: 上边界点列表
            lower: 下边界点列表
            x_points: x坐标采样点

        Returns:
            网格块数组
        """
        # 确定哪个边界点更多
        if len(upper) >= len(lower):
            begin_boundary = lower
            end_boundary = upper
        else:
            begin_boundary = upper
            end_boundary = lower

        blocks = []

        for i, point in enumerate(begin_boundary[:-1]):
            # 找到对侧最近的点
            other_side = min_dist_x(end_boundary, point)
            # 找到右侧的点
            point_right = get_next_point(point, begin_boundary)
            other_side_right = get_next_point(other_side, end_boundary)

            # 创建四边形块
            block = [
                point,
                point_right,
                other_side,
                other_side_right
            ]
            blocks.append(block)

        return np.array(blocks)

    def generate_simple_grid(
        self,
        upper_boundary: List,
        lower_boundary: List
    ) -> np.ndarray:
        """
        生成简单的均匀网格（用于单页模式）

        Generate simple uniform grid (for single page mode).

        Args:
            upper_boundary: 上边界点列表
            lower_boundary: 下边界点列表

        Returns:
            网格块数组
        """
        return self._create_blocks(upper_boundary, lower_boundary, None)
