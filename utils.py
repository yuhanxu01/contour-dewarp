"""
工具函数模块 - 提供通用的辅助函数

Utility functions module providing common helper functions.
"""

import numpy as np
from typing import List, Tuple
from operator import itemgetter


def to_np_array(cv_contour: np.ndarray) -> np.ndarray:
    """
    将OpenCV轮廓格式转换为NumPy数组格式

    Convert OpenCV contour format to NumPy array format.

    Args:
        cv_contour: OpenCV轮廓，形状为 (n, 1, 2)

    Returns:
        NumPy数组，形状为 (n, 2)
    """
    return np.array([[point[0][0], point[0][1]] for point in cv_contour])


def get_xy(points: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    从点列表中提取x和y坐标

    Extract x and y coordinates from list of points.

    Args:
        points: 点列表，每个点为 [x, y]

    Returns:
        x坐标列表和y坐标列表的元组
    """
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return x, y


def xy_to_points(x: np.ndarray, y: np.ndarray) -> List[List[int]]:
    """
    将x和y坐标数组转换为点列表

    Convert x and y coordinate arrays to list of points.

    Args:
        x: x坐标数组
        y: y坐标数组

    Returns:
        点列表，每个点为 [x, y]
    """
    points = []
    for i in range(len(x)):
        points.append([int(x[i]), int(y[i])])
    return points


def sort_by_x(points: List) -> List:
    """
    按x坐标排序点列表

    Sort points by x coordinate.

    Args:
        points: 点列表

    Returns:
        排序后的点列表
    """
    return sorted(points, key=itemgetter(0))


def sort_by_y(points: List) -> List:
    """
    按y坐标排序点列表

    Sort points by y coordinate.

    Args:
        points: 点列表

    Returns:
        排序后的点列表
    """
    return sorted(points, key=itemgetter(1))


def min_dist_point(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    找到距离目标点最近的点

    Find the point with minimum distance to target point.

    Args:
        points: 点数组
        target: 目标点

    Returns:
        距离最近的点
    """
    min_dist = np.inf
    min_point = None
    for point in points:
        dist = np.sum(np.square(target - point))
        if dist < min_dist:
            min_dist = dist
            min_point = point
    return min_point


def min_dist_x(points: List, target: List) -> List:
    """
    找到x坐标距离目标点最近的点

    Find the point with minimum x-distance to target point.

    Args:
        points: 点列表
        target: 目标点

    Returns:
        x坐标距离最近的点
    """
    min_dist = np.inf
    min_point = None
    for point in points:
        dist = np.square(target[0] - point[0])
        if dist < min_dist:
            min_dist = dist
            min_point = point
    return min_point


def get_edge_points(points: np.ndarray, width: int, height: int) -> List:
    """
    获取图像四个角的边缘点

    Get edge points at four corners of the image.

    Args:
        points: 点数组
        width: 图像宽度
        height: 图像高度

    Returns:
        四个角的边缘点列表
    """
    result = []
    result.append(min_dist_point(points, np.array([0, 0])))
    result.append(min_dist_point(points, np.array([width, 0])))
    result.append(min_dist_point(points, np.array([0, height])))
    result.append(min_dist_point(points, np.array([width, height])))
    return result


def divide_by_y(points: List, y_threshold: float) -> Tuple[List, List]:
    """
    根据y阈值将点分为上下两组

    Divide points into upper and lower groups by y threshold.

    Args:
        points: 点列表
        y_threshold: y坐标阈值

    Returns:
        上方点列表和下方点列表的元组
    """
    upper = []
    lower = []
    for point in points:
        if point[1] > y_threshold:
            upper.append(point)
        else:
            lower.append(point)
    return upper, lower


def divide_by_x_inclusive(points: List, x_threshold: float) -> Tuple[List, List]:
    """
    根据x阈值将点分为左右两组（包含边界点）

    Divide points into left and right groups by x threshold (inclusive).

    Args:
        points: 点列表
        x_threshold: x坐标阈值

    Returns:
        左侧点列表和右侧点列表的元组
    """
    left = []
    right = []
    for point in points:
        if point[0] > x_threshold:
            right.append(point)
        elif point[0] == x_threshold:
            right.append(point)
            left.append(point)
        else:
            left.append(point)
    return left, right


def filter_by_x_range(points: List, x_range: Tuple[float, float]) -> List:
    """
    根据x范围过滤点

    Filter points by x range.

    Args:
        points: 点列表
        x_range: x范围，(min_x, max_x)

    Returns:
        在范围内的点列表
    """
    interval = []
    for point in points:
        if x_range[0] < point[0] < x_range[1]:
            interval.append(point)
    return interval


def get_next_point(point: List, points: List) -> List:
    """
    获取点列表中当前点的下一个点

    Get the next point in the points list.

    Args:
        point: 当前点
        points: 点列表

    Returns:
        下一个点，如果是最后一个点则返回当前点
    """
    try:
        index = points.index(point) + 1
        if index >= len(points):
            return point
        return points[index]
    except ValueError:
        return point


def create_blank_image(height: int, width: int) -> np.ndarray:
    """
    创建空白图像

    Create a blank image.

    Args:
        height: 图像高度
        width: 图像宽度

    Returns:
        空白图像数组
    """
    return np.zeros((height, width), dtype=np.uint8)
