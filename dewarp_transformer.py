"""
去畸变变换模块 - 透视变换和图像重组

Dewarp transformer module for perspective transformation and image reconstruction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import DewarpConfig
from utils import create_blank_image


class DewarpTransformer:
    """去畸变变换器

    Dewarp transformer for applying perspective transformations
    and reconstructing the flattened image.
    """

    def __init__(self, config: Optional[DewarpConfig] = None):
        """
        初始化去畸变变换器

        Args:
            config: 去畸变配置对象
        """
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG

    def perspective_transform_block(
        self,
        image: np.ndarray,
        block_corners: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        """
        对单个网格块进行透视变换

        Apply perspective transformation to a single mesh block.

        Args:
            image: 输入图像
            block_corners: 网格块的四个角点，形状为 (4, 2)

        Returns:
            (变换后的图像, 宽度, 高度) 的元组
        """
        # 计算目标矩形的宽度（基于上边的长度）
        width = int(np.sum(np.square(
            block_corners[1] - block_corners[0]
        )) ** 0.5)

        # 计算目标矩形的高度（基于左边的高度差）
        height = abs(block_corners[2][1] - block_corners[0][1])

        # 定义目标矩形的四个角点
        target_corners = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])

        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(
            np.float32(block_corners),
            target_corners
        )

        # 应用透视变换
        transformed = cv2.warpPerspective(
            image,
            transform_matrix,
            (width, height)
        )

        return transformed, width, height

    def reconstruct_image(
        self,
        image: np.ndarray,
        mesh_blocks: np.ndarray
    ) -> np.ndarray:
        """
        重组图像（将所有变换后的块拼接）

        Reconstruct image by stitching all transformed blocks.

        Args:
            image: 输入图像
            mesh_blocks: 网格块数组，形状为 (n, 4, 2)

        Returns:
            重组后的图像
        """
        # 创建输出图像
        output = create_blank_image(
            self.config.target_height,
            self.config.target_width
        )

        x_offset = 0

        for block in mesh_blocks:
            # 对每个块进行透视变换
            transformed, width, height = self.perspective_transform_block(
                image,
                block
            )

            # 确保不超出边界
            if x_offset + width > self.config.target_width:
                width = self.config.target_width - x_offset

            if height > self.config.target_height:
                height = self.config.target_height

            # 拼接到输出图像
            try:
                output[
                    0:height,
                    x_offset:x_offset + width
                ] = transformed[0:height, 0:width]
            except Exception as e:
                if self.config.debug:
                    print(f"Warning: Block stitching error: {e}")
                continue

            x_offset += width

        # 调整大小到目标尺寸
        if x_offset > 0:
            output = cv2.resize(
                output[0:self.config.target_height, 0:x_offset],
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_CUBIC
            )

        return output

    def dewarp_curved_surface(
        self,
        image: np.ndarray,
        mesh_blocks: np.ndarray
    ) -> np.ndarray:
        """
        对弯曲表面进行去畸变处理

        Dewarp a curved surface.

        Args:
            image: 输入图像
            mesh_blocks: 网格块数组

        Returns:
            去畸变后的图像
        """
        return self.reconstruct_image(image, mesh_blocks)

    def apply_global_perspective_correction(
        self,
        image: np.ndarray,
        contour: np.ndarray
    ) -> np.ndarray:
        """
        应用全局透视校正

        Apply global perspective correction to the entire document.

        Args:
            image: 输入图像
            contour: 文档轮廓

        Returns:
            校正后的图像
        """
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 裁剪并留边距
        y_start = max(0, y - self.config.boundary_offset_y)
        y_end = min(image.shape[0], y + h + self.config.boundary_offset_y)
        x_start = max(0, x - self.config.boundary_offset_x)
        x_end = min(image.shape[1], x + w + self.config.boundary_offset_x)

        cropped = image[y_start:y_end, x_start:x_end]

        return cropped

    def enhance_and_transform(
        self,
        image: np.ndarray,
        enhanced: np.ndarray,
        mesh_blocks: np.ndarray,
        regions: Optional[List] = None
    ) -> np.ndarray:
        """
        增强并变换图像

        Enhance and transform image with optional region-specific enhancement.

        Args:
            image: 原始图像
            enhanced: 增强后的图像
            mesh_blocks: 网格块数组
            regions: 可选的需要特殊增强的区域列表

        Returns:
            处理后的图像
        """
        # 先进行去畸变
        dewarped = self.dewarp_curved_surface(enhanced, mesh_blocks)

        # 如果有特定区域需要增强
        if regions:
            from image_processor import ImageProcessor
            processor = ImageProcessor(self.config)

            for region in regions:
                x, y, w, h = region
                # 提取区域
                roi = image[y:y+h, x:x+w]
                # 高对比度增强
                enhanced_roi = processor.apply_clahe(
                    roi,
                    self.config.high_contrast_limit
                )
                # 应用到结果
                dewarped[y:y+h, x:x+w] = enhanced_roi

        return dewarped
