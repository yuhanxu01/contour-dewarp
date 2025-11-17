"""
图像预处理模块 - 提供图像增强、二值化等预处理功能

Image preprocessing module providing image enhancement and binarization.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config import DewarpConfig


class ImageProcessor:
    """图像处理器类

    Image processor for preprocessing operations like enhancement,
    binarization, and morphological transformations.
    """

    def __init__(self, config: Optional[DewarpConfig] = None):
        """
        初始化图像处理器

        Args:
            config: 去畸变配置对象
        """
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        锐化图像

        Sharpen the image using unsharp masking.

        Args:
            image: 输入灰度图像

        Returns:
            锐化后的图像
        """
        blurred = cv2.GaussianBlur(
            image,
            self.config.gaussian_kernel_size,
            self.config.gaussian_sigma
        )
        sharpened = cv2.addWeighted(
            image,
            self.config.sharpen_weight,
            blurred,
            self.config.blur_weight,
            0
        )
        return sharpened

    def apply_clahe(self, image: np.ndarray, clip_limit: Optional[float] = None) -> np.ndarray:
        """
        应用CLAHE（对比度受限的自适应直方图均衡化）

        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: 输入灰度图像
            clip_limit: CLAHE对比度限制，None则使用配置值

        Returns:
            增强后的图像
        """
        if clip_limit is None:
            clip_limit = self.config.clahe_clip_limit

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        enhanced = clahe.apply(image)
        return enhanced

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        二值化图像

        Binarize the image using Otsu's method.

        Args:
            image: 输入灰度图像

        Returns:
            二值化图像
        """
        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
        return binary

    def isolate_document(self, image: np.ndarray) -> np.ndarray:
        """
        分离文档区域（用于轮廓提取）

        Isolate document region for contour extraction.

        Args:
            image: 输入灰度图像

        Returns:
            处理后的二值图像
        """
        element = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.morph_kernel_size
        )

        # 先膨胀后腐蚀
        dilated = cv2.dilate(
            image,
            element,
            iterations=self.config.dilate_iterations
        )
        eroded = cv2.erode(
            dilated,
            element,
            iterations=self.config.erode_iterations
        )

        # 二值化
        binary = self.binarize(eroded)
        return binary

    def enhance_text_contours(self, image: np.ndarray) -> np.ndarray:
        """
        增强文本轮廓

        Enhance text contours.

        Args:
            image: 输入灰度图像

        Returns:
            处理后的二值图像
        """
        element = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.text_kernel_size
        )

        # 先腐蚀后膨胀
        eroded = cv2.erode(
            image,
            element,
            iterations=self.config.text_erode_iterations
        )
        dilated = cv2.dilate(
            eroded,
            element,
            iterations=self.config.text_dilate_iterations
        )

        # 二值化
        binary = self.binarize(dilated)
        return binary

    def preprocess_for_dewarp(
        self,
        image: np.ndarray,
        resize_to: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        为去畸变准备图像

        Preprocess image for dewarping.

        Args:
            image: 输入RGB图像
            resize_to: 可选的目标尺寸 (width, height)

        Returns:
            (增强图像, 二值图像) 的元组
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 锐化
        sharpened = self.sharpen(gray)

        # 调整大小
        if resize_to is not None:
            gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_CUBIC)
            sharpened = cv2.resize(sharpened, resize_to, interpolation=cv2.INTER_CUBIC)

        # 增强对比度
        enhanced = self.apply_clahe(gray, self.config.clahe_clip_limit)

        # 二值化
        binary = self.binarize(enhanced)

        return enhanced, binary

    def enhance_regions(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        regions: list
    ) -> np.ndarray:
        """
        增强图像中的特定区域

        Enhance specific regions in the image.

        Args:
            image: 原始图像
            binary: 二值图像
            regions: 区域列表，每个区域为 [x, y, w, h]

        Returns:
            增强后的二值图像
        """
        result = binary.copy()

        for box in regions:
            x, y, w, h = box
            # 提取区域
            region = image[y:y+h, x:x+w]
            # 高对比度增强
            enhanced_region = self.apply_clahe(region, self.config.high_contrast_limit)
            # 替换到结果图像
            result[y:y+h, x:x+w] = enhanced_region

        return result
