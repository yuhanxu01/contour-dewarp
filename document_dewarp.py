"""
文档去畸变主模块 - 整合所有功能的主类

Document dewarp main module integrating all functionalities.
"""

import cv2
import numpy as np
from typing import Optional, Union, List, Tuple
from pathlib import Path

from config import DewarpConfig, DEFAULT_CONFIG
from image_processor import ImageProcessor
from contour_extractor import ContourExtractor
from surface_mesh import SurfaceMesh
from dewarp_transformer import DewarpTransformer


class DocumentDewarp:
    """文档去畸变主类

    Main class for document dewarping, supporting both single and double page modes,
    PDF and image inputs.

    Examples:
        >>> # 处理单张图片
        >>> dewarp = DocumentDewarp()
        >>> result = dewarp.process_image('input.jpg')
        >>> cv2.imwrite('output.jpg', result)
        >>>
        >>> # 处理PDF
        >>> dewarp = DocumentDewarp()
        >>> dewarp.process_pdf('input.pdf', 'output.pdf')
        >>>
        >>> # 单页模式
        >>> from config import SINGLE_PAGE_CONFIG
        >>> dewarp = DocumentDewarp(config=SINGLE_PAGE_CONFIG)
        >>> result = dewarp.process_image('single_page.jpg')
    """

    def __init__(self, config: Optional[DewarpConfig] = None):
        """
        初始化文档去畸变器

        Args:
            config: 去畸变配置对象，None则使用默认配置
        """
        self.config = config or DEFAULT_CONFIG

        # 初始化各个模块
        self.image_processor = ImageProcessor(self.config)
        self.contour_extractor = ContourExtractor(self.config)
        self.surface_mesh = SurfaceMesh(self.config)
        self.transformer = DewarpTransformer(self.config)

    def process_image(
        self,
        image: Union[str, np.ndarray],
        return_binary: bool = True
    ) -> np.ndarray:
        """
        处理单张图片

        Process a single image.

        Args:
            image: 图片路径或NumPy数组
            return_binary: 是否返回二值图像

        Returns:
            去畸变后的图像
        """
        # 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # 锐化
        sharpened = self.image_processor.sharpen(gray)

        # 查找文档区域
        doc_region = self._find_document_region(sharpened)

        # 调整大小
        resized = cv2.resize(
            doc_region,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_CUBIC
        )

        # 提取轮廓并处理
        dewarped = self._dewarp_image(resized)

        if return_binary:
            return self.image_processor.binarize(dewarped)
        else:
            return dewarped

    def _find_document_region(self, image: np.ndarray) -> np.ndarray:
        """
        查找文档区域

        Find document region in image.

        Args:
            image: 输入灰度图像

        Returns:
            文档区域图像
        """
        # 分离文档
        isolated = self.image_processor.isolate_document(image)

        # 查找轮廓
        contours = self.contour_extractor.find_contours(isolated)

        if not contours:
            return image

        # 获取最大轮廓
        largest_contour = self.contour_extractor.get_largest_contour(contours)

        # 应用全局透视校正
        cropped = self.transformer.apply_global_perspective_correction(
            image,
            largest_contour
        )

        return cropped

    def _dewarp_image(self, image: np.ndarray) -> np.ndarray:
        """
        对图像进行去畸变处理

        Dewarp the image.

        Args:
            image: 输入图像

        Returns:
            去畸变后的图像
        """
        # 分离文档轮廓
        isolated = self.image_processor.isolate_document(image)

        # 查找轮廓
        contours = self.contour_extractor.find_contours(isolated)

        if not contours:
            # 如果没有找到轮廓，返回增强后的图像
            enhanced = self.image_processor.apply_clahe(image)
            return self.image_processor.binarize(enhanced)

        # 获取最大轮廓
        largest_contour = self.contour_extractor.get_largest_contour(contours)

        # 提取边界曲线
        upper, lower = self.contour_extractor.extract_boundary_curves(
            largest_contour,
            image.shape
        )

        # 根据页面模式处理
        if self.config.page_mode == 'double':
            mesh_blocks = self._process_double_page(upper, lower)
        else:
            mesh_blocks = self._process_single_page(upper, lower)

        # 增强并变换
        enhanced = self.image_processor.apply_clahe(image)
        dewarped = self.transformer.dewarp_curved_surface(enhanced, mesh_blocks)

        return dewarped

    def _process_double_page(
        self,
        upper: List,
        lower: List
    ) -> np.ndarray:
        """
        处理双页模式

        Process double page mode.

        Args:
            upper: 上边界点列表
            lower: 下边界点列表

        Returns:
            网格块数组
        """
        # 找到书脊位置
        upper_mid_x, lower_mid_x = self.contour_extractor.split_double_page(
            upper,
            lower
        )

        # 使用平均值作为分割线
        x_mid = int((upper_mid_x + lower_mid_x) / 2)

        # 对上下边界进行平滑拟合
        upper_points, upper_left_coeffs, upper_right_coeffs = \
            self.contour_extractor.fit_smooth_curve(upper[1:-1], x_mid)

        lower_points, lower_left_coeffs, lower_right_coeffs = \
            self.contour_extractor.fit_smooth_curve(lower[1:-1], x_mid)

        # 重建完整的上下边界
        upper_full = [upper[0]] + upper_points + [upper[-1]]
        lower_full = [lower[0]] + lower_points + [lower[-1]]

        # 生成网格
        mesh_blocks = self.surface_mesh.generate_mesh_grid(
            upper_full,
            lower_full,
            upper_left_coeffs,
            upper_right_coeffs,
            lower_left_coeffs,
            lower_right_coeffs,
            x_mid
        )

        return mesh_blocks

    def _process_single_page(
        self,
        upper: List,
        lower: List
    ) -> np.ndarray:
        """
        处理单页模式

        Process single page mode.

        Args:
            upper: 上边界点列表
            lower: 下边界点列表

        Returns:
            网格块数组
        """
        # 简单模式：直接生成均匀网格
        mesh_blocks = self.surface_mesh.generate_simple_grid(upper, lower)

        return mesh_blocks

    def process_pdf(
        self,
        pdf_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        处理PDF文件

        Process PDF file.

        Args:
            pdf_path: 输入PDF路径
            output_path: 输出PDF路径
            progress_callback: 可选的进度回调函数
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import os
        except ImportError:
            raise ImportError(
                "PDF processing requires PyMuPDF and PIL. "
                "Install with: pip install PyMuPDF Pillow"
            )

        # 打开输入PDF
        pdf_input = fitz.open(pdf_path)
        pdf_output = fitz.open()

        total_pages = len(pdf_input)

        for page_num, page in enumerate(pdf_input):
            # 转换为图像
            pix = page.get_pixmap(matrix=fitz.Matrix(
                self.config.pdf_zoom,
                self.config.pdf_zoom
            ))
            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            # 处理图像
            img_array = np.array(img)
            processed = self.process_image(img_array, return_binary=True)

            # 保存为临时图片
            temp_path = f"temp_page_{page_num}.png"
            Image.fromarray(processed).save(temp_path)

            # 转换回PDF
            img_doc = fitz.open(temp_path)
            rect = img_doc[0].rect
            pdf_bytes = img_doc.convert_to_pdf()
            img_doc.close()
            os.remove(temp_path)

            # 添加到输出PDF
            img_pdf = fitz.open("pdf", pdf_bytes)
            new_page = pdf_output.new_page(
                width=rect.width,
                height=rect.height
            )
            new_page.show_pdf_page(rect, img_pdf, 0)

            # 调用进度回调
            if progress_callback:
                progress_callback(page_num + 1, total_pages)

        # 保存输出PDF
        pdf_output.save(output_path)
        pdf_output.close()
        pdf_input.close()

    def process_images_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        output_format: str = 'png'
    ) -> List[str]:
        """
        批量处理图片

        Batch process images.

        Args:
            image_paths: 输入图片路径列表
            output_dir: 输出目录
            output_format: 输出格式

        Returns:
            输出文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for img_path in image_paths:
            # 处理图像
            result = self.process_image(img_path)

            # 生成输出路径
            input_name = Path(img_path).stem
            output_path = output_dir / f"{input_name}_dewarped.{output_format}"

            # 保存
            cv2.imwrite(str(output_path), result)
            output_paths.append(str(output_path))

        return output_paths
