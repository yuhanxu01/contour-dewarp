"""
Texture Renderer - Text and Image Content Generation and Mapping

This module generates realistic document textures (text, images, diagrams)
and maps them onto curved 3D page surfaces.

Author: Yuhan Xu
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ContentType(Enum):
    """Types of document content"""
    TEXT_ONLY = "text_only"
    TEXT_WITH_IMAGES = "text_with_images"
    MIXED_LAYOUT = "mixed_layout"
    HANDWRITTEN = "handwritten"
    TECHNICAL = "technical"  # Diagrams, formulas
    # 新增练习本类型
    NOTEBOOK_GRID = "notebook_grid"           # 网格本（方格纸）
    NOTEBOOK_LINED = "notebook_lined"         # 横线本
    NOTEBOOK_GRID_TEXT = "notebook_grid_text"   # 网格本带文字
    NOTEBOOK_LINED_TEXT = "notebook_lined_text" # 横线本带文字


class PaperStyle(Enum):
    """纸张样式"""
    PLAIN = "plain"                 # 普通白纸
    GRID_SMALL = "grid_small"       # 小方格（5mm）
    GRID_MEDIUM = "grid_medium"     # 中方格（7mm）
    GRID_LARGE = "grid_large"       # 大方格（10mm）
    LINED_NARROW = "lined_narrow"   # 窄行（6mm）
    LINED_MEDIUM = "lined_medium"   # 中行（8mm）
    LINED_WIDE = "lined_wide"       # 宽行（10mm）
    DOT_GRID = "dot_grid"           # 点阵格
    GRAPH = "graph"                 # 坐标纸（带粗线）


@dataclass
class TextBlock:
    """Represents a block of text on the page"""
    x: int          # Top-left x position
    y: int          # Top-left y position
    width: int      # Block width
    height: int     # Block height
    text: str       # Text content
    font_size: int  # Font size


@dataclass
class ImageBlock:
    """Represents an image/figure on the page"""
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray


class TextureRenderer:
    """
    Generates and renders document textures onto curved surfaces.

    The renderer creates realistic document content including text,
    images, and diagrams, then maps them onto 3D curved page surfaces.
    """

    # Sample text for document generation (multilingual support)
    SAMPLE_TEXTS = {
        'english': [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "In the beginning was the Word, and the Word was with God.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step.",
            "Knowledge is power, but enthusiasm pulls the switch.",
            "The only thing we have to fear is fear itself.",
        ],
        'chinese': [
            "天行健，君子以自强不息。",
            "地势坤，君子以厚德载物。",
            "人生得意须尽欢，莫使金樽空对月。",
            "床前明月光，疑是地上霜。",
            "举头望明月，低头思故乡。",
            "春眠不觉晓，处处闻啼鸟。",
            "白日依山尽，黄河入海流。",
            "欲穷千里目，更上一层楼。",
        ]
    }

    def __init__(
        self,
        texture_size: Tuple[int, int] = (1024, 1024),
        background_color: Tuple[int, int, int] = (255, 255, 250),  # Slightly off-white
        text_color: Tuple[int, int, int] = (20, 20, 20),
        random_seed: Optional[int] = None
    ):
        """
        Initialize the texture renderer.

        Args:
            texture_size: Size of the texture (height, width)
            background_color: Background color (RGB)
            text_color: Default text color (RGB)
            random_seed: Random seed for reproducibility
        """
        self.texture_size = texture_size
        self.background_color = background_color
        self.text_color = text_color
        self.rng = np.random.default_rng(random_seed)

    def generate_flat_texture(
        self,
        content_type: ContentType = ContentType.TEXT_ONLY,
        language: str = 'english',
        add_noise: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a flat 2D document texture.

        Args:
            content_type: Type of content to generate
            language: Language for text content
            add_noise: Whether to add paper texture noise
            **kwargs: Additional parameters

        Returns:
            Texture image as numpy array (H, W, 3)
        """
        h, w = self.texture_size

        # Create base background
        texture = np.full((h, w, 3), self.background_color, dtype=np.uint8)

        # Add paper texture
        if add_noise:
            texture = self._add_paper_texture(texture)

        # Generate content based on type
        if content_type == ContentType.TEXT_ONLY:
            texture = self._add_text_content(texture, language, **kwargs)
        elif content_type == ContentType.TEXT_WITH_IMAGES:
            texture = self._add_text_and_images(texture, language, **kwargs)
        elif content_type == ContentType.MIXED_LAYOUT:
            texture = self._add_mixed_layout(texture, language, **kwargs)
        elif content_type == ContentType.TECHNICAL:
            texture = self._add_technical_content(texture, **kwargs)
        elif content_type == ContentType.HANDWRITTEN:
            texture = self._add_handwritten_style(texture, language, **kwargs)
        # 新增练习本类型
        elif content_type == ContentType.NOTEBOOK_GRID:
            texture = self._add_grid_lines(texture, **kwargs)
        elif content_type == ContentType.NOTEBOOK_LINED:
            texture = self._add_horizontal_lines(texture, **kwargs)
        elif content_type == ContentType.NOTEBOOK_GRID_TEXT:
            texture = self._add_grid_lines(texture, **kwargs)
            texture = self._add_handwritten_style(texture, language, **kwargs)
        elif content_type == ContentType.NOTEBOOK_LINED_TEXT:
            texture = self._add_horizontal_lines(texture, **kwargs)
            texture = self._add_handwritten_style(texture, language, **kwargs)

        return texture

    def _add_paper_texture(self, texture: np.ndarray) -> np.ndarray:
        """Add realistic paper texture (fibers, slight discoloration)"""
        h, w, _ = texture.shape

        # Add subtle noise for paper grain
        noise = self.rng.normal(0, 3, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 1)

        # Add to all channels
        for i in range(3):
            texture[:, :, i] = np.clip(
                texture[:, :, i].astype(np.float32) + noise,
                0, 255
            ).astype(np.uint8)

        # Add subtle color variation (aged paper effect)
        if self.rng.random() > 0.5:
            age_factor = self.rng.uniform(0.95, 1.0)
            # Slight yellowing
            texture[:, :, 2] = np.clip(texture[:, :, 2] * age_factor, 0, 255).astype(np.uint8)
            texture[:, :, 0] = np.clip(texture[:, :, 0] * (age_factor * 0.98), 0, 255).astype(np.uint8)

        return texture

    def _add_grid_lines(
        self,
        texture: np.ndarray,
        paper_style: Optional[PaperStyle] = None,
        line_color: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        添加网格线（方格纸效果）。

        Args:
            texture: 输入纹理
            paper_style: 纸张样式（决定网格大小）
            line_color: 线条颜色
        """
        h, w, _ = texture.shape

        # 随机选择样式
        if paper_style is None:
            paper_style = self.rng.choice([
                PaperStyle.GRID_SMALL,
                PaperStyle.GRID_MEDIUM,
                PaperStyle.GRID_LARGE,
                PaperStyle.DOT_GRID,
                PaperStyle.GRAPH,
            ])

        # 根据样式确定网格间距（像素）
        # 假设纹理代表约 200mm x 280mm 的纸张
        scale = h / 280.0  # pixels per mm

        if paper_style == PaperStyle.GRID_SMALL:
            spacing = int(5 * scale)  # 5mm 方格
        elif paper_style == PaperStyle.GRID_MEDIUM:
            spacing = int(7 * scale)  # 7mm 方格
        elif paper_style == PaperStyle.GRID_LARGE:
            spacing = int(10 * scale)  # 10mm 方格
        elif paper_style == PaperStyle.DOT_GRID:
            spacing = int(5 * scale)  # 5mm 点阵
        elif paper_style == PaperStyle.GRAPH:
            spacing = int(5 * scale)  # 5mm 小格
        else:
            spacing = int(7 * scale)

        spacing = max(8, spacing)  # 最小8像素间距

        # 线条颜色 - 练习本通常是浅蓝色或浅灰色
        if line_color is None:
            color_choice = self.rng.choice(['blue', 'gray', 'green'])
            if color_choice == 'blue':
                line_color = (220, 200, 180)  # BGR - 浅蓝色
            elif color_choice == 'gray':
                line_color = (200, 200, 200)  # 浅灰色
            else:
                line_color = (200, 220, 200)  # 浅绿色

        # 边距
        margin = kwargs.get('margin', int(20 * scale))

        if paper_style == PaperStyle.DOT_GRID:
            # 点阵网格
            dot_radius = max(1, int(scale * 0.3))
            for y in range(margin, h - margin, spacing):
                for x in range(margin, w - margin, spacing):
                    cv2.circle(texture, (x, y), dot_radius, line_color, -1)

        elif paper_style == PaperStyle.GRAPH:
            # 坐标纸 - 细线和粗线组合
            thin_color = line_color
            # 粗线颜色更深
            thick_color = tuple(max(0, c - 40) for c in line_color)

            major_spacing = spacing * 5  # 每5个小格一条粗线

            # 画细线
            for y in range(margin, h - margin, spacing):
                thickness = 2 if (y - margin) % major_spacing == 0 else 1
                color = thick_color if thickness == 2 else thin_color
                cv2.line(texture, (margin, y), (w - margin, y), color, thickness)

            for x in range(margin, w - margin, spacing):
                thickness = 2 if (x - margin) % major_spacing == 0 else 1
                color = thick_color if thickness == 2 else thin_color
                cv2.line(texture, (x, margin), (x, h - margin), color, thickness)

        else:
            # 普通网格线
            line_thickness = kwargs.get('line_thickness', 1)

            # 水平线
            for y in range(margin, h - margin, spacing):
                cv2.line(texture, (margin, y), (w - margin, y), line_color, line_thickness)

            # 垂直线
            for x in range(margin, w - margin, spacing):
                cv2.line(texture, (x, margin), (x, h - margin), line_color, line_thickness)

        # 可选：添加边框
        if kwargs.get('add_border', self.rng.random() > 0.7):
            border_color = tuple(max(0, c - 30) for c in line_color)
            cv2.rectangle(texture, (margin, margin), (w - margin, h - margin), border_color, 2)

        return texture

    def _add_horizontal_lines(
        self,
        texture: np.ndarray,
        paper_style: Optional[PaperStyle] = None,
        line_color: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        添加水平横线（横线本效果）。

        Args:
            texture: 输入纹理
            paper_style: 纸张样式（决定行距）
            line_color: 线条颜色
        """
        h, w, _ = texture.shape

        # 随机选择样式
        if paper_style is None:
            paper_style = self.rng.choice([
                PaperStyle.LINED_NARROW,
                PaperStyle.LINED_MEDIUM,
                PaperStyle.LINED_WIDE,
            ])

        # 根据样式确定行距
        scale = h / 280.0  # pixels per mm

        if paper_style == PaperStyle.LINED_NARROW:
            spacing = int(6 * scale)  # 6mm 行距
        elif paper_style == PaperStyle.LINED_MEDIUM:
            spacing = int(8 * scale)  # 8mm 行距
        elif paper_style == PaperStyle.LINED_WIDE:
            spacing = int(10 * scale)  # 10mm 行距
        else:
            spacing = int(8 * scale)

        spacing = max(12, spacing)  # 最小12像素行距

        # 线条颜色
        if line_color is None:
            color_choice = self.rng.choice(['blue', 'gray', 'black'])
            if color_choice == 'blue':
                line_color = (220, 190, 170)  # BGR - 浅蓝色
            elif color_choice == 'gray':
                line_color = (190, 190, 190)  # 浅灰色
            else:
                line_color = (180, 180, 180)  # 更深的灰色

        # 边距
        margin_left = kwargs.get('margin_left', int(25 * scale))
        margin_right = kwargs.get('margin_right', int(15 * scale))
        margin_top = kwargs.get('margin_top', int(30 * scale))
        margin_bottom = kwargs.get('margin_bottom', int(20 * scale))

        line_thickness = kwargs.get('line_thickness', 1)

        # 画横线
        for y in range(margin_top, h - margin_bottom, spacing):
            cv2.line(texture, (margin_left, y), (w - margin_right, y), line_color, line_thickness)

        # 可选：添加左侧红色竖线（像作文本）
        if kwargs.get('add_margin_line', self.rng.random() > 0.5):
            margin_line_x = margin_left - int(5 * scale)
            margin_line_color = (150, 150, 220)  # BGR - 浅红色
            cv2.line(texture, (margin_line_x, margin_top - int(5 * scale)),
                    (margin_line_x, h - margin_bottom + int(5 * scale)), margin_line_color, 1)

        # 可选：添加顶部标题线（双横线）
        if kwargs.get('add_header_line', self.rng.random() > 0.7):
            header_y = margin_top - int(15 * scale)
            if header_y > 10:
                header_color = tuple(max(0, c - 20) for c in line_color)
                cv2.line(texture, (margin_left, header_y), (w - margin_right, header_y), header_color, 1)
                cv2.line(texture, (margin_left, header_y + 3), (w - margin_right, header_y + 3), header_color, 1)

        # 可选：添加装订孔标记
        if kwargs.get('add_punch_holes', self.rng.random() > 0.8):
            hole_x = int(15 * scale)
            hole_positions = [h // 4, h // 2, 3 * h // 4]
            hole_color = (210, 210, 210)
            for hole_y in hole_positions:
                cv2.circle(texture, (hole_x, hole_y), int(3 * scale), hole_color, 2)

        return texture

    def _add_text_content(
        self,
        texture: np.ndarray,
        language: str,
        num_lines: Optional[int] = None,
        font_size_range: Tuple[int, int] = (16, 24),
        line_spacing: float = 1.5,
        margins: Tuple[int, int, int, int] = (50, 50, 50, 50),  # top, right, bottom, left
        **kwargs
    ) -> np.ndarray:
        """Add text content to the texture"""
        h, w, _ = texture.shape
        margin_top, margin_right, margin_bottom, margin_left = margins

        # Available text area
        text_width = w - margin_left - margin_right
        text_height = h - margin_top - margin_bottom

        # Select font size
        font_size = self.rng.integers(*font_size_range)
        line_height = int(font_size * line_spacing)

        # Calculate number of lines that fit
        max_lines = text_height // line_height
        if num_lines is None:
            num_lines = self.rng.integers(max_lines // 2, max_lines)
        num_lines = min(num_lines, max_lines)

        # Get sample texts
        texts = self.SAMPLE_TEXTS.get(language, self.SAMPLE_TEXTS['english'])

        # Draw text lines
        y = margin_top
        for i in range(num_lines):
            text = self.rng.choice(texts)

            # Draw text using OpenCV (fallback when PIL not available)
            self._draw_text_line(
                texture, text,
                x=margin_left,
                y=y + font_size,
                font_size=font_size,
                color=self.text_color
            )

            y += line_height

            if y + font_size > h - margin_bottom:
                break

        return texture

    def _draw_text_line(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_size: int,
        color: Tuple[int, int, int]
    ):
        """Draw a line of text on the image"""
        if HAS_PIL:
            self._draw_text_pil(image, text, x, y, font_size, color)
        else:
            self._draw_text_cv2(image, text, x, y, font_size, color)

    def _draw_text_pil(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_size: int,
        color: Tuple[int, int, int]
    ):
        """Draw text using PIL (better font support)"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            # Try to use a system font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        draw.text((x, y - font_size), text, font=font, fill=color)

        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        image[:] = result

    def _draw_text_cv2(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_size: int,
        color: Tuple[int, int, int]
    ):
        """Draw text using OpenCV (fallback)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = font_size / 30  # Approximate scale
        thickness = max(1, font_size // 15)

        # OpenCV uses BGR
        cv2.putText(image, text, (x, y), font, scale, color[::-1], thickness, cv2.LINE_AA)

    def _add_text_and_images(
        self,
        texture: np.ndarray,
        language: str,
        num_images: int = 2,
        **kwargs
    ) -> np.ndarray:
        """Add text content with embedded images"""
        h, w, _ = texture.shape

        # First add text with gaps for images
        texture = self._add_text_content(
            texture, language,
            margins=(50, 50, h // 3, 50),  # Leave bottom third for image
            **kwargs
        )

        # Add synthetic images
        for i in range(num_images):
            img_w = self.rng.integers(w // 4, w // 2)
            img_h = self.rng.integers(h // 6, h // 4)
            img_x = self.rng.integers(50, w - img_w - 50)
            img_y = self.rng.integers(h // 2, h - img_h - 50)

            # Generate synthetic image content
            synth_img = self._generate_synthetic_image(img_w, img_h)

            # Add border
            synth_img = cv2.copyMakeBorder(
                synth_img, 2, 2, 2, 2,
                cv2.BORDER_CONSTANT, value=(100, 100, 100)
            )

            # Blend into texture
            img_h_with_border = synth_img.shape[0]
            img_w_with_border = synth_img.shape[1]

            if img_y + img_h_with_border < h and img_x + img_w_with_border < w:
                texture[img_y:img_y + img_h_with_border,
                        img_x:img_x + img_w_with_border] = synth_img

        return texture

    def _generate_synthetic_image(self, width: int, height: int) -> np.ndarray:
        """Generate a synthetic placeholder image"""
        img = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)

        # Add some geometric shapes
        num_shapes = self.rng.integers(3, 8)

        for _ in range(num_shapes):
            shape_type = self.rng.choice(['rect', 'circle', 'line'])
            color = tuple(self.rng.integers(50, 200, 3).tolist())

            if shape_type == 'rect':
                x1 = self.rng.integers(0, width - 20)
                y1 = self.rng.integers(0, height - 20)
                x2 = self.rng.integers(x1 + 10, min(width, x1 + 100))
                y2 = self.rng.integers(y1 + 10, min(height, y1 + 100))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            elif shape_type == 'circle':
                cx = self.rng.integers(10, width - 10)
                cy = self.rng.integers(10, height - 10)
                r = self.rng.integers(5, min(30, min(width, height) // 4))
                cv2.circle(img, (cx, cy), r, color, -1)
            else:  # line
                x1 = self.rng.integers(0, width)
                y1 = self.rng.integers(0, height)
                x2 = self.rng.integers(0, width)
                y2 = self.rng.integers(0, height)
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

        return img

    def _add_mixed_layout(
        self,
        texture: np.ndarray,
        language: str,
        **kwargs
    ) -> np.ndarray:
        """Add mixed layout with columns and varied content"""
        h, w, _ = texture.shape

        # Decide on layout type
        layout_type = self.rng.choice(['two_column', 'sidebar', 'header_body'])

        if layout_type == 'two_column':
            # Left column
            left_region = texture[:, :w // 2 - 20].copy()
            left_region = self._add_text_content(
                left_region, language,
                margins=(50, 20, 50, 50),
                font_size_range=(12, 18)
            )
            texture[:, :w // 2 - 20] = left_region

            # Right column
            right_region = texture[:, w // 2 + 20:].copy()
            right_region = self._add_text_content(
                right_region, language,
                margins=(50, 50, 50, 20),
                font_size_range=(12, 18)
            )
            texture[:, w // 2 + 20:] = right_region

            # Draw separator line
            cv2.line(texture, (w // 2, 50), (w // 2, h - 50), (180, 180, 180), 1)

        elif layout_type == 'sidebar':
            # Main content (70%)
            main_w = int(w * 0.7)
            main_region = texture[:, :main_w - 20].copy()
            main_region = self._add_text_content(
                main_region, language,
                margins=(50, 20, 50, 50)
            )
            texture[:, :main_w - 20] = main_region

            # Sidebar with image
            sidebar = texture[:, main_w:].copy()
            sidebar_img = self._generate_synthetic_image(
                sidebar.shape[1] - 40,
                sidebar.shape[0] // 3
            )
            y_start = 50
            x_start = 20
            sidebar[y_start:y_start + sidebar_img.shape[0],
                    x_start:x_start + sidebar_img.shape[1]] = sidebar_img
            texture[:, main_w:] = sidebar

        else:  # header_body
            # Header
            header_h = h // 6
            header = texture[:header_h, :].copy()
            self._draw_text_line(
                header,
                "Chapter Title" if language == 'english' else "章节标题",
                x=w // 4, y=header_h // 2,
                font_size=32,
                color=self.text_color
            )
            texture[:header_h, :] = header

            # Separator
            cv2.line(texture, (50, header_h + 10), (w - 50, header_h + 10), (150, 150, 150), 2)

            # Body
            body_region = texture[header_h + 30:, :].copy()
            body_region = self._add_text_content(
                body_region, language,
                margins=(20, 50, 50, 50)
            )
            texture[header_h + 30:, :] = body_region

        return texture

    def _add_technical_content(
        self,
        texture: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Add technical content like diagrams and formulas"""
        h, w, _ = texture.shape

        # Add some text
        texture = self._add_text_content(
            texture, 'english',
            margins=(50, 50, h // 2, 50),
            num_lines=10
        )

        # Add a simple diagram
        diagram_x = w // 4
        diagram_y = h // 2
        diagram_w = w // 2
        diagram_h = h // 3

        # Draw diagram background
        cv2.rectangle(
            texture,
            (diagram_x, diagram_y),
            (diagram_x + diagram_w, diagram_y + diagram_h),
            (245, 245, 245), -1
        )
        cv2.rectangle(
            texture,
            (diagram_x, diagram_y),
            (diagram_x + diagram_w, diagram_y + diagram_h),
            (100, 100, 100), 2
        )

        # Add some boxes and arrows (flowchart-like)
        box_w = diagram_w // 4
        box_h = diagram_h // 4

        positions = [
            (diagram_x + diagram_w // 2 - box_w // 2, diagram_y + 20),  # Top center
            (diagram_x + 30, diagram_y + diagram_h // 2),  # Left
            (diagram_x + diagram_w - box_w - 30, diagram_y + diagram_h // 2),  # Right
            (diagram_x + diagram_w // 2 - box_w // 2, diagram_y + diagram_h - box_h - 20),  # Bottom
        ]

        for i, (bx, by) in enumerate(positions):
            cv2.rectangle(texture, (bx, by), (bx + box_w, by + box_h), (180, 180, 220), -1)
            cv2.rectangle(texture, (bx, by), (bx + box_w, by + box_h), (80, 80, 120), 2)

        # Draw connecting arrows
        cv2.arrowedLine(
            texture,
            (positions[0][0] + box_w // 2, positions[0][1] + box_h),
            (positions[1][0] + box_w, positions[1][1] + box_h // 2),
            (60, 60, 60), 2, tipLength=0.1
        )
        cv2.arrowedLine(
            texture,
            (positions[0][0] + box_w // 2, positions[0][1] + box_h),
            (positions[2][0], positions[2][1] + box_h // 2),
            (60, 60, 60), 2, tipLength=0.1
        )

        return texture

    def _add_handwritten_style(
        self,
        texture: np.ndarray,
        language: str,
        **kwargs
    ) -> np.ndarray:
        """Add handwritten-style content (simulated)"""
        h, w, _ = texture.shape

        # Use a more irregular baseline
        num_lines = self.rng.integers(15, 25)
        y = 60
        line_height = (h - 120) // num_lines

        texts = self.SAMPLE_TEXTS.get(language, self.SAMPLE_TEXTS['english'])

        for i in range(num_lines):
            text = self.rng.choice(texts)

            # Add baseline variation (handwriting isn't perfectly straight)
            x_offset = self.rng.integers(-5, 15)
            y_offset = self.rng.integers(-3, 3)

            # Slight font size variation
            font_size = 18 + self.rng.integers(-2, 3)

            # Slight color variation (pen pressure)
            color_var = self.rng.integers(-10, 10)
            color = tuple(max(0, min(255, c + color_var)) for c in self.text_color)

            self._draw_text_line(
                texture, text,
                x=60 + x_offset,
                y=y + y_offset,
                font_size=font_size,
                color=color
            )

            y += line_height + self.rng.integers(-2, 5)

        return texture

    def map_texture_to_surface(
        self,
        texture: np.ndarray,
        vertices: np.ndarray,
        uv_coords: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Map a 2D texture onto a 3D surface using UV coordinates.

        This creates the warped appearance of the texture following the curved surface.

        Args:
            texture: The 2D texture image (H_tex, W_tex, 3)
            vertices: 3D vertex positions (H_mesh, W_mesh, 3)
            uv_coords: UV texture coordinates (H_mesh, W_mesh, 2)
            output_size: Output image size (height, width)

        Returns:
            Warped texture image (output_H, output_W, 3)
        """
        h_mesh, w_mesh, _ = vertices.shape
        h_out, w_out = output_size
        h_tex, w_tex, _ = texture.shape

        # Project 3D vertices to 2D screen space
        # Using orthographic projection for simplicity
        # (Can be extended to perspective projection)
        screen_coords = self._project_vertices(vertices, output_size)

        # Create output image
        output = np.zeros((h_out, w_out, 3), dtype=np.uint8)

        # For each quad in the mesh, warp the texture
        for i in range(h_mesh - 1):
            for j in range(w_mesh - 1):
                # Get quad corners in screen space
                src_pts = np.array([
                    [uv_coords[i, j, 0] * w_tex, uv_coords[i, j, 1] * h_tex],
                    [uv_coords[i, j + 1, 0] * w_tex, uv_coords[i, j + 1, 1] * h_tex],
                    [uv_coords[i + 1, j + 1, 0] * w_tex, uv_coords[i + 1, j + 1, 1] * h_tex],
                    [uv_coords[i + 1, j, 0] * w_tex, uv_coords[i + 1, j, 1] * h_tex],
                ], dtype=np.float32)

                dst_pts = np.array([
                    screen_coords[i, j],
                    screen_coords[i, j + 1],
                    screen_coords[i + 1, j + 1],
                    screen_coords[i + 1, j],
                ], dtype=np.float32)

                # Compute perspective transform
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # Create mask for this quad
                mask = np.zeros((h_out, w_out), dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

                # Warp texture
                warped = cv2.warpPerspective(texture, M, (w_out, h_out))

                # Blend into output
                mask_3ch = np.stack([mask, mask, mask], axis=-1)
                output = np.where(mask_3ch > 0, warped, output)

        return output

    def _project_vertices(
        self,
        vertices: np.ndarray,
        output_size: Tuple[int, int],
        camera_distance: float = 500.0,
        use_perspective: bool = True
    ) -> np.ndarray:
        """
        Project 3D vertices to 2D screen coordinates.

        Args:
            vertices: 3D vertex positions (H, W, 3)
            output_size: Output image size (height, width)
            camera_distance: Distance from camera (for perspective)
            use_perspective: Use perspective projection (vs orthographic)

        Returns:
            Screen coordinates (H, W, 2)
        """
        h_mesh, w_mesh, _ = vertices.shape
        h_out, w_out = output_size

        # Get bounding box of vertices
        x_min, x_max = vertices[:, :, 0].min(), vertices[:, :, 0].max()
        y_min, y_max = vertices[:, :, 1].min(), vertices[:, :, 1].max()
        z_min, z_max = vertices[:, :, 2].min(), vertices[:, :, 2].max()

        # Normalize to output size
        screen_coords = np.zeros((h_mesh, w_mesh, 2), dtype=np.float32)

        for i in range(h_mesh):
            for j in range(w_mesh):
                x, y, z = vertices[i, j]

                if use_perspective and z_max > z_min:
                    # Perspective projection
                    z_normalized = (z - z_min) / (z_max - z_min + 1e-8)
                    scale = 1.0 / (1.0 + z_normalized * 0.1)  # Subtle perspective
                else:
                    scale = 1.0

                # Map to screen space
                screen_x = (x - x_min) / (x_max - x_min + 1e-8) * (w_out - 1) * scale
                screen_y = (y - y_min) / (y_max - y_min + 1e-8) * (h_out - 1) * scale

                # Center adjustment for perspective
                if use_perspective:
                    cx, cy = w_out / 2, h_out / 2
                    screen_x = cx + (screen_x - cx) * scale
                    screen_y = cy + (screen_y - cy) * scale

                screen_coords[i, j] = [screen_x, screen_y]

        return screen_coords

    def generate_random_texture(self, is_double_page: bool = False) -> np.ndarray:
        """Generate a texture with random content type and parameters"""
        if is_double_page:
            # Double the width for a book spread
            orig_size = self.texture_size
            self.texture_size = (orig_size[0], orig_size[1] * 2)

        # 原有的文档类型
        document_types = [
            ContentType.TEXT_ONLY,
            ContentType.TEXT_WITH_IMAGES,
            ContentType.MIXED_LAYOUT,
            ContentType.TECHNICAL,
        ]

        # 新增的练习本类型
        notebook_types = [
            ContentType.NOTEBOOK_GRID,
            ContentType.NOTEBOOK_LINED,
            ContentType.NOTEBOOK_GRID_TEXT,
            ContentType.NOTEBOOK_LINED_TEXT,
            ContentType.HANDWRITTEN,
        ]

        # 40%概率选择练习本类型，60%概率选择普通文档类型
        if self.rng.random() < 0.4:
            content_type = self.rng.choice(notebook_types)
        else:
            content_type = self.rng.choice(document_types)

        language = self.rng.choice(['english', 'chinese'])

        texture = self.generate_flat_texture(
            content_type=content_type,
            language=language
        )

        if is_double_page:
            self.texture_size = orig_size

        return texture
