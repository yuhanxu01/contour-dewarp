"""
Lighting Engine - Physical Light Simulation for Document Rendering

This module simulates realistic lighting conditions for curved document surfaces,
including shadows, highlights, and ambient occlusion effects.

Author: Yuhan Xu
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import cv2


class LightType(Enum):
    """Types of light sources"""
    POINT = "point"           # Point light (like a lamp)
    DIRECTIONAL = "directional"  # Directional light (like sunlight)
    AREA = "area"             # Area light (soft shadows)
    AMBIENT = "ambient"       # Ambient light (uniform)


class LightingScenario(Enum):
    """Predefined lighting scenarios for diversity"""
    OVERHEAD = "overhead"           # 顶光 - 正上方照射
    LEFT_SIDE = "left_side"         # 左侧光
    RIGHT_SIDE = "right_side"       # 右侧光
    FRONT = "front"                 # 正面光
    BACK = "back"                   # 背光/逆光
    TOP_LEFT = "top_left"           # 左上方
    TOP_RIGHT = "top_right"         # 右上方
    BOTTOM_LEFT = "bottom_left"     # 左下方
    BOTTOM_RIGHT = "bottom_right"   # 右下方
    DRAMATIC = "dramatic"           # 戏剧性侧光
    SOFT_DIFFUSE = "soft_diffuse"   # 柔和漫射光
    MULTI_POINT = "multi_point"     # 多点光源
    WINDOW = "window"               # 窗户光（单侧强光）
    DESK_LAMP = "desk_lamp"         # 台灯（点光源）
    STUDIO = "studio"               # 摄影棚三点光


@dataclass
class LightSource:
    """Represents a light source in the scene"""
    light_type: LightType
    position: np.ndarray      # Position (for point) or direction (for directional)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB, normalized
    intensity: float = 1.0
    falloff: float = 1.0      # Distance falloff (for point lights)

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)


@dataclass
class Material:
    """Surface material properties for Phong shading"""
    ambient: float = 0.2      # Ambient reflection coefficient
    diffuse: float = 0.7      # Diffuse reflection coefficient
    specular: float = 0.1     # Specular reflection coefficient
    shininess: float = 32.0   # Specular shininess exponent
    roughness: float = 0.3    # Surface roughness (for subtle variation)


class LightingEngine:
    """
    Simulates realistic lighting for 3D curved document surfaces.

    Uses Phong shading model with support for multiple light sources,
    soft shadows, and ambient occlusion.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        random_seed: Optional[int] = None
    ):
        """
        Initialize the lighting engine.

        Args:
            image_size: Output image size (height, width)
            random_seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.rng = np.random.default_rng(random_seed)

        # Default material (paper-like)
        self.material = Material(
            ambient=0.25,
            diffuse=0.65,
            specular=0.1,
            shininess=16.0,
            roughness=0.2
        )

        # Default light setup
        self.lights: List[LightSource] = []
        self._setup_default_lights()

    def _setup_default_lights(self):
        """Set up default lighting configuration"""
        # Main light (simulating desk lamp or window)
        main_light = LightSource(
            light_type=LightType.DIRECTIONAL,
            position=np.array([0.3, -0.5, 1.0]),  # From top-right
            color=(1.0, 0.98, 0.95),  # Slightly warm
            intensity=0.8
        )

        # Fill light (softer, from opposite side)
        fill_light = LightSource(
            light_type=LightType.DIRECTIONAL,
            position=np.array([-0.2, -0.3, 0.8]),
            color=(0.95, 0.97, 1.0),  # Slightly cool
            intensity=0.3
        )

        # Ambient light
        ambient_light = LightSource(
            light_type=LightType.AMBIENT,
            position=np.array([0, 0, 1]),
            color=(1.0, 1.0, 1.0),
            intensity=0.2
        )

        self.lights = [main_light, fill_light, ambient_light]

    def set_lighting_scenario(self, scenario: LightingScenario):
        """
        Set a specific lighting scenario for diverse lighting conditions.

        Args:
            scenario: The lighting scenario to apply
        """
        self.lights = []

        # Color temperature variations
        warm_color = (1.05, 1.0, 0.92)    # 暖色（钨丝灯）
        cool_color = (0.92, 0.97, 1.05)   # 冷色（日光）
        neutral_color = (1.0, 1.0, 1.0)

        if scenario == LightingScenario.OVERHEAD:
            # 正上方顶光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.0, 0.0, 1.0]),
                color=neutral_color,
                intensity=0.9
            ))

        elif scenario == LightingScenario.LEFT_SIDE:
            # 左侧强光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([-0.8, -0.2, 0.5]),
                color=cool_color,
                intensity=0.85
            ))

        elif scenario == LightingScenario.RIGHT_SIDE:
            # 右侧强光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.8, -0.2, 0.5]),
                color=cool_color,
                intensity=0.85
            ))

        elif scenario == LightingScenario.FRONT:
            # 正面光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.0, -0.9, 0.4]),
                color=neutral_color,
                intensity=0.8
            ))

        elif scenario == LightingScenario.BACK:
            # 背光/逆光（从后方照射）
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.0, 0.8, 0.6]),
                color=warm_color,
                intensity=0.7
            ))
            # 添加弱补光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.0, -0.5, 0.8]),
                color=cool_color,
                intensity=0.25
            ))

        elif scenario == LightingScenario.TOP_LEFT:
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([-0.6, -0.4, 0.7]),
                color=neutral_color,
                intensity=0.85
            ))

        elif scenario == LightingScenario.TOP_RIGHT:
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.6, -0.4, 0.7]),
                color=neutral_color,
                intensity=0.85
            ))

        elif scenario == LightingScenario.BOTTOM_LEFT:
            # 左下方光（较少见，戏剧效果）
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([-0.6, 0.5, 0.5]),
                color=warm_color,
                intensity=0.75
            ))

        elif scenario == LightingScenario.BOTTOM_RIGHT:
            # 右下方光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.6, 0.5, 0.5]),
                color=warm_color,
                intensity=0.75
            ))

        elif scenario == LightingScenario.DRAMATIC:
            # 戏剧性侧光 - 强烈的单侧光，几乎没有补光
            side = self.rng.choice([-1, 1])
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([side * 0.95, -0.1, 0.3]),
                color=warm_color,
                intensity=1.0
            ))
            # 很弱的环境光
            self.lights.append(LightSource(
                light_type=LightType.AMBIENT,
                position=np.array([0, 0, 1]),
                color=cool_color,
                intensity=0.1
            ))
            return  # 跳过默认环境光

        elif scenario == LightingScenario.SOFT_DIFFUSE:
            # 柔和漫射光 - 多个弱光源
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                self.lights.append(LightSource(
                    light_type=LightType.DIRECTIONAL,
                    position=np.array([np.cos(angle) * 0.4, np.sin(angle) * 0.4, 0.8]),
                    color=neutral_color,
                    intensity=0.25
                ))
            # 较强环境光
            self.lights.append(LightSource(
                light_type=LightType.AMBIENT,
                position=np.array([0, 0, 1]),
                color=neutral_color,
                intensity=0.35
            ))
            return

        elif scenario == LightingScenario.MULTI_POINT:
            # 多点光源 - 随机2-4个光源
            num_lights = self.rng.integers(2, 5)
            for i in range(num_lights):
                angle = self.rng.uniform(0, 2 * np.pi)
                elevation = self.rng.uniform(0.3, 0.8)
                temp = self.rng.uniform(-0.05, 0.05)
                self.lights.append(LightSource(
                    light_type=LightType.DIRECTIONAL,
                    position=np.array([
                        np.cos(angle) * np.cos(elevation),
                        np.sin(angle) * np.cos(elevation),
                        np.sin(elevation)
                    ]),
                    color=(1.0 + temp, 1.0, 1.0 - temp),
                    intensity=self.rng.uniform(0.3, 0.6)
                ))

        elif scenario == LightingScenario.WINDOW:
            # 窗户光 - 单侧强烈平行光
            side = self.rng.choice([-1, 1])
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([side * 0.7, -0.3, 0.6]),
                color=cool_color,  # 日光偏冷
                intensity=0.95
            ))
            # 环境反射光
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([-side * 0.3, 0.2, 0.5]),
                color=warm_color,  # 室内反射偏暖
                intensity=0.15
            ))

        elif scenario == LightingScenario.DESK_LAMP:
            # 台灯 - 点光源效果
            lamp_x = self.rng.uniform(-0.5, 0.5)
            lamp_y = self.rng.uniform(-0.6, -0.3)
            self.lights.append(LightSource(
                light_type=LightType.POINT,
                position=np.array([lamp_x * 200, lamp_y * 200, 150]),  # 物理位置
                color=warm_color,  # 台灯通常偏暖
                intensity=1.2,
                falloff=0.001
            ))

        elif scenario == LightingScenario.STUDIO:
            # 摄影棚三点光：主光、补光、轮廓光
            # 主光（Key light）- 45度角
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.5, -0.5, 0.7]),
                color=neutral_color,
                intensity=0.7
            ))
            # 补光（Fill light）- 对侧，较弱
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([-0.4, -0.3, 0.5]),
                color=neutral_color,
                intensity=0.3
            ))
            # 轮廓光（Rim light）- 后方
            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=np.array([0.0, 0.6, 0.4]),
                color=cool_color,
                intensity=0.25
            ))

        # 添加默认环境光（除非场景已经处理）
        self.lights.append(LightSource(
            light_type=LightType.AMBIENT,
            position=np.array([0, 0, 1]),
            color=neutral_color,
            intensity=self.rng.uniform(0.15, 0.25)
        ))

    def set_random_lighting(self, use_scenarios: bool = True):
        """
        Randomize lighting setup for variety.

        Args:
            use_scenarios: If True, randomly select from predefined scenarios.
                          If False, use completely random parameters.
        """
        if use_scenarios and self.rng.random() > 0.3:
            # 70% 概率使用预定义场景
            scenarios = list(LightingScenario)
            scenario = self.rng.choice(scenarios)
            self.set_lighting_scenario(scenario)
            return

        # 完全随机光照
        self.lights = []

        # Random main light - 扩大角度范围，覆盖所有方向
        theta = self.rng.uniform(-np.pi, np.pi)  # 完整水平角度范围
        phi = self.rng.uniform(0.1, 0.9)         # 更广的垂直角度范围

        main_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.cos(theta) * np.cos(phi),
            np.sin(phi)
        ])
        main_dir = main_dir / np.linalg.norm(main_dir)

        # Random warm/cool tint - 更大的色温变化
        temp_offset = self.rng.uniform(-0.1, 0.1)
        main_color = (
            np.clip(1.0 + temp_offset, 0.85, 1.15),
            1.0,
            np.clip(1.0 - temp_offset, 0.85, 1.15)
        )

        main_light = LightSource(
            light_type=LightType.DIRECTIONAL,
            position=main_dir,
            color=main_color,
            intensity=self.rng.uniform(0.5, 1.1)
        )
        self.lights.append(main_light)

        # 50%概率添加第二光源
        if self.rng.random() > 0.5:
            # 补光方向与主光相反或正交
            fill_theta = theta + self.rng.uniform(np.pi/2, 3*np.pi/2)
            fill_phi = self.rng.uniform(0.2, 0.7)
            fill_dir = np.array([
                np.sin(fill_theta) * np.cos(fill_phi),
                np.cos(fill_theta) * np.cos(fill_phi),
                np.sin(fill_phi)
            ])
            fill_dir = fill_dir / np.linalg.norm(fill_dir)

            fill_light = LightSource(
                light_type=LightType.DIRECTIONAL,
                position=fill_dir,
                color=(1.0, 1.0, 1.0),
                intensity=self.rng.uniform(0.1, 0.5)
            )
            self.lights.append(fill_light)

        # 30%概率添加第三光源
        if self.rng.random() > 0.7:
            third_theta = self.rng.uniform(-np.pi, np.pi)
            third_phi = self.rng.uniform(0.1, 0.5)
            third_dir = np.array([
                np.sin(third_theta) * np.cos(third_phi),
                np.cos(third_theta) * np.cos(third_phi),
                np.sin(third_phi)
            ])
            third_dir = third_dir / np.linalg.norm(third_dir)

            self.lights.append(LightSource(
                light_type=LightType.DIRECTIONAL,
                position=third_dir,
                color=self.rng.choice([(1.05, 1.0, 0.95), (0.95, 1.0, 1.05)]),
                intensity=self.rng.uniform(0.1, 0.3)
            ))

        # Always add ambient
        ambient_light = LightSource(
            light_type=LightType.AMBIENT,
            position=np.array([0, 0, 1]),
            color=(1.0, 1.0, 1.0),
            intensity=self.rng.uniform(0.1, 0.35)
        )
        self.lights.append(ambient_light)

    def compute_shading(
        self,
        normals: np.ndarray,
        vertices: np.ndarray,
        view_direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute shading values using Phong illumination model.

        Args:
            normals: Surface normals (H, W, 3)
            vertices: Vertex positions (H, W, 3) - for specular
            view_direction: Camera view direction (default: [0, 0, 1])

        Returns:
            Shading values (H, W, 3) as RGB multipliers
        """
        h, w, _ = normals.shape

        if view_direction is None:
            view_direction = np.array([0, 0, 1], dtype=np.float32)
        view_direction = view_direction / np.linalg.norm(view_direction)

        # Initialize with zeros
        total_shading = np.zeros((h, w, 3), dtype=np.float32)

        for light in self.lights:
            if light.light_type == LightType.AMBIENT:
                # Ambient contribution
                ambient = self.material.ambient * light.intensity
                for c in range(3):
                    total_shading[:, :, c] += ambient * light.color[c]

            elif light.light_type == LightType.DIRECTIONAL:
                # Normalize light direction
                light_dir = light.position / np.linalg.norm(light.position)

                # Diffuse component (Lambertian)
                # N · L
                n_dot_l = np.sum(normals * light_dir, axis=-1)
                n_dot_l = np.clip(n_dot_l, 0, 1)

                diffuse = self.material.diffuse * n_dot_l * light.intensity

                # Specular component (Blinn-Phong)
                # Half vector: H = (L + V) / |L + V|
                half_vector = light_dir + view_direction
                half_vector = half_vector / np.linalg.norm(half_vector)

                n_dot_h = np.sum(normals * half_vector, axis=-1)
                n_dot_h = np.clip(n_dot_h, 0, 1)

                specular = self.material.specular * (n_dot_h ** self.material.shininess) * light.intensity

                # Add to total
                for c in range(3):
                    total_shading[:, :, c] += (diffuse + specular) * light.color[c]

            elif light.light_type == LightType.POINT:
                # Point light - compute direction per vertex
                # Direction from vertex to light
                light_vectors = light.position - vertices
                distances = np.linalg.norm(light_vectors, axis=-1, keepdims=True)
                light_dirs = light_vectors / (distances + 1e-8)

                # Distance attenuation
                attenuation = 1.0 / (1.0 + light.falloff * distances[:, :, 0] ** 2)

                # Diffuse
                n_dot_l = np.sum(normals * light_dirs, axis=-1)
                n_dot_l = np.clip(n_dot_l, 0, 1)

                diffuse = self.material.diffuse * n_dot_l * light.intensity * attenuation

                # Specular
                half_vectors = light_dirs + view_direction
                half_norms = np.linalg.norm(half_vectors, axis=-1, keepdims=True)
                half_vectors = half_vectors / (half_norms + 1e-8)

                n_dot_h = np.sum(normals * half_vectors, axis=-1)
                n_dot_h = np.clip(n_dot_h, 0, 1)

                specular = self.material.specular * (n_dot_h ** self.material.shininess) * light.intensity * attenuation

                for c in range(3):
                    total_shading[:, :, c] += (diffuse + specular) * light.color[c]

        # Add subtle roughness variation
        if self.material.roughness > 0:
            roughness_noise = self.rng.normal(0, self.material.roughness * 0.1, (h, w))
            roughness_noise = cv2.GaussianBlur(roughness_noise.astype(np.float32), (5, 5), 1)
            for c in range(3):
                total_shading[:, :, c] += roughness_noise

        # Clamp to valid range
        total_shading = np.clip(total_shading, 0, 2)  # Allow some overexposure

        return total_shading

    def apply_shading(
        self,
        texture: np.ndarray,
        shading: np.ndarray
    ) -> np.ndarray:
        """
        Apply shading to a texture image.

        Args:
            texture: Input texture (H, W, 3), uint8
            shading: Shading multipliers (H, W, 3), float

        Returns:
            Shaded texture (H, W, 3), uint8
        """
        # Resize shading if necessary
        if texture.shape[:2] != shading.shape[:2]:
            shading = cv2.resize(
                shading,
                (texture.shape[1], texture.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Apply shading
        texture_float = texture.astype(np.float32) / 255.0
        shaded = texture_float * shading

        # Gamma correction for more natural look
        shaded = np.power(shaded, 0.95)

        # Clamp and convert back
        shaded = np.clip(shaded * 255, 0, 255).astype(np.uint8)

        return shaded

    def compute_ambient_occlusion(
        self,
        depth_map: np.ndarray,
        radius: int = 5,
        strength: float = 0.3
    ) -> np.ndarray:
        """
        Compute screen-space ambient occlusion from depth map.

        Args:
            depth_map: Depth values (H, W)
            radius: Sample radius in pixels
            strength: AO strength

        Returns:
            Occlusion values (H, W), 0=occluded, 1=unoccluded
        """
        h, w = depth_map.shape

        # Normalize depth - ensure float32 type
        depth_float = depth_map.astype(np.float32)
        depth_min = depth_float.min()
        depth_max = depth_float.max()
        depth_norm = (depth_float - depth_min) / (depth_max - depth_min + 1e-8)
        depth_norm = depth_norm.astype(np.float32)

        # Compute depth gradients
        grad_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)

        # Areas with high curvature (large gradients) get more occlusion
        curvature = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        # Blur for soft effect
        curvature_blurred = cv2.GaussianBlur(curvature, (radius * 2 + 1, radius * 2 + 1), float(radius / 2))

        # Convert to occlusion factor
        occlusion = 1.0 - curvature_blurred * strength * 10
        occlusion = np.clip(occlusion, 0.5, 1.0)

        return occlusion

    def add_shadows(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        shadow_softness: float = 0.5,
        shadow_intensity: float = 0.3
    ) -> np.ndarray:
        """
        Add soft shadows based on depth map.

        This simulates shadows cast by raised portions of the page.

        Args:
            image: Input image (H, W, 3)
            depth_map: Depth values (H, W)
            shadow_softness: Blur amount for soft shadows
            shadow_intensity: Shadow darkness

        Returns:
            Image with shadows (H, W, 3)
        """
        h, w = depth_map.shape

        # Convert to float32
        depth_float = depth_map.astype(np.float32)

        # Resize depth map if needed
        if image.shape[:2] != (h, w):
            depth_float = cv2.resize(depth_float, (image.shape[1], image.shape[0]))
            h, w = image.shape[:2]

        # Normalize depth
        depth_norm = (depth_float - depth_float.min()) / (depth_float.max() - depth_float.min() + 1e-8)
        depth_norm = depth_norm.astype(np.float32)

        # Compute shadow based on depth gradient
        # Shadows appear on the side opposite to light
        main_light_dir = self.lights[0].position if self.lights else np.array([0.3, -0.5, 1.0])

        # Project light direction to 2D
        light_2d = np.array([main_light_dir[0], main_light_dir[1]])
        light_2d = light_2d / (np.linalg.norm(light_2d) + 1e-8)

        # Shift depth map in light direction
        shift_x = int(light_2d[0] * 10)
        shift_y = int(light_2d[1] * 10)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_depth = cv2.warpAffine(depth_norm, M, (w, h))

        # Shadow where shifted depth > current depth
        shadow_map = np.maximum(0, shifted_depth - depth_norm).astype(np.float32)

        # Blur for soft shadows
        blur_size = int(shadow_softness * 20) * 2 + 1
        shadow_map = cv2.GaussianBlur(shadow_map, (blur_size, blur_size), float(blur_size / 3))

        # Normalize
        shadow_map = shadow_map / (shadow_map.max() + 1e-8)

        # Apply shadow
        shadow_factor = 1.0 - shadow_map * shadow_intensity
        shadow_factor = np.clip(shadow_factor, 0.5, 1.0)

        # Apply to image
        image_float = image.astype(np.float32)
        for c in range(3):
            image_float[:, :, c] *= shadow_factor

        return image_float.astype(np.uint8)

    def render_with_lighting(
        self,
        texture: np.ndarray,
        normals: np.ndarray,
        vertices: np.ndarray,
        depth_map: np.ndarray,
        add_ao: bool = True,
        add_shadows: bool = True
    ) -> np.ndarray:
        """
        Complete lighting render pipeline.

        Args:
            texture: Input texture (H, W, 3)
            normals: Surface normals (H_mesh, W_mesh, 3)
            vertices: Vertex positions (H_mesh, W_mesh, 3)
            depth_map: Depth values (H_mesh, W_mesh)
            add_ao: Add ambient occlusion
            add_shadows: Add soft shadows

        Returns:
            Fully lit and shaded image (H, W, 3)
        """
        # Compute base shading
        shading = self.compute_shading(normals, vertices)

        # Apply base shading
        result = self.apply_shading(texture, shading)

        # Add ambient occlusion
        if add_ao:
            ao = self.compute_ambient_occlusion(depth_map)
            ao_resized = cv2.resize(ao, (result.shape[1], result.shape[0]))
            ao_3ch = np.stack([ao_resized] * 3, axis=-1)
            result = (result.astype(np.float32) * ao_3ch).astype(np.uint8)

        # Add shadows
        if add_shadows:
            result = self.add_shadows(result, depth_map)

        return result

    def create_light_from_angle(
        self,
        azimuth: float,
        elevation: float,
        intensity: float = 1.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> LightSource:
        """
        Create a directional light from spherical angles.

        Args:
            azimuth: Horizontal angle in radians (0 = front, positive = right)
            elevation: Vertical angle in radians (0 = horizontal, positive = up)
            intensity: Light intensity
            color: Light color RGB

        Returns:
            LightSource object
        """
        direction = np.array([
            np.sin(azimuth) * np.cos(elevation),
            -np.sin(elevation),
            np.cos(azimuth) * np.cos(elevation)
        ])

        return LightSource(
            light_type=LightType.DIRECTIONAL,
            position=direction,
            color=color,
            intensity=intensity
        )
