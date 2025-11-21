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

    def set_random_lighting(self):
        """Randomize lighting setup for variety"""
        self.lights = []

        # Random main light
        theta = self.rng.uniform(-0.5, 0.5)  # Horizontal angle
        phi = self.rng.uniform(0.3, 0.8)     # Vertical angle

        main_dir = np.array([
            np.sin(theta),
            -np.cos(phi),
            np.sin(phi)
        ])
        main_dir = main_dir / np.linalg.norm(main_dir)

        # Random warm/cool tint
        temp_offset = self.rng.uniform(-0.05, 0.05)
        main_color = (
            1.0 + temp_offset,
            1.0,
            1.0 - temp_offset
        )

        main_light = LightSource(
            light_type=LightType.DIRECTIONAL,
            position=main_dir,
            color=main_color,
            intensity=self.rng.uniform(0.6, 1.0)
        )
        self.lights.append(main_light)

        # Sometimes add a secondary light
        if self.rng.random() > 0.3:
            fill_dir = np.array([
                -main_dir[0] + self.rng.uniform(-0.2, 0.2),
                main_dir[1] * 0.5,
                main_dir[2] * 0.8
            ])
            fill_dir = fill_dir / np.linalg.norm(fill_dir)

            fill_light = LightSource(
                light_type=LightType.DIRECTIONAL,
                position=fill_dir,
                color=(1.0, 1.0, 1.0),
                intensity=self.rng.uniform(0.1, 0.4)
            )
            self.lights.append(fill_light)

        # Always add ambient
        ambient_light = LightSource(
            light_type=LightType.AMBIENT,
            position=np.array([0, 0, 1]),
            color=(1.0, 1.0, 1.0),
            intensity=self.rng.uniform(0.15, 0.3)
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
