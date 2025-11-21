"""
Page Mesh Generator - 3D Curved Page Surface Generation

This module generates realistic 3D curved page surfaces for simulating
bent/folded book pages. Supports both single-page and double-page (book spread)
configurations with various curvature types.

Author: Yuhan Xu
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, List
from dataclasses import dataclass


class CurvatureType(Enum):
    """Types of page curvature for simulation"""
    CYLINDRICAL = "cylindrical"       # Simple cylindrical bend (like a rolled page)
    BOOK_SPINE = "book_spine"         # V-shaped bend at book spine
    WAVE = "wave"                     # Wavy surface (multiple bends)
    CORNER_FOLD = "corner_fold"       # Folded corner
    RANDOM_DEFORMATION = "random"     # Random smooth deformation
    PERSPECTIVE = "perspective"       # Page at an angle (perspective distortion)
    COMBINED = "combined"             # Combination of multiple curvature types


@dataclass
class PageGeometry:
    """Stores the 3D geometry of a generated page surface"""
    vertices: np.ndarray      # Shape: (H, W, 3) - 3D vertex positions
    normals: np.ndarray       # Shape: (H, W, 3) - Surface normals
    uv_coords: np.ndarray     # Shape: (H, W, 2) - Texture coordinates
    depth_map: np.ndarray     # Shape: (H, W) - Depth values (Z coordinates)
    is_double_page: bool      # Whether this is a double-page spread


class PageMeshGenerator:
    """
    Generates 3D curved page surfaces for document simulation.

    The generator creates parametric surfaces that simulate various types of
    page curvature commonly seen in scanned books and documents.
    """

    def __init__(
        self,
        page_width: float = 210.0,      # mm (A4 width)
        page_height: float = 297.0,     # mm (A4 height)
        mesh_resolution: Tuple[int, int] = (256, 256),  # (height, width)
        random_seed: Optional[int] = None
    ):
        """
        Initialize the page mesh generator.

        Args:
            page_width: Physical page width in mm
            page_height: Physical page height in mm
            mesh_resolution: Resolution of the mesh grid (height, width)
            random_seed: Random seed for reproducibility
        """
        self.page_width = page_width
        self.page_height = page_height
        self.mesh_resolution = mesh_resolution
        self.rng = np.random.default_rng(random_seed)

        # Pre-compute base grid
        self._create_base_grid()

    def _create_base_grid(self):
        """Create the base 2D grid for surface generation"""
        h, w = self.mesh_resolution

        # Normalized coordinates [0, 1]
        u = np.linspace(0, 1, w)
        v = np.linspace(0, 1, h)
        self.u_grid, self.v_grid = np.meshgrid(u, v)

        # Physical coordinates
        self.x_grid = self.u_grid * self.page_width
        self.y_grid = self.v_grid * self.page_height

    def generate_single_page(
        self,
        curvature_type: CurvatureType = CurvatureType.CYLINDRICAL,
        curvature_strength: float = 0.1,
        **kwargs
    ) -> PageGeometry:
        """
        Generate a single curved page surface.

        Args:
            curvature_type: Type of curvature to apply
            curvature_strength: Strength of the curvature (0.0 = flat, 1.0 = maximum)
            **kwargs: Additional parameters for specific curvature types

        Returns:
            PageGeometry object containing the 3D surface data
        """
        # Start with flat surface
        z_grid = np.zeros_like(self.x_grid)

        # Apply curvature based on type
        if curvature_type == CurvatureType.CYLINDRICAL:
            z_grid = self._apply_cylindrical_curvature(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.BOOK_SPINE:
            z_grid = self._apply_book_spine_curvature(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.WAVE:
            z_grid = self._apply_wave_curvature(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.CORNER_FOLD:
            z_grid = self._apply_corner_fold(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.RANDOM_DEFORMATION:
            z_grid = self._apply_random_deformation(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.PERSPECTIVE:
            z_grid = self._apply_perspective(z_grid, curvature_strength, **kwargs)
        elif curvature_type == CurvatureType.COMBINED:
            z_grid = self._apply_combined_curvature(z_grid, curvature_strength, **kwargs)

        # Build geometry
        return self._build_geometry(z_grid, is_double_page=False)

    def generate_double_page(
        self,
        curvature_strength: float = 0.15,
        spine_position: float = 0.5,
        asymmetry: float = 0.0,
        **kwargs
    ) -> PageGeometry:
        """
        Generate a double-page (book spread) surface with spine curvature.

        Args:
            curvature_strength: Overall curvature strength
            spine_position: Position of spine (0.0 = left, 1.0 = right)
            asymmetry: Asymmetry between left and right pages (-1 to 1)
            **kwargs: Additional parameters

        Returns:
            PageGeometry object containing the 3D surface data
        """
        # Double the width for a spread
        h, w = self.mesh_resolution
        u = np.linspace(0, 1, w)
        v = np.linspace(0, 1, h)
        u_grid, v_grid = np.meshgrid(u, v)

        # Physical coordinates (double width)
        x_grid = u_grid * self.page_width * 2
        y_grid = v_grid * self.page_height

        # Create V-shaped spine curvature
        z_grid = np.zeros_like(x_grid)

        # Normalize x to [-1, 1] centered at spine
        x_normalized = (u_grid - spine_position) * 2

        # Base spine curvature (V-shape)
        spine_depth = curvature_strength * self.page_width * 0.3
        z_spine = spine_depth * (1 - np.abs(x_normalized) ** 1.5)

        # Add asymmetry
        if asymmetry != 0:
            asymmetry_factor = 1 + asymmetry * x_normalized
            z_spine *= asymmetry_factor

        z_grid += z_spine

        # Add subtle cylindrical curvature along y-axis
        y_curvature = curvature_strength * self.page_height * 0.05
        y_normalized = v_grid - 0.5
        z_grid += y_curvature * (1 - 4 * y_normalized**2)

        # Add page edge lift (pages tend to lift at outer edges)
        edge_lift = curvature_strength * self.page_width * 0.02
        edge_factor = np.abs(x_normalized) ** 3
        z_grid -= edge_lift * edge_factor

        # Add optional random perturbation
        if kwargs.get('add_noise', True):
            noise_strength = curvature_strength * 0.1
            z_grid += self._generate_smooth_noise(z_grid.shape, noise_strength * self.page_width * 0.01)

        # Build geometry with double-page grid
        vertices = np.stack([x_grid, y_grid, z_grid], axis=-1)
        normals = self._compute_normals(vertices)
        uv_coords = np.stack([u_grid, v_grid], axis=-1)
        depth_map = z_grid

        return PageGeometry(
            vertices=vertices,
            normals=normals,
            uv_coords=uv_coords,
            depth_map=depth_map,
            is_double_page=True
        )

    def _apply_cylindrical_curvature(
        self,
        z_grid: np.ndarray,
        strength: float,
        axis: str = 'x',
        radius_factor: float = 1.0
    ) -> np.ndarray:
        """Apply cylindrical curvature (like a rolled page)"""
        radius = self.page_width * radius_factor / (strength + 0.1)

        if axis == 'x':
            # Curve along x-axis
            x_normalized = self.u_grid - 0.5
            # Circular arc: z = R - sqrt(R^2 - x^2) ≈ x^2/(2R) for small x
            z_curvature = (x_normalized * self.page_width)**2 / (2 * radius)
        else:
            # Curve along y-axis
            y_normalized = self.v_grid - 0.5
            z_curvature = (y_normalized * self.page_height)**2 / (2 * radius)

        return z_grid + z_curvature * strength * 10

    def _apply_book_spine_curvature(
        self,
        z_grid: np.ndarray,
        strength: float,
        spine_position: float = 0.0,  # 0 = left edge, 0.5 = center
        **kwargs
    ) -> np.ndarray:
        """Apply book spine curvature (page curving up from spine)"""
        # Distance from spine
        x_from_spine = self.u_grid - spine_position

        # Exponential decay from spine
        decay_rate = kwargs.get('decay_rate', 3.0)
        spine_depth = strength * self.page_width * 0.2

        # Page rises as it moves away from spine
        z_curvature = spine_depth * (1 - np.exp(-decay_rate * np.abs(x_from_spine)))

        # Add subtle variation along y
        y_variation = 0.1 * np.sin(2 * np.pi * self.v_grid) * strength * self.page_height * 0.02

        return z_grid + z_curvature + y_variation

    def _apply_wave_curvature(
        self,
        z_grid: np.ndarray,
        strength: float,
        frequency_x: float = 1.5,
        frequency_y: float = 0.5,
        **kwargs
    ) -> np.ndarray:
        """Apply wavy surface curvature"""
        amplitude = strength * self.page_width * 0.1

        # Combination of sinusoidal waves
        phase_x = kwargs.get('phase_x', 0.0)
        phase_y = kwargs.get('phase_y', 0.0)

        wave_x = np.sin(2 * np.pi * frequency_x * self.u_grid + phase_x)
        wave_y = np.sin(2 * np.pi * frequency_y * self.v_grid + phase_y)

        z_curvature = amplitude * (wave_x * 0.7 + wave_y * 0.3)

        return z_grid + z_curvature

    def _apply_corner_fold(
        self,
        z_grid: np.ndarray,
        strength: float,
        corner: str = 'top_right',
        fold_size: float = 0.2,
        **kwargs
    ) -> np.ndarray:
        """Apply corner fold deformation"""
        # Determine corner position
        if corner == 'top_right':
            cx, cy = 1.0, 0.0
        elif corner == 'top_left':
            cx, cy = 0.0, 0.0
        elif corner == 'bottom_right':
            cx, cy = 1.0, 1.0
        else:  # bottom_left
            cx, cy = 0.0, 1.0

        # Distance from corner
        dist = np.sqrt((self.u_grid - cx)**2 + (self.v_grid - cy)**2)

        # Fold effect
        fold_radius = fold_size
        fold_height = strength * self.page_width * 0.15

        # Smooth fold using sigmoid-like function
        fold_factor = np.clip(1 - dist / fold_radius, 0, 1)
        fold_factor = fold_factor ** 2  # Smooth transition

        z_curvature = fold_height * fold_factor

        return z_grid + z_curvature

    def _apply_random_deformation(
        self,
        z_grid: np.ndarray,
        strength: float,
        smoothness: float = 0.1,
        **kwargs
    ) -> np.ndarray:
        """Apply smooth random deformation"""
        amplitude = strength * self.page_width * 0.15

        # Generate smooth random field using multiple frequency components
        z_random = self._generate_smooth_noise(z_grid.shape, amplitude, smoothness)

        return z_grid + z_random

    def _apply_perspective(
        self,
        z_grid: np.ndarray,
        strength: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """Apply perspective tilt (page at an angle)"""
        if tilt_x == 0.0 and tilt_y == 0.0:
            # Random tilt
            tilt_x = (self.rng.random() - 0.5) * strength
            tilt_y = (self.rng.random() - 0.5) * strength * 0.5

        # Linear tilt
        z_tilt = (self.u_grid - 0.5) * self.page_width * tilt_x * 0.3
        z_tilt += (self.v_grid - 0.5) * self.page_height * tilt_y * 0.3

        return z_grid + z_tilt

    def _apply_combined_curvature(
        self,
        z_grid: np.ndarray,
        strength: float,
        **kwargs
    ) -> np.ndarray:
        """Apply a combination of curvature types"""
        # Randomly select and combine curvature types
        components = kwargs.get('components', None)

        if components is None:
            # Default: cylindrical + wave + random
            z_grid = self._apply_cylindrical_curvature(z_grid, strength * 0.5)
            z_grid = self._apply_wave_curvature(
                z_grid, strength * 0.3,
                frequency_x=self.rng.uniform(0.5, 2.0),
                frequency_y=self.rng.uniform(0.3, 1.0)
            )
            z_grid = self._apply_random_deformation(z_grid, strength * 0.2)
        else:
            for component, weight in components:
                if component == CurvatureType.CYLINDRICAL:
                    z_grid = self._apply_cylindrical_curvature(z_grid, strength * weight)
                elif component == CurvatureType.WAVE:
                    z_grid = self._apply_wave_curvature(z_grid, strength * weight)
                elif component == CurvatureType.RANDOM_DEFORMATION:
                    z_grid = self._apply_random_deformation(z_grid, strength * weight)

        return z_grid

    def _generate_smooth_noise(
        self,
        shape: Tuple[int, int],
        amplitude: float,
        smoothness: float = 0.1
    ) -> np.ndarray:
        """Generate smooth random noise using Perlin-like approach"""
        h, w = shape

        # Multi-scale noise
        noise = np.zeros(shape)

        for scale in [4, 8, 16, 32]:
            # Generate low-res noise
            small_h = max(2, h // scale)
            small_w = max(2, w // scale)
            small_noise = self.rng.random((small_h, small_w)) - 0.5

            # Upscale with interpolation
            from scipy.ndimage import zoom
            upscaled = zoom(small_noise, (h / small_h, w / small_w), order=3)

            # Add with decreasing weight for higher frequencies
            weight = 1.0 / (scale ** smoothness)
            noise += upscaled[:h, :w] * weight

        # Normalize and scale
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        return noise * amplitude

    def _build_geometry(self, z_grid: np.ndarray, is_double_page: bool) -> PageGeometry:
        """Build PageGeometry from z-grid"""
        vertices = np.stack([self.x_grid, self.y_grid, z_grid], axis=-1)
        normals = self._compute_normals(vertices)
        uv_coords = np.stack([self.u_grid, self.v_grid], axis=-1)

        return PageGeometry(
            vertices=vertices,
            normals=normals,
            uv_coords=uv_coords,
            depth_map=z_grid,
            is_double_page=is_double_page
        )

    def _compute_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Compute surface normals from vertex grid"""
        h, w, _ = vertices.shape
        normals = np.zeros_like(vertices)

        # Compute partial derivatives using central differences
        # dV/du (along x)
        du = np.zeros_like(vertices)
        du[:, 1:-1] = (vertices[:, 2:] - vertices[:, :-2]) / 2
        du[:, 0] = vertices[:, 1] - vertices[:, 0]
        du[:, -1] = vertices[:, -1] - vertices[:, -2]

        # dV/dv (along y)
        dv = np.zeros_like(vertices)
        dv[1:-1, :] = (vertices[2:, :] - vertices[:-2, :]) / 2
        dv[0, :] = vertices[1, :] - vertices[0, :]
        dv[-1, :] = vertices[-1, :] - vertices[-2, :]

        # Normal = du × dv (cross product)
        normals = np.cross(du, dv)

        # Normalize
        magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (magnitude + 1e-8)

        # Ensure normals point upward (positive z)
        normals[normals[:, :, 2] < 0] *= -1

        return normals

    def generate_random_page(
        self,
        is_double_page: Optional[bool] = None,
        curvature_range: Tuple[float, float] = (0.05, 0.25)
    ) -> PageGeometry:
        """
        Generate a page with random curvature parameters.

        Args:
            is_double_page: If None, randomly choose single or double
            curvature_range: Range of curvature strength

        Returns:
            PageGeometry with random curvature
        """
        if is_double_page is None:
            is_double_page = self.rng.random() > 0.5

        strength = self.rng.uniform(*curvature_range)

        if is_double_page:
            asymmetry = self.rng.uniform(-0.3, 0.3)
            spine_position = self.rng.uniform(0.45, 0.55)
            return self.generate_double_page(
                curvature_strength=strength,
                spine_position=spine_position,
                asymmetry=asymmetry
            )
        else:
            # Randomly select curvature type
            curvature_types = [
                CurvatureType.CYLINDRICAL,
                CurvatureType.BOOK_SPINE,
                CurvatureType.WAVE,
                CurvatureType.COMBINED,
                CurvatureType.PERSPECTIVE,
            ]
            curvature_type = self.rng.choice(curvature_types)

            return self.generate_single_page(
                curvature_type=curvature_type,
                curvature_strength=strength
            )
