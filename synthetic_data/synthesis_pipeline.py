"""
Synthesis Data Generator - Main Pipeline for Training Data Generation

This module orchestrates the complete synthetic data generation pipeline,
combining page mesh generation, texture rendering, and lighting simulation
to produce training pairs of curved document images and their depth maps.

Author: Yuhan Xu
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .page_mesh import PageMeshGenerator, CurvatureType, PageGeometry
from .texture_renderer import TextureRenderer, ContentType
from .lighting_engine import LightingEngine, LightSource, LightType


@dataclass
class SynthesisConfig:
    """Configuration for synthetic data generation"""
    # Output settings
    output_size: Tuple[int, int] = (512, 512)  # (height, width)
    depth_map_size: Tuple[int, int] = (512, 512)

    # Mesh settings
    mesh_resolution: Tuple[int, int] = (128, 128)
    page_width: float = 210.0   # mm
    page_height: float = 297.0  # mm

    # Texture settings
    texture_size: Tuple[int, int] = (1024, 1024)

    # Curvature settings
    curvature_range: Tuple[float, float] = (0.05, 0.3)
    double_page_probability: float = 0.4

    # Lighting settings
    randomize_lighting: bool = True
    add_shadows: bool = True
    add_ambient_occlusion: bool = True

    # Data augmentation
    add_noise: bool = True
    noise_level: float = 0.02
    add_blur: bool = True
    blur_probability: float = 0.3
    blur_kernel_range: Tuple[int, int] = (3, 7)

    # Content settings
    language_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'english': 0.6,
        'chinese': 0.4
    })
    content_type_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'text_only': 0.4,
        'text_with_images': 0.3,
        'mixed_layout': 0.2,
        'technical': 0.1
    })


@dataclass
class SynthesisResult:
    """Result of a single synthesis operation"""
    image: np.ndarray           # Rendered curved document image
    depth_map: np.ndarray       # Ground truth depth map
    flat_texture: np.ndarray    # Original flat texture (for reference)
    metadata: Dict[str, Any]    # Generation parameters


class SynthesisDataGenerator:
    """
    Main pipeline for generating synthetic curved document data.

    This class orchestrates the complete data generation process:
    1. Generate curved page mesh
    2. Create document texture (text + images)
    3. Map texture onto curved surface
    4. Apply realistic lighting
    5. Generate ground truth depth map
    """

    def __init__(
        self,
        config: Optional[SynthesisConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            config: Synthesis configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or SynthesisConfig()
        self.rng = np.random.default_rng(random_seed)

        # Initialize components
        self.mesh_generator = PageMeshGenerator(
            page_width=self.config.page_width,
            page_height=self.config.page_height,
            mesh_resolution=self.config.mesh_resolution,
            random_seed=random_seed
        )

        self.texture_renderer = TextureRenderer(
            texture_size=self.config.texture_size,
            random_seed=random_seed
        )

        self.lighting_engine = LightingEngine(
            image_size=self.config.output_size,
            random_seed=random_seed
        )

    def generate_single_sample(
        self,
        is_double_page: Optional[bool] = None,
        curvature_type: Optional[CurvatureType] = None,
        content_type: Optional[ContentType] = None,
        language: Optional[str] = None,
        curvature_strength: Optional[float] = None
    ) -> SynthesisResult:
        """
        Generate a single synthetic sample.

        Args:
            is_double_page: Force single/double page mode (None = random)
            curvature_type: Specific curvature type (None = random)
            content_type: Specific content type (None = random)
            language: Content language (None = random)
            curvature_strength: Curvature intensity (None = random from range)

        Returns:
            SynthesisResult with image, depth map, and metadata
        """
        metadata = {}

        # Determine page type
        if is_double_page is None:
            is_double_page = self.rng.random() < self.config.double_page_probability
        metadata['is_double_page'] = is_double_page

        # Determine curvature strength
        if curvature_strength is None:
            curvature_strength = self.rng.uniform(*self.config.curvature_range)
        metadata['curvature_strength'] = float(curvature_strength)

        # Generate page geometry
        if is_double_page:
            asymmetry = self.rng.uniform(-0.3, 0.3)
            spine_position = self.rng.uniform(0.45, 0.55)
            geometry = self.mesh_generator.generate_double_page(
                curvature_strength=curvature_strength,
                spine_position=spine_position,
                asymmetry=asymmetry
            )
            metadata['asymmetry'] = float(asymmetry)
            metadata['spine_position'] = float(spine_position)
            metadata['curvature_type'] = 'book_spine'
        else:
            if curvature_type is None:
                curvature_type = self.rng.choice([
                    CurvatureType.CYLINDRICAL,
                    CurvatureType.BOOK_SPINE,
                    CurvatureType.WAVE,
                    CurvatureType.COMBINED,
                    CurvatureType.PERSPECTIVE,
                ])
            geometry = self.mesh_generator.generate_single_page(
                curvature_type=curvature_type,
                curvature_strength=curvature_strength
            )
            metadata['curvature_type'] = curvature_type.value

        # Determine content type
        if content_type is None:
            content_types = list(self.config.content_type_probabilities.keys())
            content_probs = list(self.config.content_type_probabilities.values())
            content_type_str = self.rng.choice(content_types, p=content_probs)
            content_type = ContentType(content_type_str)
        metadata['content_type'] = content_type.value

        # Determine language
        if language is None:
            languages = list(self.config.language_probabilities.keys())
            lang_probs = list(self.config.language_probabilities.values())
            language = self.rng.choice(languages, p=lang_probs)
        metadata['language'] = language

        # Generate flat texture
        if is_double_page:
            orig_size = self.texture_renderer.texture_size
            self.texture_renderer.texture_size = (orig_size[0], orig_size[1] * 2)
            flat_texture = self.texture_renderer.generate_flat_texture(
                content_type=content_type,
                language=language
            )
            self.texture_renderer.texture_size = orig_size
        else:
            flat_texture = self.texture_renderer.generate_flat_texture(
                content_type=content_type,
                language=language
            )

        # Map texture to curved surface
        warped_texture = self.texture_renderer.map_texture_to_surface(
            flat_texture,
            geometry.vertices,
            geometry.uv_coords,
            self.config.output_size
        )

        # Setup lighting
        if self.config.randomize_lighting:
            self.lighting_engine.set_random_lighting()

        # Store lighting info
        metadata['lighting'] = {
            'num_lights': len(self.lighting_engine.lights),
            'main_light_intensity': float(self.lighting_engine.lights[0].intensity) if self.lighting_engine.lights else 0
        }

        # Resize normals and depth for lighting
        h_out, w_out = self.config.output_size
        normals_resized = cv2.resize(
            geometry.normals,
            (w_out, h_out),
            interpolation=cv2.INTER_LINEAR
        )
        vertices_resized = cv2.resize(
            geometry.vertices,
            (w_out, h_out),
            interpolation=cv2.INTER_LINEAR
        )
        depth_resized = cv2.resize(
            geometry.depth_map,
            (w_out, h_out),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply lighting
        lit_image = self.lighting_engine.render_with_lighting(
            warped_texture,
            normals_resized,
            vertices_resized,
            depth_resized,
            add_ao=self.config.add_ambient_occlusion,
            add_shadows=self.config.add_shadows
        )

        # Apply data augmentation
        if self.config.add_noise:
            lit_image = self._add_noise(lit_image)

        if self.config.add_blur and self.rng.random() < self.config.blur_probability:
            lit_image = self._add_blur(lit_image)

        # Prepare depth map
        depth_map = self._normalize_depth_map(
            geometry.depth_map,
            self.config.depth_map_size
        )

        return SynthesisResult(
            image=lit_image,
            depth_map=depth_map,
            flat_texture=flat_texture,
            metadata=metadata
        )

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic sensor noise to image"""
        noise_level = self.config.noise_level * 255
        noise = self.rng.normal(0, noise_level, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """Add slight blur to simulate camera focus issues"""
        kernel_size = self.rng.integers(*self.config.blur_kernel_range)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _normalize_depth_map(
        self,
        depth_map: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Normalize and resize depth map for output.

        Args:
            depth_map: Raw depth values
            output_size: Target size (height, width)

        Returns:
            Normalized depth map as uint16 (0-65535 range)
        """
        # Resize
        depth_resized = cv2.resize(
            depth_map.astype(np.float32),
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to 0-1
        d_min, d_max = depth_resized.min(), depth_resized.max()
        if d_max > d_min:
            depth_normalized = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)

        # Convert to uint16 for better precision
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)

        return depth_uint16

    def generate_batch(
        self,
        num_samples: int,
        output_dir: Optional[str] = None,
        save_flat_texture: bool = False,
        show_progress: bool = True
    ) -> List[SynthesisResult]:
        """
        Generate a batch of synthetic samples.

        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save samples (None = don't save)
            save_flat_texture: Also save flat textures
            show_progress: Show progress bar

        Returns:
            List of SynthesisResult objects
        """
        results = []

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / 'images').mkdir(exist_ok=True)
            (output_path / 'depth_maps').mkdir(exist_ok=True)
            if save_flat_texture:
                (output_path / 'flat_textures').mkdir(exist_ok=True)

        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating samples")

        for i in iterator:
            try:
                result = self.generate_single_sample()
                results.append(result)

                if output_dir:
                    self._save_sample(
                        result, i, output_path,
                        save_flat_texture=save_flat_texture
                    )
            except Exception as e:
                print(f"Warning: Failed to generate sample {i}: {e}")
                continue

        # Save metadata summary
        if output_dir:
            self._save_metadata_summary(results, output_path)

        return results

    def _save_sample(
        self,
        result: SynthesisResult,
        index: int,
        output_path: Path,
        save_flat_texture: bool = False
    ):
        """Save a single sample to disk"""
        # Save image
        img_path = output_path / 'images' / f'{index:05d}.png'
        cv2.imwrite(str(img_path), result.image)

        # Save depth map
        depth_path = output_path / 'depth_maps' / f'{index:05d}.png'
        cv2.imwrite(str(depth_path), result.depth_map)

        # Save flat texture if requested
        if save_flat_texture:
            flat_path = output_path / 'flat_textures' / f'{index:05d}.png'
            cv2.imwrite(str(flat_path), result.flat_texture)

        # Save per-sample metadata
        meta_path = output_path / 'images' / f'{index:05d}.json'
        with open(meta_path, 'w') as f:
            json.dump(result.metadata, f, indent=2)

    def _save_metadata_summary(
        self,
        results: List[SynthesisResult],
        output_path: Path
    ):
        """Save summary metadata for the dataset"""
        summary = {
            'num_samples': len(results),
            'config': {
                'output_size': self.config.output_size,
                'depth_map_size': self.config.depth_map_size,
                'curvature_range': self.config.curvature_range,
                'double_page_probability': self.config.double_page_probability,
            },
            'statistics': {}
        }

        # Compute statistics
        if results:
            double_page_count = sum(1 for r in results if r.metadata.get('is_double_page', False))
            summary['statistics']['double_page_ratio'] = double_page_count / len(results)

            curvature_values = [r.metadata.get('curvature_strength', 0) for r in results]
            summary['statistics']['mean_curvature'] = float(np.mean(curvature_values))
            summary['statistics']['std_curvature'] = float(np.std(curvature_values))

            # Content type distribution
            content_types = [r.metadata.get('content_type', 'unknown') for r in results]
            content_dist = {}
            for ct in set(content_types):
                content_dist[ct] = content_types.count(ct) / len(content_types)
            summary['statistics']['content_type_distribution'] = content_dist

        summary_path = output_path / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def create_visualization(
    result: SynthesisResult,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a visualization of a synthesis result.

    Shows the curved image, depth map, and flat texture side by side.

    Args:
        result: SynthesisResult to visualize
        output_path: Path to save visualization (None = don't save)

    Returns:
        Visualization image
    """
    h, w = result.image.shape[:2]

    # Resize flat texture to match
    flat_resized = cv2.resize(result.flat_texture, (w, h))

    # Convert depth to colored visualization
    depth_normalized = (result.depth_map.astype(np.float32) / 65535 * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

    # Stack horizontally
    viz = np.hstack([result.image, depth_colored, flat_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, 'Curved Image', (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(viz, 'Depth Map', (w + 10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(viz, 'Flat Texture', (2 * w + 10, 25), font, 0.7, (255, 255, 255), 2)

    # Add metadata info
    meta_text = f"Type: {'Double' if result.metadata.get('is_double_page') else 'Single'} | "
    meta_text += f"Curvature: {result.metadata.get('curvature_strength', 0):.3f}"
    cv2.putText(viz, meta_text, (10, h - 10), font, 0.5, (200, 200, 200), 1)

    if output_path:
        cv2.imwrite(output_path, viz)

    return viz


# Convenience function for quick generation
def generate_training_data(
    num_samples: int,
    output_dir: str,
    config: Optional[SynthesisConfig] = None,
    random_seed: Optional[int] = None,
    **kwargs
) -> List[SynthesisResult]:
    """
    Convenience function to generate training data.

    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory
        config: Synthesis configuration
        random_seed: Random seed
        **kwargs: Additional arguments passed to generate_batch

    Returns:
        List of SynthesisResult objects
    """
    generator = SynthesisDataGenerator(config=config, random_seed=random_seed)
    return generator.generate_batch(num_samples, output_dir=output_dir, **kwargs)
