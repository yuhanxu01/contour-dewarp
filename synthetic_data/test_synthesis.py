#!/usr/bin/env python3
"""
Test Script for Synthetic Data Generation Framework

This script tests all components of the synthetic data generation pipeline
and generates 20 test samples to verify the framework functionality.

Usage:
    python -m synthetic_data.test_synthesis

Author: Yuhan Xu
"""

import numpy as np
import cv2
import sys
import os
from pathlib import Path
from typing import List, Tuple
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.page_mesh import PageMeshGenerator, CurvatureType, PageGeometry
from synthetic_data.texture_renderer import TextureRenderer, ContentType
from synthetic_data.lighting_engine import LightingEngine, LightSource, LightType
from synthetic_data.synthesis_pipeline import (
    SynthesisDataGenerator,
    SynthesisConfig,
    SynthesisResult,
    create_visualization
)


def test_page_mesh_generator():
    """Test the PageMeshGenerator component"""
    print("\n" + "=" * 60)
    print("Testing PageMeshGenerator")
    print("=" * 60)

    generator = PageMeshGenerator(
        page_width=210.0,
        page_height=297.0,
        mesh_resolution=(64, 64),
        random_seed=42
    )

    # Test single page with different curvature types
    curvature_types = [
        CurvatureType.CYLINDRICAL,
        CurvatureType.BOOK_SPINE,
        CurvatureType.WAVE,
        CurvatureType.CORNER_FOLD,
        CurvatureType.RANDOM_DEFORMATION,
        CurvatureType.PERSPECTIVE,
        CurvatureType.COMBINED,
    ]

    for ct in curvature_types:
        geometry = generator.generate_single_page(
            curvature_type=ct,
            curvature_strength=0.15
        )
        assert geometry.vertices.shape == (64, 64, 3), f"Wrong vertex shape for {ct.value}"
        assert geometry.normals.shape == (64, 64, 3), f"Wrong normals shape for {ct.value}"
        assert geometry.uv_coords.shape == (64, 64, 2), f"Wrong UV shape for {ct.value}"
        assert geometry.depth_map.shape == (64, 64), f"Wrong depth shape for {ct.value}"
        print(f"  [PASS] {ct.value} curvature")

    # Test double page
    geometry = generator.generate_double_page(
        curvature_strength=0.15,
        spine_position=0.5,
        asymmetry=0.1
    )
    assert geometry.is_double_page == True
    print(f"  [PASS] Double page generation")

    # Test random page
    geometry = generator.generate_random_page()
    assert geometry.vertices.shape[-1] == 3
    print(f"  [PASS] Random page generation")

    print("\nPageMeshGenerator: All tests passed!")
    return True


def test_texture_renderer():
    """Test the TextureRenderer component"""
    print("\n" + "=" * 60)
    print("Testing TextureRenderer")
    print("=" * 60)

    renderer = TextureRenderer(
        texture_size=(256, 256),
        random_seed=42
    )

    # Test different content types
    content_types = [
        ContentType.TEXT_ONLY,
        ContentType.TEXT_WITH_IMAGES,
        ContentType.MIXED_LAYOUT,
        ContentType.TECHNICAL,
        ContentType.HANDWRITTEN,
    ]

    for ct in content_types:
        texture = renderer.generate_flat_texture(
            content_type=ct,
            language='english'
        )
        assert texture.shape == (256, 256, 3), f"Wrong texture shape for {ct.value}"
        assert texture.dtype == np.uint8, f"Wrong dtype for {ct.value}"
        print(f"  [PASS] {ct.value} content")

    # Test Chinese text
    texture = renderer.generate_flat_texture(
        content_type=ContentType.TEXT_ONLY,
        language='chinese'
    )
    assert texture.shape == (256, 256, 3)
    print(f"  [PASS] Chinese text generation")

    # Test texture mapping
    mesh_gen = PageMeshGenerator(mesh_resolution=(32, 32), random_seed=42)
    geometry = mesh_gen.generate_single_page(
        curvature_type=CurvatureType.CYLINDRICAL,
        curvature_strength=0.1
    )

    warped = renderer.map_texture_to_surface(
        texture,
        geometry.vertices,
        geometry.uv_coords,
        output_size=(256, 256)
    )
    assert warped.shape == (256, 256, 3)
    print(f"  [PASS] Texture mapping to surface")

    # Test random texture
    texture = renderer.generate_random_texture(is_double_page=False)
    assert texture.shape[0] == 256
    print(f"  [PASS] Random texture generation")

    print("\nTextureRenderer: All tests passed!")
    return True


def test_lighting_engine():
    """Test the LightingEngine component"""
    print("\n" + "=" * 60)
    print("Testing LightingEngine")
    print("=" * 60)

    engine = LightingEngine(
        image_size=(256, 256),
        random_seed=42
    )

    # Create test surface
    mesh_gen = PageMeshGenerator(mesh_resolution=(64, 64), random_seed=42)
    geometry = mesh_gen.generate_single_page(
        curvature_type=CurvatureType.CYLINDRICAL,
        curvature_strength=0.15
    )

    # Test shading computation
    shading = engine.compute_shading(
        geometry.normals,
        geometry.vertices
    )
    assert shading.shape == (64, 64, 3)
    assert shading.min() >= 0
    print(f"  [PASS] Shading computation")

    # Test ambient occlusion
    ao = engine.compute_ambient_occlusion(geometry.depth_map)
    assert ao.shape == (64, 64)
    assert ao.min() >= 0 and ao.max() <= 1
    print(f"  [PASS] Ambient occlusion")

    # Test shading application
    test_texture = np.ones((64, 64, 3), dtype=np.uint8) * 200
    shaded = engine.apply_shading(test_texture, shading)
    assert shaded.shape == (64, 64, 3)
    assert shaded.dtype == np.uint8
    print(f"  [PASS] Shading application")

    # Test shadow addition
    with_shadows = engine.add_shadows(shaded, geometry.depth_map)
    assert with_shadows.shape == (64, 64, 3)
    print(f"  [PASS] Shadow addition")

    # Test full render
    rendered = engine.render_with_lighting(
        test_texture,
        geometry.normals,
        geometry.vertices,
        geometry.depth_map
    )
    assert rendered.shape == (64, 64, 3)
    print(f"  [PASS] Full lighting render")

    # Test random lighting
    engine.set_random_lighting()
    assert len(engine.lights) >= 1
    print(f"  [PASS] Random lighting setup")

    # Test light creation
    light = engine.create_light_from_angle(
        azimuth=0.5,
        elevation=0.7,
        intensity=0.8
    )
    assert light.light_type == LightType.DIRECTIONAL
    print(f"  [PASS] Light creation from angles")

    print("\nLightingEngine: All tests passed!")
    return True


def test_synthesis_pipeline():
    """Test the complete synthesis pipeline"""
    print("\n" + "=" * 60)
    print("Testing SynthesisDataGenerator Pipeline")
    print("=" * 60)

    config = SynthesisConfig(
        output_size=(256, 256),
        depth_map_size=(256, 256),
        mesh_resolution=(64, 64),
        texture_size=(512, 512),
        curvature_range=(0.05, 0.25),
        double_page_probability=0.4,
    )

    generator = SynthesisDataGenerator(config=config, random_seed=42)

    # Test single sample generation
    result = generator.generate_single_sample()
    assert result.image.shape == (256, 256, 3)
    assert result.depth_map.shape == (256, 256)
    assert 'curvature_type' in result.metadata
    print(f"  [PASS] Single sample generation")

    # Test forced single page
    result = generator.generate_single_sample(is_double_page=False)
    assert result.metadata['is_double_page'] == False
    print(f"  [PASS] Forced single page")

    # Test forced double page
    result = generator.generate_single_sample(is_double_page=True)
    assert result.metadata['is_double_page'] == True
    print(f"  [PASS] Forced double page")

    # Test specific curvature type
    result = generator.generate_single_sample(
        is_double_page=False,
        curvature_type=CurvatureType.WAVE
    )
    assert result.metadata['curvature_type'] == 'wave'
    print(f"  [PASS] Specific curvature type")

    # Test specific content type
    result = generator.generate_single_sample(
        content_type=ContentType.TECHNICAL
    )
    assert result.metadata['content_type'] == 'technical'
    print(f"  [PASS] Specific content type")

    print("\nSynthesisDataGenerator: All tests passed!")
    return True


def generate_test_samples(
    num_samples: int = 20,
    output_dir: str = None,
    save_visualizations: bool = True
) -> List[SynthesisResult]:
    """
    Generate test samples and optionally save them.

    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save outputs
        save_visualizations: Whether to save visualization images

    Returns:
        List of generated results
    """
    print("\n" + "=" * 60)
    print(f"Generating {num_samples} Test Samples")
    print("=" * 60)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'test_output'
    output_dir = Path(output_dir)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'depth_maps').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'flat_textures').mkdir(exist_ok=True)

    config = SynthesisConfig(
        output_size=(512, 512),
        depth_map_size=(512, 512),
        mesh_resolution=(128, 128),
        texture_size=(1024, 1024),
        curvature_range=(0.05, 0.3),
        double_page_probability=0.4,
        randomize_lighting=True,
        add_shadows=True,
        add_ambient_occlusion=True,
        add_noise=True,
        noise_level=0.015,
        add_blur=True,
        blur_probability=0.2,
    )

    generator = SynthesisDataGenerator(config=config, random_seed=42)

    results = []
    start_time = time.time()

    for i in range(num_samples):
        print(f"\rGenerating sample {i + 1}/{num_samples}...", end='', flush=True)

        try:
            result = generator.generate_single_sample()
            results.append(result)

            # Save image
            img_path = output_dir / 'images' / f'sample_{i:03d}.png'
            cv2.imwrite(str(img_path), result.image)

            # Save depth map (as 16-bit PNG)
            depth_path = output_dir / 'depth_maps' / f'sample_{i:03d}.png'
            cv2.imwrite(str(depth_path), result.depth_map)

            # Save flat texture
            flat_path = output_dir / 'flat_textures' / f'sample_{i:03d}.png'
            cv2.imwrite(str(flat_path), result.flat_texture)

            # Save visualization
            if save_visualizations:
                viz_path = output_dir / 'visualizations' / f'sample_{i:03d}.png'
                create_visualization(result, str(viz_path))

        except Exception as e:
            print(f"\nWarning: Failed to generate sample {i}: {e}")

    elapsed = time.time() - start_time
    print(f"\nGenerated {len(results)} samples in {elapsed:.2f} seconds")
    print(f"Average: {elapsed / num_samples:.3f} seconds per sample")
    print(f"Output saved to: {output_dir}")

    # Print sample statistics
    print("\nSample Statistics:")
    double_page_count = sum(1 for r in results if r.metadata.get('is_double_page', False))
    print(f"  Double page samples: {double_page_count}/{len(results)} ({100*double_page_count/len(results):.1f}%)")

    curvature_values = [r.metadata.get('curvature_strength', 0) for r in results]
    print(f"  Curvature range: {min(curvature_values):.3f} - {max(curvature_values):.3f}")
    print(f"  Mean curvature: {np.mean(curvature_values):.3f}")

    # Content type distribution
    content_types = [r.metadata.get('content_type', 'unknown') for r in results]
    print("  Content type distribution:")
    for ct in set(content_types):
        count = content_types.count(ct)
        print(f"    {ct}: {count} ({100*count/len(results):.1f}%)")

    return results


def run_all_tests():
    """Run all component tests"""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION FRAMEWORK - TEST SUITE")
    print("=" * 60)

    tests = [
        ("PageMeshGenerator", test_page_mesh_generator),
        ("TextureRenderer", test_texture_renderer),
        ("LightingEngine", test_lighting_engine),
        ("SynthesisDataGenerator", test_synthesis_pipeline),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    return all_passed


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test synthetic data generation framework"
    )
    parser.add_argument(
        '--test-only', '-t',
        action='store_true',
        help="Only run tests, don't generate samples"
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=20,
        help="Number of test samples to generate (default: 20)"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help="Output directory for test samples"
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help="Don't save visualization images"
    )

    args = parser.parse_args()

    # Run tests
    tests_passed = run_all_tests()

    if not tests_passed:
        print("\nSome tests failed. Fix issues before generating samples.")
        sys.exit(1)

    # Generate test samples
    if not args.test_only:
        generate_test_samples(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            save_visualizations=not args.no_viz
        )

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
