"""
3D Synthetic Data Generation Framework for Document Dewarping

This module provides a complete pipeline for generating synthetic curved document
images with corresponding depth maps for training depth estimation neural networks.

Components:
- PageMeshGenerator: Generates 3D curved page surfaces (single/double page)
- TextureRenderer: Renders text and images onto curved surfaces
- LightingEngine: Simulates realistic lighting and shadows
- SynthesisDataGenerator: Main pipeline for generating training data

Author: Yuhan Xu
"""

from .page_mesh import PageMeshGenerator, CurvatureType
from .texture_renderer import TextureRenderer, ContentType
from .lighting_engine import LightingEngine, LightSource
from .synthesis_pipeline import SynthesisDataGenerator, SynthesisConfig

__all__ = [
    'PageMeshGenerator',
    'CurvatureType',
    'TextureRenderer',
    'ContentType',
    'LightingEngine',
    'LightSource',
    'SynthesisDataGenerator',
    'SynthesisConfig',
]
