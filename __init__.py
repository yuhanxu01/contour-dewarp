"""
Contour-Dewarp - 文档去畸变工具

基于轮廓估计曲面细分的文档图像平整化工具。

Document dewarping tool based on contour estimation and surface subdivision.
"""

__version__ = '2.0.0'
__author__ = 'Yuhan Xu'
__email__ = 'yuhanx5@uw.edu'

from .document_dewarp import DocumentDewarp
from .config import (
    DewarpConfig,
    DEFAULT_CONFIG,
    SINGLE_PAGE_CONFIG,
    HIGH_QUALITY_CONFIG,
    FAST_CONFIG
)

__all__ = [
    'DocumentDewarp',
    'DewarpConfig',
    'DEFAULT_CONFIG',
    'SINGLE_PAGE_CONFIG',
    'HIGH_QUALITY_CONFIG',
    'FAST_CONFIG',
]
