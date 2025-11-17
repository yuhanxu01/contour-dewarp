#!/usr/bin/env python3
"""
使用示例 - 展示如何使用document-dewarp库

Usage examples for the document-dewarp library.
"""

import cv2
import numpy as np
from document_dewarp import DocumentDewarp
from config import (
    DewarpConfig,
    DEFAULT_CONFIG,
    SINGLE_PAGE_CONFIG,
    HIGH_QUALITY_CONFIG,
    FAST_CONFIG
)


def example_1_basic_image():
    """示例1: 处理单张图片（基础用法）"""
    print("Example 1: Basic image processing")
    print("-" * 50)

    # 创建去畸变器
    dewarp = DocumentDewarp()

    # 处理图像
    result = dewarp.process_image('input.jpg')

    # 保存结果
    cv2.imwrite('output.jpg', result)

    print("✓ Image processed and saved to output.jpg\n")


def example_2_single_page():
    """示例2: 处理单页文档"""
    print("Example 2: Single page mode")
    print("-" * 50)

    # 使用单页配置
    dewarp = DocumentDewarp(config=SINGLE_PAGE_CONFIG)

    # 处理图像
    result = dewarp.process_image('single_page.jpg')

    # 保存结果
    cv2.imwrite('single_page_output.jpg', result)

    print("✓ Single page processed\n")


def example_3_double_page():
    """示例3: 处理双页文档（书籍）"""
    print("Example 3: Double page mode")
    print("-" * 50)

    # 使用默认配置（双页模式）
    dewarp = DocumentDewarp(config=DEFAULT_CONFIG)

    # 处理图像
    result = dewarp.process_image('book_spread.jpg')

    # 保存结果
    cv2.imwrite('book_spread_output.jpg', result)

    print("✓ Double page processed\n")


def example_4_custom_config():
    """示例4: 自定义配置"""
    print("Example 4: Custom configuration")
    print("-" * 50)

    # 创建自定义配置
    custom_config = DewarpConfig(
        page_mode='single',
        target_width=5000,
        target_height=3500,
        poly_degree=8,
        debug=True
    )

    # 创建去畸变器
    dewarp = DocumentDewarp(config=custom_config)

    # 处理图像
    result = dewarp.process_image('custom_input.jpg')

    # 保存结果
    cv2.imwrite('custom_output.jpg', result)

    print("✓ Custom configuration applied\n")


def example_5_pdf():
    """示例5: 处理PDF文件"""
    print("Example 5: PDF processing")
    print("-" * 50)

    # 创建去畸变器
    dewarp = DocumentDewarp()

    # 定义进度回调
    def progress(current, total):
        print(f"Processing page {current}/{total}")

    # 处理PDF
    dewarp.process_pdf(
        'input.pdf',
        'output.pdf',
        progress_callback=progress
    )

    print("✓ PDF processed\n")


def example_6_batch():
    """示例6: 批量处理图片"""
    print("Example 6: Batch processing")
    print("-" * 50)

    # 创建去畸变器
    dewarp = DocumentDewarp()

    # 批量处理
    input_images = ['img1.jpg', 'img2.jpg', 'img3.jpg']

    output_paths = dewarp.process_images_batch(
        input_images,
        output_dir='output_batch/',
        output_format='png'
    )

    print(f"✓ Processed {len(output_paths)} images\n")


def example_7_numpy_array():
    """示例7: 直接处理NumPy数组"""
    print("Example 7: Processing NumPy arrays")
    print("-" * 50)

    # 读取图像为NumPy数组
    image = cv2.imread('input.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建去畸变器
    dewarp = DocumentDewarp()

    # 处理数组
    result = dewarp.process_image(image, return_binary=True)

    # 保存结果
    cv2.imwrite('array_output.jpg', result)

    print("✓ NumPy array processed\n")


def example_8_high_quality():
    """示例8: 高质量模式"""
    print("Example 8: High quality mode")
    print("-" * 50)

    # 使用高质量配置
    dewarp = DocumentDewarp(config=HIGH_QUALITY_CONFIG)

    # 处理图像
    result = dewarp.process_image('high_res_input.jpg')

    # 保存结果
    cv2.imwrite('high_quality_output.jpg', result)

    print("✓ High quality processing complete\n")


def example_9_fast_mode():
    """示例9: 快速模式"""
    print("Example 9: Fast mode")
    print("-" * 50)

    # 使用快速配置
    dewarp = DocumentDewarp(config=FAST_CONFIG)

    # 处理图像
    result = dewarp.process_image('input.jpg')

    # 保存结果
    cv2.imwrite('fast_output.jpg', result)

    print("✓ Fast processing complete\n")


def example_10_grayscale_output():
    """示例10: 获取灰度输出（非二值化）"""
    print("Example 10: Grayscale output")
    print("-" * 50)

    # 创建去畸变器
    dewarp = DocumentDewarp()

    # 处理图像（返回灰度图而非二值图）
    result = dewarp.process_image('input.jpg', return_binary=False)

    # 保存结果
    cv2.imwrite('grayscale_output.jpg', result)

    print("✓ Grayscale output saved\n")


def main():
    """运行所有示例"""
    print("\n" + "="*50)
    print("Document Dewarp - Usage Examples")
    print("="*50 + "\n")

    examples = [
        ("Basic Image Processing", example_1_basic_image),
        ("Single Page Mode", example_2_single_page),
        ("Double Page Mode", example_3_double_page),
        ("Custom Configuration", example_4_custom_config),
        ("PDF Processing", example_5_pdf),
        ("Batch Processing", example_6_batch),
        ("NumPy Array Input", example_7_numpy_array),
        ("High Quality Mode", example_8_high_quality),
        ("Fast Mode", example_9_fast_mode),
        ("Grayscale Output", example_10_grayscale_output),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nTo run a specific example, uncomment it in the code.")
    print("\nNote: Make sure to provide appropriate input files.\n")

    # 取消注释以运行特定示例
    # example_1_basic_image()
    # example_2_single_page()
    # example_3_double_page()
    # example_4_custom_config()
    # example_5_pdf()
    # example_6_batch()
    # example_7_numpy_array()
    # example_8_high_quality()
    # example_9_fast_mode()
    # example_10_grayscale_output()


if __name__ == '__main__':
    main()
