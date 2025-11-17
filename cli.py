#!/usr/bin/env python3
"""
命令行接口 - 提供命令行工具

Command-line interface for document dewarping.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from config import (
    DewarpConfig,
    DEFAULT_CONFIG,
    SINGLE_PAGE_CONFIG,
    HIGH_QUALITY_CONFIG,
    FAST_CONFIG
)
from document_dewarp import DocumentDewarp


def create_config(args: argparse.Namespace) -> DewarpConfig:
    """
    根据命令行参数创建配置

    Create configuration from command-line arguments.

    Args:
        args: 命令行参数

    Returns:
        配置对象
    """
    # 选择基础配置
    if args.preset == 'single':
        config = SINGLE_PAGE_CONFIG
    elif args.preset == 'high_quality':
        config = HIGH_QUALITY_CONFIG
    elif args.preset == 'fast':
        config = FAST_CONFIG
    else:
        config = DEFAULT_CONFIG

    # 覆盖配置
    if args.page_mode:
        config.page_mode = args.page_mode

    if args.debug:
        config.debug = True

    return config


def process_image_file(
    input_path: str,
    output_path: str,
    config: DewarpConfig
) -> None:
    """
    处理单张图片文件

    Process a single image file.

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        config: 配置对象
    """
    print(f"Processing image: {input_path}")

    # 创建去畸变器
    dewarp = DocumentDewarp(config)

    # 处理图像
    result = dewarp.process_image(input_path)

    # 保存结果
    cv2.imwrite(output_path, result)

    print(f"Saved result to: {output_path}")


def process_pdf_file(
    input_path: str,
    output_path: str,
    config: DewarpConfig
) -> None:
    """
    处理PDF文件

    Process a PDF file.

    Args:
        input_path: 输入PDF路径
        output_path: 输出PDF路径
        config: 配置对象
    """
    print(f"Processing PDF: {input_path}")

    # 创建去畸变器
    dewarp = DocumentDewarp(config)

    # 进度条
    pbar = None

    def progress_callback(current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Processing pages")
        pbar.update(1)

    # 处理PDF
    dewarp.process_pdf(input_path, output_path, progress_callback)

    if pbar:
        pbar.close()

    print(f"Saved result to: {output_path}")


def process_batch(
    input_dir: str,
    output_dir: str,
    config: DewarpConfig,
    pattern: str = '*'
) -> None:
    """
    批量处理图片

    Batch process images.

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config: 配置对象
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 收集所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"{pattern}{ext}"))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s)")

    # 创建去畸变器
    dewarp = DocumentDewarp(config)

    # 处理每张图片
    for img_file in tqdm(image_files, desc="Processing images"):
        # 处理
        result = dewarp.process_image(str(img_file))

        # 生成输出路径
        output_file = output_path / f"{img_file.stem}_dewarped{img_file.suffix}"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 保存
        cv2.imwrite(str(output_file), result)

    print(f"Saved results to: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Document Dewarping Tool - Flatten curved document images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python cli.py image input.jpg output.jpg

  # Process a PDF
  python cli.py pdf input.pdf output.pdf

  # Process with single page mode
  python cli.py image input.jpg output.jpg --page-mode single

  # Batch process images in a directory
  python cli.py batch input_dir/ output_dir/

  # Use preset configurations
  python cli.py image input.jpg output.jpg --preset high_quality
        """
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # 图片处理命令
    image_parser = subparsers.add_parser('image', help='Process a single image')
    image_parser.add_argument('input', help='Input image path')
    image_parser.add_argument('output', help='Output image path')

    # PDF处理命令
    pdf_parser = subparsers.add_parser('pdf', help='Process a PDF file')
    pdf_parser.add_argument('input', help='Input PDF path')
    pdf_parser.add_argument('output', help='Output PDF path')

    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='Batch process images')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--pattern', default='*', help='File pattern (default: *)')

    # 共同参数
    for p in [image_parser, pdf_parser, batch_parser]:
        p.add_argument(
            '--page-mode',
            choices=['single', 'double'],
            help='Page mode: single or double page'
        )
        p.add_argument(
            '--preset',
            choices=['default', 'single', 'high_quality', 'fast'],
            default='default',
            help='Configuration preset'
        )
        p.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode'
        )

    args = parser.parse_args()

    # 检查命令
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 创建配置
    config = create_config(args)

    try:
        # 执行命令
        if args.command == 'image':
            process_image_file(args.input, args.output, config)
        elif args.command == 'pdf':
            process_pdf_file(args.input, args.output, config)
        elif args.command == 'batch':
            process_batch(args.input_dir, args.output_dir, config, args.pattern)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if config.debug:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
