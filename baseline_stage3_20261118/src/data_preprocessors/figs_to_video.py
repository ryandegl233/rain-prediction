import os
import re
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm


def create_video_from_rain_thumbnails(
    input_dir: str,
    output_path: Union[str, Path],
    fps: int = 2,
    resize_factor: float = 1.0,
    codec: str = "mp4v",
    quality: int = 23,
):
    """
    将 rain_thumbnail 图像按时间顺序制作成视频

    Parameters:
    input_dir (str): 包含 rain_thumbnail 图像的目录路径
    output_path (Union[str, Path]): 输出视频文件的路径
    fps (int): 视频帧率，默认为 2fps
    resize_factor (float): 图像缩放因子，0.0-1.0，默认为 1.0（不缩放）
    codec (str): 视频编码器，可选 'mp4v' (H.264), 'avc1' (H.264), 'MJPG' (Motion JPEG)
    quality (int): 视频质量，1-50，数字越小质量越高，文件越大。23 是默认值
    """
    # 查找所有 modalities_*.jpg 文件
    input_path = Path(input_dir)
    thumbnail_files = list(input_path.glob("modalities_*.jpg"))

    if not thumbnail_files:
        print(f"在目录 {input_dir} 中未找到任何 rain_thumbnail 图像")
        return

    print(f"找到 {len(thumbnail_files)} 个图像文件")

    # 从文件名中提取时间戳并排序
    def extract_timestamp(filename):
        # 文件名格式: modalities_YYYYMMDD_HHMMSS.jpg
        match = re.search(r"modalities_(\d{8}_\d{6})\.jpg", filename.name)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return None

    # 过滤掉无法解析时间戳的文件并排序
    timestamped_files = [(extract_timestamp(f), f) for f in thumbnail_files]
    timestamped_files = [(ts, f) for ts, f in timestamped_files if ts is not None]
    timestamped_files.sort(key=lambda x: x[0])  # 按时间戳排序

    if not timestamped_files:
        print("未能从文件名中提取有效的时间戳")
        return

    print(f"按时间顺序排序后有 {len(timestamped_files)} 个有效图像文件")

    # 保存为视频
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # 使用 OpenCV 创建视频
    create_video_with_opencv(
        timestamped_files, output_path_obj, fps, resize_factor, codec, quality
    )

    print(f"已成功创建视频文件: {output_path_obj}")


def create_video_with_opencv(
    timestamped_files, output_path, fps, resize_factor, codec, quality
):
    """使用 OpenCV 创建视频"""
    print("使用 OpenCV 库生成视频...")

    # 获取第一帧图像以确定视频尺寸
    first_img = cv2.imread(str(timestamped_files[0][1]))
    if first_img is None:
        print("无法读取第一帧图像")
        return

    # 调整图像大小
    if resize_factor != 1.0:
        height, width = first_img.shape[:2]
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        first_img = cv2.resize(first_img, (new_width, new_height))

    height, width = first_img.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter.fourcc(*codec)
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("无法创建视频写入器")
        return

    # 设置编码参数（如果支持）
    if codec in ["mp4v", "avc1"]:
        # H.264 编码参数
        video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)

    # 写入第一帧
    video_writer.write(first_img)

    # 写入后续帧
    for _, file_path in tqdm(timestamped_files[1:], desc="处理视频帧"):
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"无法读取图像文件: {file_path}")
            continue

        # 调整图像大小
        if resize_factor != 1.0:
            height, width = img.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            img = cv2.resize(img, (new_width, new_height))

        # 确保图像尺寸正确
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        video_writer.write(img)

    # 释放资源
    video_writer.release()
    print(f"视频生成完成，共处理 {len(timestamped_files)} 帧")


def main():
    # 默认参数
    input_dir = (
        "data_original/zihan_processed/interval_30/202305/rain_thumbnail"  # 示例路径
    )
    output_path = "data_original/zihan_processed/interval_30/202305/rain_animation.mp4"  # 示例输出路径

    # 可以通过命令行参数或环境变量修改这些路径
    import argparse

    parser = argparse.ArgumentParser(description="将 rain_thumbnail 图像制作成视频")
    parser.add_argument("--input_dir", default=input_dir, help="输入目录路径")
    parser.add_argument("--output_path", default=output_path, help="输出视频文件路径")
    parser.add_argument("--fps", type=int, default=4, help="视频帧率，默认为 4fps")
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=1.0,
        help="图像缩放因子，0.0-1.0，默认为 1.0（不缩放）",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        choices=["mp4v", "avc1", "MJPG", "hev1", "hvc1"],
        help="视频编码器，可选 'mp4v' (H.264), 'avc1' (H.264), 'MJPG' (Motion JPEG)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=23,
        help="视频质量，1-50，数字越小质量越高，文件越大。23 是默认值",
    )

    args = parser.parse_args()

    create_video_from_rain_thumbnails(
        args.input_dir,
        args.output_path,
        args.fps,
        args.resize_factor,
        args.codec,
        args.quality,
    )

    """
    ffmpeg -framerate 6 -pattern_type glob -i \
        "data_original/zihan_processed/interval_30/202305/rain_thumbnail/modalities_*.jpg" \
        -c:v libx265 -crf 28 -vf "scale=2424:954" output.mp4
    """


if __name__ == "__main__":
    main()
