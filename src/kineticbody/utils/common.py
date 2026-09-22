import json
import os
import numpy as np
import yaml
from pathlib import Path
from typing import Union
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import subprocess


def read_json(path:str) -> dict:
    """ 
    Reads in a JSON file
    """
    with open(path, "r") as file:
        return json.load(file)
    
def save_dict_to_json(dict_obj: dict, save_to_path: str):
    with open(save_to_path, "w") as json_file:
        json.dump(dict_obj, json_file, indent=4)


def read_yaml(path:str):
    with open(path) as f:
        return yaml.safe_load(f)
    



def mov_to_mp4(input_path: str, output_path: str = None, overwrite: bool = True):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp4"

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",  # overwrite or not
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    return output_path



def video_to_frames(
    video_path: Union[str, Path]
) -> list[Image.Image]:
    """
    Extract all frames from a video as PIL images.

    Args:
        video_path: Path to the input video.

    Returns:
        List of video frames as PIL.Image objects.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # OpenCV: BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # NumPy array -> PIL image
            frame_pil = Image.fromarray(frame_rgb)

            frames.append(frame_pil)

    finally:
        cap.release()

    return frames


def get_modality(input_path: str) -> str:
    """
    Determine modality based on file extension.

    Returns:
        "video" for video files
        "image" for image files

    Raises:
        ValueError if the file type is unsupported.
    """

    video_suffix_list = [".mp4", ".mov"]
    image_suffix_list = [".png", ".jpg", ".jpeg"]

    suffix = Path(input_path).suffix.lower()

    if suffix in video_suffix_list:
        return "video"

    if suffix in image_suffix_list:
        return "image"

    raise ValueError(f"Unsupported file type: {suffix}")