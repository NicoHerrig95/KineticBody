import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from kineticbody.kinetics.body import KineticBody
from kineticbody.utils.common import read_yaml
from pathlib import Path


####################################################################
# Helper functions 
####################################################################
def get_pixel_coordinates(coordinates:tuple, w:int, h:int) -> tuple:
    x = coordinates[0]
    y = coordinates[1]
    return (int(x * w), int(y * h))


def scale_line_thickness(scale:float|int, width:int, height:int):
    """ 
    Scaling line thicknes proportional to image size
    """
    return max(1, int(min(height, width) / 200)) * scale


def make_writer(path, fps, width, height):
    print(path)
    candidates = [
        ("mp4v", path),                        # .mp4
        ("MJPG", path.replace(".mp4", ".avi")),# .avi
        ("XVID", path.replace(".mp4", ".avi")),# .avi
    ]

    for codec, out in candidates:
        writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc(*codec), # type: ignore
            fps,
            (width, height),
        )
        if writer.isOpened():
            print(f"Using codec={codec}, output={out}")
            return writer, out

    raise RuntimeError("Could not open VideoWriter with mp4v, MJPG, or XVID.")


####################################################################
# Drawing functions
# > in-place manipulation of arrays
####################################################################

def draw_limb(frame:np.ndarray, coords:tuple, thickness_scaler:int, color:tuple):
    h, w = frame.shape[:2]
    p1 = get_pixel_coordinates(coords[0], w, h)
    p2 = get_pixel_coordinates(coords[1], w, h)
    line_thickness = scale_line_thickness(
        scale = thickness_scaler,
        width=w,
        height=h
    )
    cv2.line(frame, p1, p2, color, line_thickness)



def draw_joint_cirlce(
        frame:np.ndarray, 
        center:tuple, 
        radius:float, 
        thickness_scaler:int, 
        color:tuple
        ):

    """ 
    Drawing joint circles.
    """
    h, w = frame.shape[:2]
    center_normalized = get_pixel_coordinates(center, w, h)
    radius_normalized = scale_line_thickness(radius, w, h)
    thickness_normalized = scale_line_thickness(thickness_scaler, w, h)
    cv2.circle(frame, center_normalized, radius_normalized, color, thickness_normalized)
    


def draw_angle(
        frame:np.ndarray, 
        value:float, # angle value 
        org:tuple, # bottom left coordinate of the text
        text:Optional[str] = ""
        ):

    h, w = frame.shape[:2]
    text = f"{text}{int(value)} deg."
    org = get_pixel_coordinates(org, w, h)
    text_position = (org[0] + 10, org[1] + 10)
    cv2.putText(
        frame,
        text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        2
    )
