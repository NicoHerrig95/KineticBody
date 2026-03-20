import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
from utils.common import read_json, save_dict_to_json
import time
from src.kinetics.bodyparts.base import Bodypart, get_angle, get_vector
from dotenv import load_dotenv
# Load .env file
load_dotenv()
LANDMARK_MAPPING = read_json(os.getenv("POSE_LANDMARK_MAPPING_PATH"))


################################################################################
# ANGLE CLASS
################################################################################

class Angles(Bodypart):
    def __init__(self, positions:dict, N_frames):
        super().__init__()
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodypart()

    def _initialize_bodypart(self):

        N_frames = self.N_frames
        BILATERAL_ANGLES = {
            "Knee" : ("HIP", "KNEE", "ANKLE"),
            "Elbow" : ("SHOULDER", "ELBOW", "WRIST")
            }

        for side in ["RIGHT", "LEFT"]:
            # Bilateral Angles
            for angle_name, (start, center, end) in BILATERAL_ANGLES.items():
                self._objects.setdefault(angle_name, {})[side] = [get_angle(
                    coord_a = self.positions[f"{side}_{start}"][i],
                    coord_b = self.positions[f"{side}_{center}"][i],
                    coord_c = self.positions[f"{side}_{end}"][i]
                ) for i in range(self.N_frames)]