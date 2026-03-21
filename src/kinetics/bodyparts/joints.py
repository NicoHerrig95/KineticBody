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
# JOINTS CLASS
################################################################################
class Joints(Bodypart):
    """Stores the body's joints' coordinates."""

    def __init__(self, positions: dict):
        super().__init__()
        self.positions = positions
        self._initialize_bodypart()

    def _initialize_bodypart(self):

        BILATERAL_JOINTS = {
            'Elbow': 'ELBOW',
            'Wrist': 'WRIST',
            'Pinky': 'PINKY',
            'Index': 'INDEX',
            'Thumb': 'THUMB',
            'Hip': 'HIP',
            'Shoulder': 'SHOULDER',
            'Knee': 'KNEE',
            'Ankle': 'ANKLE',
            'Heel': 'HEEL',
            'FootIndex': 'FOOT_INDEX'
        }

        for side in ["RIGHT", "LEFT"]:
            for joint_name, pose_marker_name in BILATERAL_JOINTS.items():
                full_name = f"{side}_{pose_marker_name}"
                self._objects.setdefault(joint_name, {})[side] = self.positions[full_name]
