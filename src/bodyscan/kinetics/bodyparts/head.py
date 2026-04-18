import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
from bodyscan.utils.common import read_json, save_dict_to_json
import time
from bodyscan.kinetics.bodyparts.base import Bodypart, get_angle, get_vector
from dotenv import load_dotenv
# Load .env file
load_dotenv()
LANDMARK_MAPPING = read_json(os.getenv("POSE_LANDMARK_MAPPING_PATH"))



class Head(Bodypart):

    def __init__(self, positions: dict, N_frames: int):
        super().__init__()
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodypart()
    
    def _initialize_bodypart(self):
        # Nose
        self._objects.update({
            "Nose" :  self.positions["NOSE"]
            })
        # Ears & Eyes (midpoint)
        for side in ["LEFT", "RIGHT"]:
            self._objects.setdefault("Ear", {})[side] = self.positions[f"{side}_EAR"]
            self._objects.setdefault("Eye", {})[side] = self.positions[f"{side}_EYE"]



