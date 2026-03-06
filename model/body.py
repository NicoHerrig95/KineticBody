import os
import sys
from abc import ABC, abstractmethod
import numpy as np 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.common import read_json
import time


LANDMARK_MAPPING = read_json(os.path.join(BASE_DIR, "POSE_landmark_mapping.json"))



def get_limb_from_landmarks(lm_a, lm_b, positions, N:int) -> list:
    return [(positions[lm_a][i], positions[lm_b][i]) for i in range(N)]




class Body(object):

    """  
    Kinetic model of human body
    """

    def __init__(self, positions:dict, N_frames:int):
        
        self.positions = positions
        # > computing limbs 
        # Upper Arm
        for side in ["RIGHT", "LEFT"]:
            self._upper_arm[side] = get_limb_from_landmarks(
                f"{side}_SHOULDER", 
                f"{side}_ELBOW", 
                positions, 
                N_frames
            )

    @property
    def UpperArmRight(self):
        return self._upper_arm["RIGHT"]

    @property
    def UpperArmLeft(self):
        return self._upper_arm["LEFT"]

    


    