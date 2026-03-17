"""   
KineticBody
-> The KineticBody model gives information about coordinates of limbs and joints.
"""



import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.common import read_json, save_dict_to_json
import time
LANDMARK_MAPPING = read_json(os.path.join(BASE_DIR, "POSE_landmark_mapping.json"))


################################################################################
# HELPERS
################################################################################

def get_vector(lm_a, lm_b, positions, N:int) -> list:
    """ 
    Gets a vector from positions by definition of two landmark keys. 
    """
    return [(positions[lm_a][i], positions[lm_b][i]) for i in range(N)]



def get_angle(coord_a:tuple, coord_b:tuple, coord_c:tuple):
    """
    Computes angle from three coordinates (x,y).
    NOTE: Returns the angle for lm_b!
    """

    a = np.array(coord_a)
    b = np.array(coord_b)
    c = np.array(coord_c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle


################################################################################
# LIMBS CLASS
################################################################################

class Limbs:
    def __init__(self, positions: dict, N_frames: int):
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodyparts()

    def _initialize_bodyparts(self):
        N_frames = self.N_frames
        BILATERAL_LIMBS = {
            "UpperArm": ("SHOULDER", "ELBOW"),
            "Forearm": ("ELBOW", "WRIST"),
            "UpperLeg": ("HIP", "KNEE"),
            "LowerLeg": ("KNEE", "ANKLE"),
            "Torso": ("SHOULDER", "HIP"),
            "Thumb": ("WRIST", "THUMB"),
            "Index": ("WRIST", "INDEX"),
            "Pinky": ("WRIST", "PINKY"),
            "Heel": ("ANKLE", "HEEL"),
            "Foot": ("HEEL", "FOOT_INDEX")
        }

        UNILATERAL_LIMBS = {
            "UpperBack" : ("LEFT_SHOULDER","RIGHT_SHOULDER"),
            "Hip" : ("LEFT_HIP", "RIGHT_HIP")
        }

        self._limbs = {}

        for side in ["RIGHT", "LEFT"]:
            # Bilateral Limbs
            for limb_name, (start, end) in BILATERAL_LIMBS.items():
                self._limbs.setdefault(limb_name, {})[side] = get_vector(
                    f"{side}_{start}",
                    f"{side}_{end}",
                    self.positions,
                    N_frames
                )

        # Unilateral limbs
        self._limbs["UpperBack"] = get_vector(UNILATERAL_LIMBS["UpperBack"][0], UNILATERAL_LIMBS["UpperBack"][1], self.positions, N_frames)
        self._limbs["Hip"] = get_vector(UNILATERAL_LIMBS["Hip"][0], UNILATERAL_LIMBS["Hip"][1], self.positions, N_frames)

    def __getattr__(self, name):
        """
        Dynamically generate properties like UpperArmRight, FootLeft, etc.
        """
        # Split the name into limb and side (assumes side is "Left" or "Right")
        for side in ["Right", "Left"]:
            if name.endswith(side):
                limb_name = name[:-len(side)]
                side_upper = side.upper()
                if limb_name in self._limbs:
                    return self._limbs[limb_name][side_upper]
        # Check unilateral limbs
        if name in self._limbs:
            return self._limbs[name]
        raise AttributeError(f"{name} not found in Limbs")




################################################################################
# JOINTS CLASS
################################################################################
class Joints:
    """ 
    Stores the body's joints' coordinates.
    Directly derived from pose predictions.
    """
    def __init__(self, positions: dict):
        # Arms
        self.ElbowLeft = positions["LEFT_ELBOW"]
        self.ElbowRight = positions["RIGHT_ELBOW"]
        self.WristLeft = positions["LEFT_WRIST"]
        self.WristRight = positions["RIGHT_WRIST"]
        self.PinkyLeft = positions["LEFT_PINKY"]
        self.PinkyRight = positions["RIGHT_PINKY"]
        self.IndexLeft = positions["LEFT_INDEX"]
        self.IndexRight = positions["RIGHT_INDEX"]
        self.ThumbLeft = positions["LEFT_THUMB"]
        self.ThumbRight = positions["RIGHT_THUMB"]

        # Torso / Hips
        self.HipLeft = positions["LEFT_HIP"]
        self.HipRight = positions["RIGHT_HIP"]
        self.ShoulderLeft = positions["LEFT_SHOULDER"]
        self.ShoulderRight = positions["RIGHT_SHOULDER"]

        # Legs
        self.KneeLeft = positions["LEFT_KNEE"]
        self.KneeRight = positions["RIGHT_KNEE"]
        self.AnkleLeft = positions["LEFT_ANKLE"]
        self.AnkleRight = positions["RIGHT_ANKLE"]
        self.HeelLeft = positions["LEFT_HEEL"]
        self.HeelRight = positions["RIGHT_HEEL"]
        self.FootIndexLeft = positions["LEFT_FOOT_INDEX"]
        self.FootIndexRight = positions["RIGHT_FOOT_INDEX"]


################################################################################
# ANGLE CLASS
################################################################################

class Angles:
    def __init__(self, positions:dict, N_frames):
        
        self.KneeRight = [get_angle(
            coord_a = positions["RIGHT_HIP"][i],
            coord_b = positions["RIGHT_KNEE"][i],
            coord_c = positions["RIGHT_ANKLE"][i]
        ) for i in range(N_frames)]

        self.KneeLeft = [get_angle(
            coord_a = positions["LEFT_HIP"][i],
            coord_b = positions["LEFT_KNEE"][i],
            coord_c = positions["LEFT_ANKLE"][i]
        ) for i in range(N_frames)]
         





################################################################################
# MAIN CLASS (KINETIC BODY)
################################################################################

class KineticBody(object):
    """  
    Kinetic model of human body. Contains:
    - Limbs
    - Joints
    - Angles
    """

    def __init__(
        self,
        positions: dict, # positions dictionary 
        metadata: dict,  
        ):

        # unpacking metadata
        self.meta = metadata
        self.N_frames = metadata["frame_count"]
        self.mode = metadata["mode"]
        self.positions = positions # position of joints from Pose Estimation
        # Setting joints
        self.joints = Joints(positions)
        #Setting Limbs
        self.limbs = Limbs(positions, self.N_frames)
        # Computing angles
        self.angles = Angles(positions, self.N_frames)







if __name__ == "__main__":
    pass
