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
# JOINT CLASS
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
    Kinetic model of human body
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
        # Computing angles
        self.angles = Angles(positions, self.N_frames)
        # initialising body parts
        self._initialize_bodyparts()



    def _initialize_bodyparts(self):
        # Initialize limb dictionaries
        self._upper_arm = {}
        self._upper_leg = {}
        self._lower_leg = {}
        self._torso = {}
        self._upper_back = {}
        self._hip = {}
        self._forearm = {}
        self._thumb = {}
        self._index = {}
        self._pinky = {}
        self._foot = {}
        self._heel = {}
        N_frames = self.N_frames


        # > Bilateral body parts()
        for side in ["RIGHT", "LEFT"]:
            # Upper Arm
            self._upper_arm[side] = get_vector(
                f"{side}_SHOULDER", 
                f"{side}_ELBOW", 
                self.positions, 
                N_frames
            )
        
            # Forearm
            self._forearm[side] = get_vector(
                f"{side}_ELBOW", 
                f"{side}_WRIST", 
                self.positions,
                N_frames
            )

            # Upper Leg (Hip to Knee)
            self._upper_leg[side] = get_vector(
                f"{side}_HIP", 
                f"{side}_KNEE", 
                self.positions, 
                N_frames
            )
        
            # Lower Leg (Knee to Ankle)
            self._lower_leg[side] = get_vector(
                f"{side}_KNEE", 
                f"{side}_ANKLE", 
                self.positions, 
                N_frames
            )
        
            # Torso (Shoulder to Hip)
            self._torso[side] = get_vector(
                f"{side}_SHOULDER", 
                f"{side}_HIP", 
                self.positions, 
                N_frames
            )


            # ----- HAND ------
            # Thumb
            self._thumb[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_THUMB",
                self.positions,
                N_frames
            )
            # Index 
            self._index[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_INDEX",
                self.positions,
                N_frames
            )
            #Pinky
            self._pinky[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_PINKY",
                self.positions,
                N_frames
            )

            # ----- FOOT ------
            # Heel
            self._heel[side] = get_vector(
                f"{side}_ANKLE",
                f"{side}_HEEL",
                self.positions,
                N_frames
            )

            # Foot (Heel to Foot Index)
            self._foot[side] = get_vector(
                f"{side}_HEEL",
                f"{side}_FOOT_INDEX",
                self.positions,
                N_frames
            )


        # > Unilateral body parts
        # Upper back (left to right shoulder)
        self._upper_back = get_vector(
            "LEFT_SHOULDER", 
            "RIGHT_SHOULDER", 
            self.positions, 
            N_frames
        )

        # Hip 
        self._hip = get_vector(
            "LEFT_HIP",
            "RIGHT_HIP",
            self.positions,
            N_frames
        )




    def save(self, output_path:str) -> None:
        save_dict_to_json(
            dict_obj={"positions" : self.positions,
                      "meta" : self.meta},
                      save_to_path=output_path
        )


    ############################################################## 
    # Properties
    ##############################################################
    @property
    def UpperArmRight(self):
        return self._upper_arm["RIGHT"]

    @property
    def UpperArmLeft(self):
        return self._upper_arm["LEFT"]

    @property
    def ForearmRight(self):
        return self._forearm["RIGHT"]

    @property
    def ForearmLeft(self):
        return self._forearm["LEFT"]

    @property
    def UpperLegRight(self):
        return self._upper_leg["RIGHT"]

    @property
    def UpperLegLeft(self):
        return self._upper_leg["LEFT"]

    @property
    def LowerLegRight(self):
        return self._lower_leg["RIGHT"]

    @property
    def LowerLegLeft(self):
        return self._lower_leg["LEFT"]

    @property
    def TorsoRight(self):
        return self._torso["RIGHT"]

    @property
    def TorsoLeft(self):
        return self._torso["LEFT"]

    @property
    def ThumbLeft(self):
        return self._thumb["LEFT"]
    
    @property
    def ThumbRight(self):
        return self._thumb["RIGHT"]

    @property
    def IndexLeft(self):
        return self._index["LEFT"]
    
    @property
    def IndexRight(self):
        return self._index["RIGHT"]

    @property
    def PinkyLeft(self):
        return self._pinky["LEFT"]
    
    @property
    def PinkyRight(self):
        return self._pinky["RIGHT"]

    @property
    def UpperBack(self):
        return self._upper_back

    @property
    def Hip(self):
        return self._hip

    @property
    def HeelLeft(self):
        return self._heel["LEFT"]

    @property
    def HeelRight(self):
        return self._heel["RIGHT"]


    @property
    def FootLeft(self):
        return self._foot["LEFT"]

    @property
    def FootRight(self):
        return self._foot["RIGHT"]


if __name__ == "__main__":
    pass
