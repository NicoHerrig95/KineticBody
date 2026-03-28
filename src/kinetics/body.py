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
from utils.common import read_json, save_dict_to_json
import time
from src.kinetics.bodyparts.base import Bodypart, get_angle, get_vector
from dotenv import load_dotenv
from kinetics.bodyparts.angles import Angles
from kinetics.bodyparts.joints import Joints
from kinetics.bodyparts.limbs import Limbs


# Load .env file
load_dotenv()
LANDMARK_MAPPING = read_json(os.getenv("POSE_LANDMARK_MAPPING_PATH"))




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
