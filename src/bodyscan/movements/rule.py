""" 
Movements
"""

import os
import sys
from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np 
import cv2
import time
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from bodyscan.utils.common import read_json, save_dict_to_json
from bodyscan.kinetics.body import KineticBody
from bodyscan.model.pose_estimation import PoseEstimator
from bodyscan.assets.visualization import visualize_video


""" 
Base Classes for movement
"""


class Rule(ABC):
    def __init__(self, 
                 body:KineticBody,
                 name:str
                 ):
        self.body = body
        self.name = name

    @abstractmethod
    def check(self) -> dict:
        """ 
        Checks kinetic body on a constraint. 
        Triggers recommendation if required.
        Returns a dictionary called result.
        """
        pass



    def __call__(self) -> str|None:
        result = self.check()
        return result


