import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2


################################################################################
# HELPERS
################################################################################

def get_vector(lm_a, lm_b, positions, N:int) -> list:
    """ 
    Gets a vector from positions by definition of two landmark keys. 
    """
    # print(N)
    # print(len(positions[lm_a]))
    # print(len(positions[lm_b]))
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
# BODYPARTS CLASS (Baseclass)
################################################################################


class Bodypart(ABC):

    def __init__(self):
        self._objects = {}  # storage

    @abstractmethod
    def _initialize_bodypart(self):
        pass 

    def __getattr__(self, name):
        """
        Dynamically generate properties.
        Bilateral parts stored as self._objects[name][side] (e.g., self._objects["Shoulder"]["RIGHT"])
        Unilateral parts stored as self._objects[name] (e.g., self._objects["Nose"])
        """
        for side in ["Right", "Left"]:
            if name.endswith(side):
                object_name = name[:-len(side)]
                side_upper = side.upper() 
                if object_name in self._objects:
                    return self._objects[object_name][side_upper]

        # Check unilateral parts
        if name in self._objects:
            return self._objects[name]

        raise AttributeError(f"{name} not found in self._objects dictionary. Found: {self._objects}")
