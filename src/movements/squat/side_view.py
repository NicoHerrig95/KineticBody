import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from kinetics.body import KineticBody
from utils.common import read_yaml
from movements.movement import Rule

""" 
Barbell squat side view
"""

# Reading config 
CONFIG_PATH = "./config/body.yaml"
config = read_yaml(CONFIG_PATH)["visualization"]
THICKNESS_SCALER = config["thickness_scale"]
JOINT_SCALER = config["joint_radius_scale"]
BODY_COLOR = tuple(config["body_color"])
JOINT_COLOR = tuple(config["joint_color"])



# Rule 1
# Going lower than 90 degrees on squat
# If not reached, recommend increasing hip mobility

class KneeBelow90Degrees(Rule):

    def __init__(self, 
                 body:KineticBody,
                 side = None
                 ):
        super().__init__(body=body)
        if side is not None and side not in ["Left", "Right"]:
            raise ValueError("Side must be either left or right")
        self.side = ["Left", "Right"] if side == None else [side]


    def check(self):
        result = {}
        if self.side is None:
            sides = ["Left", "Right"]
        else:
            sides = self.side
        for side in sides:
            angles = getattr(self.body.angles, f"Knee{side}")
            # checks if knee angles are below 90 degrees
            checkup = [(int(x) < 90) for x in angles]
            result[side] = any(checkup) # checks if angle is below 90 at any point
        return result
    
    def get_recommendation(self, result):
        output = {"analysis" : [], "recommendation" : None}
        for side in self.side:
            if not result[side]:
                output["analysis"].append(f"Through the movement, {side} knee angle is never below 90 degrees.")
        if len(output["analysis"]) > 0:
            output["recommendation"] = "Increasing hip mobility is recommended."
        return output








