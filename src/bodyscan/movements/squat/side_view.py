import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from bodyscan.kinetics.body import KineticBody
from bodyscan.utils.common import read_yaml, read_json
from bodyscan.movements.rule import Rule
from bodyscan.model.pose_estimation import PoseEstimator
from bodyscan.model.proc.filtering import SavGol
from bodyscan.utils.movement import check_joint_angle_range
from dotenv import load_dotenv
load_dotenv()






""" 
Barbell squat side view
"""


ANGLE_CONSTRAINTS = read_json("src/bodyscan/movements/squat/constraints/joint_angle_constraints.json")

class KneeAngle(Rule):

    def __init__(self, 
                 body:KineticBody,
                 name = "SquatKneeAngle",
                 side = None
                 ):

        """  
        Checks knee angle compliance with given biomechanical constraints.
        """
        super().__init__(body=body, name=name)
        self.angle_constraints = ANGLE_CONSTRAINTS["knee"]
        if side is not None and side not in ["Left", "Right"]:
            raise ValueError("Side must be either left or right")
        self.side = ["Left", "Right"] if side == None else [side]


    def check(self):
        result = {}
        if self.side is None:
            # testing both sides
            sides = ["Left", "Right"]
        else:
            sides = self.side
        for side in sides:
            angles = getattr(self.body.angles, f"Knee{side}")
            result[side] = check_joint_angle_range(angles, self.angle_constraints)
        return result


class HipAngle(Rule):

    def __init__(self, 
                 body:KineticBody,
                 name = "SquatHipAngle",
                 side = None
                 ):

        """  
        Checks knee angle compliance with given biomechanical constraints.
        """
        super().__init__(body=body, name = name)
        self.angle_constraints = ANGLE_CONSTRAINTS["hip"]
        if side is not None and side not in ["Left", "Right"]:
            raise ValueError("Side must be either left or right")
        self.side = ["Left", "Right"] if side == None else [side]


    def check(self):
        result = {}
        if self.side is None:
            # testing both sides
            sides = ["Left", "Right"]
        else:
            sides = self.side
        for side in sides:
            angles = getattr(self.body.angles, f"Hip{side}")
            result[side] = check_joint_angle_range(angles, self.angle_constraints)

        return result
    


##########################################################################
############################ PERSPECTIVE DEIFNITION ######################
##########################################################################

SIDE_VIEW_RULES = [KneeAngle, HipAngle]


if __name__ == "__main__":

    DATA_DIR_CLOUD = os.getenv("DATA_ON_CLOUD")
    VIDEO_PATH = os.path.join(DATA_DIR_CLOUD, "self_filmed_videos_squat/side_view_less_clothes.mov")


    model = PoseEstimator(**{
        "modularity" : "video",
        "reduce_lag" : True,
        "filter" : SavGol()
        })

    body = model(VIDEO_PATH)
    results = {}
    for rule in SIDE_VIEW_RULES:
        try:
            r = rule(body)
            rule_name = r.name
            results[rule_name] = r()
        except Exception as e:
            print(e)

        print(results)