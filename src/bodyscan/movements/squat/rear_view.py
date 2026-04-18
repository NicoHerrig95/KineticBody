import os
from bodyscan.kinetics.body import KineticBody
from bodyscan.utils.common import read_json
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

class ShoulderBalance(Rule):

    def __init__(self, body:KineticBody):
        super().__init__(body = body)

    def check(self):
        pass



class HipBalance(Rule):

    def __init__(self, body:KineticBody):
        super().__init__(body = body)

    def check(self):
        pass


