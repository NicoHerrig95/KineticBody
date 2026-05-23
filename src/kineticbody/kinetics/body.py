"""   
KineticBody
-> The KineticBody model gives information about coordinates of limbs and joints.
"""
from kineticbody.utils.common import read_json
from kineticbody.config.paths import LANDMARK_MAPPING_PATH 
LANDMARK_MAPPING = read_json(LANDMARK_MAPPING_PATH)
from kineticbody.kinetics.bodyparts.angles import Angles
from kineticbody.kinetics.bodyparts.joints import Joints
from kineticbody.kinetics.bodyparts.limbs import Limbs
from kineticbody.kinetics.bodyparts.head import Head




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
        # Setting head
        self.head = Head(positions, self.N_frames)
        #Setting Limbs
        self.limbs = Limbs(positions, self.N_frames)
        # Computing angles
        self.angles = Angles(positions, self.N_frames)
