"""   
KineticBody
-> The KineticBody model gives information about coordinates of limbs and joints.
"""
import os
from kineticbody.utils.common import read_json
from kineticbody.config.paths import LANDMARK_MAPPING_PATH 
LANDMARK_MAPPING = read_json(LANDMARK_MAPPING_PATH)
from kineticbody.kinetics.bodyparts.angles import Angles
from kineticbody.kinetics.bodyparts.joints import Joints
from kineticbody.kinetics.bodyparts.limbs import Limbs
from kineticbody.kinetics.bodyparts.head import Head
from kineticbody.utils.common import save_dict_to_json, read_json



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

        # initialise body parts
        self._compute_body_parts(
            positions=positions,
            N_frames=metadata["frame_count"]
        )

    def _compute_body_parts(self, positions:dict, N_frames:int) -> None:
        # Setting joints
        self.joints = Joints(positions)
        # Setting head
        self.head = Head(positions, N_frames)
        #Setting Limbs
        self.limbs = Limbs(positions, N_frames)
        # Computing angles
        self.angles = Angles(positions, N_frames)


    def save_model(self, path:str) -> None:

        save_dict_to_json(
            {
                "positions" : self.positions,
                "metadata" : self.meta,
            },
            path
        )

    @classmethod
    def load_model(cls ,path:str):

        data = read_json(path)
        if "positions" not in data:
            raise KeyError("Positions key is not in in-read data.")
        if "metadata" not in data:
            raise KeyError("Metadata key is not in in-read data.")
        
        return cls(
            positions=data["positions"],
            metadata=data["metadata"]
        )

    
