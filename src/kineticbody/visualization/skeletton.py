import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from kineticbody.kinetics.body import KineticBody
from kineticbody.utils.common import read_yaml
from kineticbody.visualization.config.bodyparts_visuals_config import bodyparts_visalization_config as config
from kineticbody.visualization.functions import (
    draw_angle, 
    draw_joint_cirlce,
    draw_limb
)


###############################################################
# Visualisation Config 
###############################################################





def get_skeletton(
        body:KineticBody,
        skeletton:list,
        joints:list,
        unilaterals:list,
        angles:list,
        frame:np.ndarray,
        frame_idx:int
    ):
    """  
    Adding body skeletton to a frame.
    """

    # Draw all limbs
    for limb_name in skeletton:
        for side in ["Left", "Right"]:
            config = config["limbs"][limb_name]
            attr_name = f"{limb_name}{side}"
            coords = getattr(body.limbs, attr_name)[frame_idx]
            draw_limb(frame, coords, config["thickness_scaler"], config["color"])

    # Draw Unilaterals
    for uni_name in unilaterals:
        config = config["unilaterals"][uni_name]
        coords = getattr(body.limbs, uni_name)[frame_idx]
        draw_limb(frame, coords, config["thickness_scaler"], config["color"])


    # Draw all joints
    for joint_name in joints:
        for side in ["Left", "Right"]:
            config = config["joints"][joint_name]
            attr_name = f"{joint_name}{side}"
            center = getattr(body.joints, attr_name)[frame_idx]
            draw_joint_cirlce(
                frame, 
                center, 
                config["joint_radius_scaler"], 
                config["thickness_scaler"], 
                config["color"]
            )

    # Draw angles
    for joint_angle in angles:
        for side in ["Left", "Right"]:
            attr_name = f"{joint_angle}{side}"
            org = getattr(body.joints, attr_name)[frame_idx]

            value = getattr(body.angles, attr_name)[frame_idx]
            draw_angle(
                frame = frame,
                value=value,
                org=org
            )

    return frame