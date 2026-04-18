import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from bodyscan.kinetics.body import KineticBody
from bodyscan.utils.common import read_yaml
from bodyscan.config.paths import VISUALIZATION_CONFIG_PATH
from bodyscan.utils.visualization import (
    draw_angle, 
    draw_joint_cirlce,
    draw_limb
)


###############################################################
# Visualisation Config 
###############################################################

# Reading config 
config = read_yaml(VISUALIZATION_CONFIG_PATH)["body"]
THICKNESS_SCALER = config["thickness_scale"]
JOINT_SCALER = config["joint_radius_scale"]
BODY_COLOR = tuple(config["body_color"])
JOINT_COLOR = tuple(config["joint_color"])
# Define config for each limb
LIMBS_CONFIG = {
    # Bilateral Limbs
    "UpperArm": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Forearm": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "UpperLeg": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "LowerLeg": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Thumb": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Index": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Pinky": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Heel": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Foot": {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Torso" : {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
}

UNILATERALS_CONFIG = {
    "UpperBack" : {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
    "Hip" : {"thickness_scaler": THICKNESS_SCALER, "color": BODY_COLOR},
}

JOINTS_CONFIG = {
    "Elbow" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Wrist" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Hip" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Knee" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Ankle" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Heel" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
    "Shoulder" : {"joint_radius_scaler" : JOINT_SCALER, "color" : JOINT_COLOR, "thickness_scaler": THICKNESS_SCALER,},
}

ANGLES_CONFIG = {
    "Knee" : {},
    "Elbow" : {},
    "Hip" : {}
}


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
            config = LIMBS_CONFIG[limb_name]
            attr_name = f"{limb_name}{side}"
            coords = getattr(body.limbs, attr_name)[frame_idx]
            draw_limb(frame, coords, config["thickness_scaler"], config["color"])

    # Draw Unilaterals
    for uni_name in unilaterals:
        config = UNILATERALS_CONFIG[uni_name]
        coords = getattr(body.limbs, uni_name)[frame_idx]
        draw_limb(frame, coords, config["thickness_scaler"], config["color"])


    # Draw all joints
    for joint_name in joints:
        for side in ["Left", "Right"]:
            config = JOINTS_CONFIG[joint_name]
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