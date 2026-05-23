import os
from kineticbody.utils.common import read_yaml 
from kineticbody.config.paths import VISUALIZATION_GENERAL_CONFIG_PATH

# Reading config 
config = read_yaml(VISUALIZATION_GENERAL_CONFIG_PATH)["body"]
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



bodyparts_visalization_config = {
    "limbs" : LIMBS_CONFIG,
    "unilaterals" : UNILATERALS_CONFIG,
    "joints" : JOINTS_CONFIG,
    "angles" : ANGLES_CONFIG
 }