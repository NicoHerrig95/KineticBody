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
from pathlib import Path

# Reading config 
CONFIG_PATH = "./config/body.yaml"
config = read_yaml(CONFIG_PATH)["visualization"]
THICKNESS_SCALER = config["thickness_scale"]
JOINT_SCALER = config["joint_radius_scale"]
BODY_COLOR = tuple(config["body_color"])
JOINT_COLOR = tuple(config["joint_color"])



# HELPERS
def make_writer(path, fps, width, height):
    print(path)
    candidates = [
        ("mp4v", path),                        # .mp4
        ("MJPG", path.replace(".mp4", ".avi")),# .avi
        ("XVID", path.replace(".mp4", ".avi")),# .avi
    ]

    for codec, out in candidates:
        writer = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            print(f"Using codec={codec}, output={out}")
            return writer, out

    raise RuntimeError("Could not open VideoWriter with mp4v, MJPG, or XVID.")




###############################################################
# Visualisation Config 
###############################################################
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

skeletton_default = list(LIMBS_CONFIG.keys())
joints_default = list(JOINTS_CONFIG.keys())
unilaterals_default = list(UNILATERALS_CONFIG.keys())
angles_default = list(ANGLES_CONFIG.keys())

####################################################################
# Helper functions 
####################################################################
def get_pixel_coordinates(coordinates:tuple, w:int, h:int) -> tuple:
    x = coordinates[0]
    y = coordinates[1]
    return (int(x * w), int(y * h))


def scale_line_thickness(scale:int, width:int, height:int):
    """ 
    Scaling line thicknes proportional to image size
    """
    return max(1, int(min(height, width) / 200)) * scale


####################################################################
# Drawing functions
# > in-place manipulation of arrays
####################################################################

def draw_limb(frame:np.ndarray, coords:tuple, thickness_scaler:int, color:tuple):
    h, w = frame.shape[:2]
    p1 = get_pixel_coordinates(coords[0], w, h)
    p2 = get_pixel_coordinates(coords[1], w, h)
    line_thickness = scale_line_thickness(
        scale = thickness_scaler,
        width=w,
        height=h
    )
    cv2.line(frame, p1, p2, color, line_thickness)



def draw_joint_cirlce(
        frame:np.ndarray, 
        center:tuple, 
        radius:float, 
        thickness_scaler:int, 
        color:tuple
        ):

    """ 
    Drawing joint circles.
    """
    h, w = frame.shape[:2]
    center_normalized = get_pixel_coordinates(center, w, h)
    radius_normalized = scale_line_thickness(radius, w, h)
    thickness_normalized = scale_line_thickness(thickness_scaler, w, h)
    cv2.circle(frame, center_normalized, radius_normalized, color, thickness_normalized)
    

def draw_torso(
        frame:np.ndarray,
        left_shoulder:tuple,
        right_shoulder:tuple,
        left_hip:tuple,
        right_hip:tuple
    ):
    """ 
    Draws the torso as polygon.
    """
    h, w = frame.shape[:2]
    # Transforming normalized coordinates to pixel coordinares
    left_shoulder = get_pixel_coordinates(left_shoulder, w, h),
    right_shoulder = get_pixel_coordinates(right_shoulder, w, h),
    left_hip = get_pixel_coordinates(left_hip, w, h),
    right_hip = get_pixel_coordinates(right_hip, w, h),
    torso_pts = np.array([left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.int32)
    cv2.fillPoly(frame, [torso_pts], BODY_COLOR)



def draw_angle(
        frame:np.ndarray, 
        value:float, # angle value 
        org:tuple, # bottom left coordinate of the text
        text:Optional[str] = ""
        ):

    h, w = frame.shape[:2]
    text = f"{text}{int(value)} deg."
    org = get_pixel_coordinates(org, w, h)
    text_position = (org[0] + 10, org[1] + 10)
    cv2.putText(
        frame,
        text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        2
    )







def visualize_skeletton(
        body:KineticBody,
        skeletton:list,
        joints:list,
        unilaterals:list,
        angles:list,
        frame:np.ndarray,
        frame_idx:int
    ):
    """  
    Manipulating array by adding limbs etc.
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




    # Draw torso as polygon
    # draw_torso(
    #     frame=frame,
    #     left_shoulder = body.UpperBack[frame_idx][0],
    #     right_shoulder = body.UpperBack[frame_idx][1],
    #     left_hip = body.Hip[frame_idx][0],
    #     right_hip = body.Hip[frame_idx][1],
    # )

    return frame


def visualize_image(
        body: KineticBody, 
        skeletton:list = skeletton_default, # defines which limbs shall be visualized
        joints:list = joints_default, # defines which joints shall be highlighted
        unilaterals:list = unilaterals_default,
        angles:list = angles_default,
        background: Optional[np.ndarray] = None
        ):
    """ Visualize a KineticBody on a given background (image) or black canvas. """

    # Getting metadata
    mode = body.meta["mode"] # string
    if mode != "image":
        raise ValueError(f"body.meta.mode should be image but is {mode}")

    # Setup background
    if isinstance(background, np.ndarray):
        image = background
    else:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
    h, w = image.shape[:2]

    image = visualize_skeletton(
        body=body,
        skeletton=skeletton,
        joints=joints,
        unilaterals=unilaterals,
        angles=angles,
        frame=image,
        frame_idx=0, # always 0 when single frame/image
    )

    # Show image
    cv2.namedWindow("KineticBody", cv2.WINDOW_NORMAL)
    cv2.imshow("KineticBody", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def visualize_video(
    body,
    capture,
    out_path,
    skeletton=skeletton_default,
    joints=joints_default,
    unilaterals=unilaterals_default,
    angles=angles_default,
    ms_between_frames=30,
):
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not isinstance(capture, cv2.VideoCapture):
        raise TypeError("capture must be a cv2.VideoCapture")

    # default out_path is always .mp4 
    out_path = Path(out_path)
    if out_path.suffix != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path = str(out_path)
    # Instantiating writer
    writer, out_path = make_writer(
        path=out_path,
        fps =fps,
        width=width,
        height=height
    )

    # Sanity check
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("VideoWriter failed to open. Codec may be unavailable.")
    # Main Loop
    idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # Visualising skeletton (joints, limbs, angles etc.) on frame
        # NOTE: Visualisation is done in-place within function, so no object is returned!
        visualize_skeletton(
            body=body,
            skeletton=skeletton,
            joints=joints,
            unilaterals=unilaterals,
            angles=angles,
            frame=frame,
            frame_idx=idx,
        )

        writer.write(frame)
        idx += 1
    # Releasing caputre, writer and destoying cv2 windows
    capture.release()
    writer.release()
    cv2.destroyAllWindows()
