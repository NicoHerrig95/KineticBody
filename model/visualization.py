import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from model.body import KineticBody


####################################################################
# Helper functions 
####################################################################
def get_pixel_coordinates(coordinates:tuple, w:int, h:int) -> tuple:
    x = coordinates[0]
    y = coordinates[1]
    return (int(x * w), int(y * h))




####################################################################
# Drawing functions
# > in-place manipulation of arrays
####################################################################

def draw_limb(frame:np.ndarray, coords:tuple, thickness:int, color:tuple):
    h, w = frame.shape[:2]
    p1 = get_pixel_coordinates(coords[0], w, h)
    p2 = get_pixel_coordinates(coords[1], w, h)
    cv2.line(frame, p1, p2, color, thickness)


def draw_torso(
        frame:np.ndarray,
        left_shoulder:tuple,
        right_shoulder:tuple,
        left_hip:tuple,
        right_hip:tuple
    ):
    h, w = frame.shape[:2]
    # Transforming normalized coordinates to pixel coordinares
    left_shoulder = get_pixel_coordinates(left_shoulder, w, h),
    right_shoulder = get_pixel_coordinates(right_shoulder, w, h),
    left_hip = get_pixel_coordinates(left_hip, w, h),
    right_hip = get_pixel_coordinates(right_hip, w, h),
    torso_pts = np.array([left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.int32)
    cv2.fillPoly(frame, [torso_pts], BODY_COLOR)



def visualize_skeletton(
        body:KineticBody,
        skeletton:list,
        frame:np.ndarray,
        frame_idx:int
    ):
    """  
    Manipulating array by adding limbs etc.
    """

    # Draw all limbs
    for limb_name in skeletton:
        config = LIMBS_CONFIG[limb_name]
        coords = getattr(body, limb_name)[frame_idx]
        draw_limb(frame, coords, config["thickness"], config["color"])

    # Draw torso as polygon
    draw_torso(
        frame=frame,
        left_shoulder = body.UpperBack[frame_idx][0],
        right_shoulder = body.UpperBack[frame_idx][1],
        left_hip = body.Hip[frame_idx][0],
        right_hip = body.Hip[frame_idx][1],
    )

    return frame












# VARS
LIMB_THICKNESS = {
    "UPPER_ARM" : 20,
    "FOREARM" : 20,
    "UPPER_LEG": 40,
    "LOWER_LEG" : 30
}
BODY_COLOR = (0, 255, 0) # GREEN


# Define config for each limb
LIMBS_CONFIG = {
    "UpperArmRight": {"thickness": LIMB_THICKNESS["UPPER_ARM"], "color": BODY_COLOR},
    "UpperArmLeft": {"thickness": LIMB_THICKNESS["UPPER_ARM"], "color": BODY_COLOR},

    "ForearmRight": {"thickness": LIMB_THICKNESS["FOREARM"], "color": BODY_COLOR},
    "ForearmLeft": {"thickness": LIMB_THICKNESS["FOREARM"], "color": BODY_COLOR},

    "UpperLegRight": {"thickness": LIMB_THICKNESS["UPPER_LEG"], "color": BODY_COLOR},
    "UpperLegLeft": {"thickness": LIMB_THICKNESS["UPPER_LEG"], "color": BODY_COLOR},

    "LowerLegRight": {"thickness": LIMB_THICKNESS["LOWER_LEG"], "color": BODY_COLOR},
    "LowerLegLeft": {"thickness": LIMB_THICKNESS["LOWER_LEG"], "color": BODY_COLOR},

    "ThumbRight": {"thickness": 10, "color": BODY_COLOR},
    "ThumbLeft": {"thickness": 10, "color": BODY_COLOR},

    "IndexRight": {"thickness": 10, "color": BODY_COLOR},
    "IndexLeft": {"thickness": 10, "color": BODY_COLOR},

    "PinkyRight": {"thickness": 10, "color": BODY_COLOR},
    "PinkyLeft": {"thickness": 10, "color": BODY_COLOR},

    "HeelRight": {"thickness": 15, "color": BODY_COLOR},
    "HeelLeft": {"thickness": 15, "color": BODY_COLOR},

    "FootRight": {"thickness": 15, "color": BODY_COLOR},
    "FootLeft": {"thickness": 15, "color": BODY_COLOR},
}
skeletton_default = list(LIMBS_CONFIG.keys())



def visualize_image(
        body: KineticBody, 
        skeletton:list = skeletton_default, # defines which body attributes shall be visualized
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
        frame=image,
        frame_idx=0, # always 0 when single frame/image
    )

    # Show image
    cv2.namedWindow("KineticBody", cv2.WINDOW_NORMAL)
    cv2.imshow("KineticBody", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def visualize_video(
        body: KineticBody, 
        capture: cv2.VideoCapture,
        skeletton:list = skeletton_default, # defines which body attributes shall be visualized
        ms_between_frames:int = 10
    ):

    # Getting metadata
    mode = body.meta["mode"] # string


    if mode == "video":
        fps = body.meta["fps"]
    else:
        raise ValueError(f"body.meta.mode should be video but is {mode}")
    
    if isinstance(capture, cv2.VideoCapture):
        idx = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            # > drawing logic here
            visualize_skeletton(
                body=body,
                skeletton=skeletton,
                frame=frame,
                frame_idx=idx
            )
            cv2.namedWindow("Kinetic Body", cv2.WINDOW_NORMAL)            
            cv2.imshow("Kinetic Body", frame)
            idx +=1 
            if cv2.waitKey(ms_between_frames) & 0xFF == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()



