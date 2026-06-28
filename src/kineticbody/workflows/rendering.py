from typing import Optional
import numpy as np 
import cv2
from pathlib import Path

from kineticbody.visualization.data_class import Video
from kineticbody.kinetics.body import KineticBody
from kineticbody.visualization.functions import make_writer
from kineticbody.visualization.skeletton import get_skeletton
from kineticbody.visualization.config.bodyparts_visuals_config import (
    LIMBS_CONFIG,
    JOINTS_CONFIG,
    UNILATERALS_CONFIG,
    ANGLES_CONFIG
)

# Getting default objects from each category to visualize
skeletton_default = list(LIMBS_CONFIG.keys())
joints_default = list(JOINTS_CONFIG.keys())
unilaterals_default = list(UNILATERALS_CONFIG.keys())
angles_default = list(ANGLES_CONFIG.keys())



def visualize_video(
    body,
    capture,
    skeletton=skeletton_default,
    joints=joints_default,
    unilaterals=unilaterals_default,
    angles=angles_default,
) -> Video:
    if not isinstance(capture, cv2.VideoCapture):
        raise TypeError("capture must be a cv2.VideoCapture")


    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # modifies frame in-place
        get_skeletton(
            body=body,
            skeletton=skeletton,
            joints=joints,
            unilaterals=unilaterals,
            angles=angles,
            frame=frame,
            frame_idx=idx,
        )

        frames.append(frame.copy())
        idx += 1

    capture.release()
    cv2.destroyAllWindows()

    video = Video(
        frames=frames,
        fps = fps,
        width=width,
        height=height
    )

    return video
