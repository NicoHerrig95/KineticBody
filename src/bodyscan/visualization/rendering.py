import os
import sys
from typing import Optional
import numpy as np 
import cv2
from bodyscan.kinetics.body import KineticBody
from bodyscan.utils.visualization import make_writer
from pathlib import Path
from bodyscan.visualization.skeletton import (
    get_skeletton,
    LIMBS_CONFIG,
    JOINTS_CONFIG,
    UNILATERALS_CONFIG,
    ANGLES_CONFIG
)


skeletton_default = list(LIMBS_CONFIG.keys())
joints_default = list(JOINTS_CONFIG.keys())
unilaterals_default = list(UNILATERALS_CONFIG.keys())
angles_default = list(ANGLES_CONFIG.keys())


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

    image = get_skeletton(
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
    angles=angles_default
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
        get_skeletton(
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
