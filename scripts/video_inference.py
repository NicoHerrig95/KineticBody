import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from src.bodyscan.utils.common import read_json, mov_to_mp4
from bodyscan.model.pose_estimation import PoseEstimator
from bodyscan.kinetics.body import KineticBody
from bodyscan.assets.visualization import visualize_video, visualize_image
from bodyscan.model.proc.filtering import SavGol
from bodyscan.utils.common import save_dict_to_json
from dotenv import load_dotenv
load_dotenv()

DATA_DIR_CLOUD = os.getenv("DATA_ON_CLOUD")

# Default parameters
LAG_REDUCTION = True
# VIDEO_PATH = os.path.join(DATA_DIR_CLOUD, "self_filmed_videos_squat/side_view_less_clothes.mov")
VIDEO_PATH = os.path.join(DATA_DIR_CLOUD, "angle_test.mov")

def video_inference(
        input_path:str,
        out_path:str,
        reduce_lag:bool,
        filter:None
        ):
    
    # NOTE: Redundant - To be deleted
    # defaulting out path as .mp4
    #if out_path is not None and not out_path.endswith(".mp4"):
    #    raise ValueError("Output must be of format .mp4")
    
    model = PoseEstimator(
        modality="video",
        reduce_lag=reduce_lag,
        filter=filter
        )
    # inference
    body = model(input_path)
    save_dict_to_json(body.positions, "squat_side_view_positions.json")

    # Adds video visuals and saves video
    if out_path is not None:
        visualize_video(
            body = body,
            capture= cv2.VideoCapture(input_path),
            out_path=out_path
        )
    else:
        pass 


        


if __name__ == "__main__":
    
    # minimal sample
    video_inference(
        input_path=VIDEO_PATH ,
        out_path="angle_test.mp4",
        reduce_lag=True,
        filter=SavGol(),
    )