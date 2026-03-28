import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from utils.common import read_json, mov_to_mp4
from model.pose_estimation import PoseEstimator
from kinetics.body import KineticBody
from assets.visualization import visualize_video, visualize_image
from src.model.proc.filtering import SavGol


from src.movements.squat.side_view import KneeBelow90Degrees

# Default parameters
LAG_REDUCTION = True
VIDEO_PATH = "./data/real_time_exercise_recognition/3/final_kaggle_with_additional_video/barbell biceps curl/barbell biceps curl_3.mp4"
V2 = "./data/real_time_exercise_recognition/3/similar_dataset/squat/1e2c254b-0d5a-4fd6-a6d4-2681333d927b.mp4"
V3 = "./data/videos/self_squat_example.mov"

def video_inference(
        input_path:str,
        out_path:str,
        reduce_lag:bool,
        filter:None
        ):
    if out_path is not None and not out_path.endswith(".mp4"):
        raise ValueError("Output must be of format .mp4")
    
    if input_path.endswith(".mov") or input_path.endswith(".MOV"):
        mov_to_mp4(input_path)
        # overwriting input path
        input_path = os.path.splitext(input_path)[0] + ".mp4"



    model = PoseEstimator(
        modularity="video",
        reduce_lag=reduce_lag,
        filter=filter
        )
    # inference
    body = model(input_path)
    # Adds video visuals and saves video
    if out_path is not None:
        visualize_video(
            body = body,
            capture= cv2.VideoCapture(input_path),
            out_path=out_path
        )
        # applying rule
        r1 = KneeBelow90Degrees(body)
        xx = r1()
        print(xx)
    # NOTE: TESTING
    elif out_path is None:
        # applying rule
        r1 = KneeBelow90Degrees(body)
        xx = r1()
        print(xx)

        


if __name__ == "__main__":
    
    # minimal sample
    video_inference(
        input_path=V2 ,
        out_path="example3.mp4",
        reduce_lag=True,
        filter=SavGol(),
    )