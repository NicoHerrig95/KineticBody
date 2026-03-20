import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from utils.common import read_json
from model.pose_estimation import PoseEstimator
from kinetics.body import KineticBody
from assets.visualization import visualize_video, visualize_image


# Default parameters
LAG_REDUCTION = True
VIDEO_PATH = "./data/real_time_exercise_recognition/3/final_kaggle_with_additional_video/barbell biceps curl/barbell biceps curl_3.mp4"
V2 = "./data/real_time_exercise_recognition/3/similar_dataset/squat/1e2c254b-0d5a-4fd6-a6d4-2681333d927b.mp4"

def video_inference(
        input_path:str,
        out_path:str,
        reduce_lag:bool,
        ):
    if not out_path.endswith(".mp4"):
        raise ValueError("Output must be of format .mp4")
    
    model = PoseEstimator(
        modularity="video",
        reduce_lag=reduce_lag
        )
    # inference
    body = model(input_path)
    # Adds video visuals and saves video
    visualize_video(
        body = body,
        capture= cv2.VideoCapture(input_path),
        out_path=out_path
    )


if __name__ == "__main__":
    
    # sample
    video_inference(
        input_path=V2 ,
        out_path="./example_squat.mp4",
        reduce_lag=True
    )