import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from utils.common import read_json
from model.pose_estimation import PoseEstimator
from model.body import KineticBody
from model.visualization import visualize_video, visualize_image

VIDEO_PATH = "./data/real_time_exercise_recognition/3/final_kaggle_with_additional_video/barbell biceps curl/barbell biceps curl_3.mp4"



if __name__ == "__main__":
    MODE ="video"
    model = PoseEstimator(mode = MODE)
    body = model(VIDEO_PATH)
    visualize_video(body = body,capture=cv2.VideoCapture(VIDEO_PATH))