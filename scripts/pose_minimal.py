import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from bodyscan.utils.common import read_json
from bodyscan.model.pose_estimation import PoseEstimator, PoseEstimatorVideoEnhancement
from model.body import KineticBody
from model.visualization import visualize_video, visualize_image

IMAGE_PATH = "./data/images/treadmill_test_img.PNG"
VIDEO_PATH = "./data/videos/treadmill_test_vid.MOV"
MODEL_PATH = "pose_landmarker_lite.task"



if __name__ == "__main__":
    MODE = sys.argv[1]

    model = PoseEstimatorVideoEnhancement(mode = MODE)
    if MODE == "image":
        body = model(IMAGE_PATH)        
        body.save("positions_image.json")
        visualize_image(
            body = body,
            background=cv2.imread(IMAGE_PATH)
        )
    elif MODE == "video":
        # body = model(VIDEO_PATH)
        # # saving body 
        # body.save("positions_video.json")
        # print(body.UpperArmRight)
        positions = read_json("positions_video.json")
        pos = positions["positions"]
        meta = positions["meta"]
        body = KineticBody(
            positions=pos,
            metadata=meta
        )
        
        visualize_video(body = body,capture=cv2.VideoCapture(VIDEO_PATH))