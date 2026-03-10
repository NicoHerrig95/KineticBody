import os 
import sys
import cv2
import mediapipe as mp
import numpy as np 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from model.model_base_class import POSE

IMAGE_PATH = "./data/images/treadmill_test_img.PNG"
MODEL_PATH = "pose_landmarker_lite.task"



if __name__ == "__main__":
    MODE = sys.argv[1]
    model = POSE(mode = MODE)
    if MODE == "image":
        body = model(IMAGE_PATH)
    
    # body.visualize(background = cv2.imread(IMAGE_PATH))
    body.visualize(background = None)

    
