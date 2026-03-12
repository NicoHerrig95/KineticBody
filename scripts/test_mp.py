import os 
import sys
import cv2
import mediapipe as mp
import numpy as np 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from model.model_base_class import POSE

IMAGE_PATH = "./data/images/treadmill_test_img.PNG"
VIDEO_PATH = "./data/videos/treadmill_test_vid.MOV"
MODEL_PATH = "pose_landmarker_lite.task"


x = mp.Image.create_from_file(IMAGE_PATH)
print(x)