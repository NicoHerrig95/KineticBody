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
from tqdm import tqdm
from pathlib import Path

# Default parameters
LAG_REDUCTION = True
TEST_DIR = "./data/videos/test"
OUT_DIR = "./data/annotations/test"


if __name__ == "__main__":

    input_paths = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)]
    model = PoseEstimator(
        modularity="video",
        reduce_lag=True,
        filter=SavGol()
    )
    for p in tqdm(input_paths):
        out_path = Path(p)
        out_path = out_path.with_name(f"{out_path.stem}_annotated{out_path.suffix}")
        out_path = str(out_path) # converting back to string
        print(p)
        body = model(p)
        visualize_video(
            body = body,
            capture=cv2.VideoCapture(p),
            out_path=out_path
        )
