import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import cv2
import mediapipe as mp
import numpy as np 
from bodyscan.utils.common import save_dict_to_json
from bodyscan.model.pose_estimation import PoseEstimator
from bodyscan.kinetics.body import KineticBody
from bodyscan.assets.visualization import visualize_video, visualize_image
from bodyscan.model.proc.filtering import SavGol
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DATA_DIR_CLOUD = os.getenv("DATA_ON_CLOUD")


# Default parameters
LAG_REDUCTION = True
TEST_DIR = os.path.join(DATA_DIR_CLOUD, "self_filmed_videos_squat")
OUT_DIR = TEST_DIR 



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
        file_name = out_path.stem
        out_path = str(out_path) # converting back to string
        # print(p)
        body = model(p)
        save_dict_to_json(body.positions, f"{file_name}_positions.json")
        
        
        visualize_video(
            body = body,
            capture=cv2.VideoCapture(p),
            out_path=out_path
        )
