import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import kagglehub

""" 
Loading real_time_exercise_recognition dataset
"""

os.environ["KAGGLEHUB_CACHE"] = "./data/real_time_exercise_recognition"
print("Dataset saving to:", os.environ["KAGGLEHUB_CACHE"])
path = kagglehub.dataset_download(
    "riccardoriccio/real-time-exercise-recognition-dataset"
)
print("Dataset stored at:", path)