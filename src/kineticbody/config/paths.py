""" 
Path definitions
"""

import os 
import sys 
from pathlib import Path



###### DIRS ######
ROOT_DIR = Path(__file__).resolve().parents[3]
KINETICBODY_DIR = ROOT_DIR / "kineticbody"

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

TMP_DIR = ROOT_DIR / "tmp"
CACHE_DIR = TMP_DIR / "cache"
VISUALIZATION_DIR = KINETICBODY_DIR / "visualization"
CONFIG_DIR = KINETICBODY_DIR / "config"
KINETICS_DIR = KINETICBODY_DIR / "kinetics"
MODEL_DIR = KINETICBODY_DIR / "model"

##### PATHS #####
VISUALIZATION_GENERAL_CONFIG_PATH = VISUALIZATION_DIR / "config" / "general.yaml"
LANDMARK_MAPPING_PATH = KINETICS_DIR / "config" / "landmark_mapping.json"
POSE_TASK_FILE_DIR = MODEL_DIR / "config" / "tensorflow" 