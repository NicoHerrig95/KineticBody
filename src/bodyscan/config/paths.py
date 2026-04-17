""" 
Path definitions
"""

import os 
import sys 
from pathlib import Path



###### DIRS ######
ROOT_DIR = Path(__file__).resolve().parents[3]
BODYSCAN_DIR = ROOT_DIR / "bodyscan"


SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

TMP_DIR = ROOT_DIR / "tmp"
CACHE_DIR = TMP_DIR / "cache"

CONFIG_DIR = SRC_DIR / "bodyscan" / "config"


##### PATHS #####
VISUALIZATION_CONFIG_PATH = CONFIG_DIR / "visualization_config.yaml"