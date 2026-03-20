""" 
Movements
"""

import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import time
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.common import read_json, save_dict_to_json
from kinetics.body import KineticBody








class Movement(ABC):

    def __init__(self, body:KineticBody):
        self.body = body

