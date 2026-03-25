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



class Rule(ABC):
    def __init__(self, 
                 body:KineticBody
                 ):
        self.body = body


    @abstractmethod
    def check(self) -> dict:
        """ 
        Checks kinetic body on a constraint. 
        Triggers recommendation if required.
        Returns a dictionary called result.
        """
        pass

    @abstractmethod
    def get_recommendation(self, result:dict) -> str|None:
        """ 
        Returns a recommendation of rule is violated.
        """
        pass

    def __call__(self) -> str|None:
        result = self.check()
        return self.get_recommendation(result)




class Movement(ABC):

    def __init__(self, 
                 body:KineticBody):
        self.body = body

    @abstractmethod
    def check_movement(self):
        """ 
        Checking movement-specific constraints via a set of rules (e.g. )
        """
        pass 


