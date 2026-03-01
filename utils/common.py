import json
import numpy as np


def read_json(path:str) -> dict:
    """ 
    Reads in a JSON file
    """
    if not path.endswith(".json"):
        path = path+".json"
    with open(path, "r") as file:
        return json.load(file)
    
def save_dict_to_json(dict_obj: dict, save_to_path: str):
    with open(save_to_path, "w") as json_file:
        json.dump(dict_obj, json_file, indent=4)