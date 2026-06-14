import json
import numpy as np
import yaml

def read_json(path:str) -> dict:
    """ 
    Reads in a JSON file
    """
    with open(path, "r") as file:
        return json.load(file)
    
def save_dict_to_json(dict_obj: dict, save_to_path: str):
    with open(save_to_path, "w") as json_file:
        json.dump(dict_obj, json_file, indent=4)


def read_yaml(path:str):
    with open(path) as f:
        return yaml.safe_load(f)
    

import subprocess
import os

def mov_to_mp4(input_path: str, output_path: str = None, overwrite: bool = True):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp4"

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",  # overwrite or not
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    return output_path