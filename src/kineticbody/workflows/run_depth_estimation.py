from kineticbody.model.depth_estimation import DepthEstimationPipeline
from kineticbody.utils.common import video_to_frames, get_modality, save_dict_to_json
import numpy as np
from typing import List
from pathlib import Path


# DEPTH_ESTIMATOR_NAME = "depth-anything/Depth-Anything-V2-Large-hf"
DEPTH_ESTIMATOR_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
STORAGE_DTYPE = np.float16



##################################################################
# Save and Load functions for handling depth map arrays 
##################################################################

def save_depth_maps(input:List[np.ndarray], save_to:str, format = STORAGE_DTYPE) -> None:
    depth_maps_fomatted = [_.astype(format) for _ in input]
    np.save(save_to, np.stack(depth_maps_fomatted))


def load_depth_maps(path:str) -> List[np.ndarray]:
    return [_ for _ in np.load(path)]



##################################################################
# Main workflow
##################################################################

def run_depth_estimation(
        input_path:str,
        save_to_path:str, 
        batch_size:int = 8, 
        verbose:bool = True
        ):

    if Path(save_to_path).suffix != ".npy":
        raise ValueError(f".npy suffix is required for save_to_path argument but got {save_to_path}")

    # reading in input
    modality = get_modality(input_path)


    # --> Image modality currently not supported
    if modality == "image":
        raise ValueError("Image modality currently not supported")


    # VIDEO BLOCK
    # extracting frames
    if modality == "video":
        frames = video_to_frames(input_path)

    pipeline = DepthEstimationPipeline(
        model_name=DEPTH_ESTIMATOR_NAME
    )

    depth_maps = pipeline.estimate(
        frames=frames,
        batch_size=batch_size,
        to_numpy=True,
        verbose=verbose
        )

    # saving depth maps
    save_depth_maps(input=depth_maps, save_to=save_to_path)

    

     