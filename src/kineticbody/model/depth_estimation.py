""" 
Computes the depthmap of each frame from a video
"""
from transformers import pipeline
import time
import logging
import torch
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


INFERENCE_DTYPE = torch.float32

# checking if cuda or MLX is available (on Apple Silicon)
def check_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class DepthEstimationPipeline():
    """ 
    Depth estimator model class for estimating depth values on a 2D image.
    """
    def __init__(self, model_name:str):
        self.device = check_device()
        self.estimator = pipeline(
            task="depth-estimation", 
            model=model_name,
            device=self.device,
            dtype=INFERENCE_DTYPE
            )

        
        logging.info(f"Running DepthEstimator on {self.device}")


    def estimate(self, 
                 frames:list, 
                 batch_size:int = 8, 
                 to_numpy:bool = True, 
                 verbose:bool = True
                 ):

        depth_maps = []
        N_frames = len(frames)
        if verbose:
            logging.info(f"Processing {N_frames} frames")

        start = time.perf_counter()
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            results = self.estimator(
                batch,
                batch_size=batch_size,
                device= self.device
            )

            depth_maps.extend(
                result["predicted_depth"]
                for result in results
            )

        inference_duration = time.perf_counter() - start
        if verbose:
            logging.info(f"Depth map estimation took {inference_duration} seconds")

        if to_numpy:
            # moving tensors from gpu to cpu (if applicable) and convering to numpy
            depth_maps = [tensor.detach().cpu().numpy() for tensor in depth_maps]

        return depth_maps



















