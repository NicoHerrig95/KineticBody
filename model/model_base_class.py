import os
import sys
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np 
import cv2
import mediapipe as mp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.common import read_json
import time
from model.body import KineticBody


def convert_coords_to_cv2(p1, p2, img_width, img_height):
    pass


class ModelBaseClass(ABC):
    def __init__(self, mode: str):
        self.mode = mode

    @abstractmethod
    def _inference(self, input_data):
        """Run inference on `input_data`; subclasses must override."""
        raise NotImplementedError

    def __call__(self, input_data) -> KineticBody:
        return self._inference(input_data)

#########################################################################################################################
#> Pose Model 
#########################################################################################################################
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# path (use BASE_DIR for absolute paths)
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
LANDMARK_MAPPING = read_json(os.path.join(BASE_DIR, "POSE_landmark_mapping.json"))

class POSE(ModelBaseClass):  # Inherit from ModelBaseClass
    def __init__(self, mode: str = "image"):
        super().__init__(mode)  # Fix: no self, pass mode
        self.mapping = LANDMARK_MAPPING
        # Declaring modularity
        if mode == "image":
            self.mode = RunningMode.IMAGE
        elif mode == "video":
            self.mode = RunningMode.VIDEO
        self.settings = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=self.mode,
            num_poses=1,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(self.settings)

        # coordinate storer 
        self.body = None


    def _image_inference(self, path:str, dimensions:int = 2, visualize:bool = True) -> Tuple[Dict, Dict]:

        positions = {k: [] for k in self.mapping}
        x = mp.Image.create_from_file(path)
        results = self.pose_landmarker.detect(x)
        if not results.pose_landmarks:
            print("No pose detected.")
            return None
        

        metadata = {
            "mode" : "image",
            "frame_count" : 1,
            "width" : x.width,
            "height" : x.height
            }

        landmarks = results.pose_landmarks[0]  # Access the first pose
        for feature in positions:
            idx = self.mapping[feature]
            lm = landmarks[idx]
            if dimensions == 2:
                coords = (lm.x, lm.y)
            elif dimensions == 3:
                coords = (lm.x, lm.y, lm.z)
            positions[feature].append(coords)
        return positions, metadata

    def _video_inference(self, path:str, dimensions:int = 2, visualize:bool = True) -> Tuple[Dict, Dict]:

        positions = {k: [] for k in self.mapping}
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            print("Error opening video file.")
            return None
        

        # Setting video metadata
        metadata = {
            "mode" : "video",
            "frame_count" : int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width" : int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height" : int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps" : capture.get(cv2.CAP_PROP_FPS)
            }

        # Inference looping over frames
        frame_idx = 0
        while capture.isOpened():

            ret, frame = capture.read()
            if not ret:
                break  # End of video
            
            # Convert frame to mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Inference on frame
            results = self.pose_landmarker.detect_for_video(mp_image, frame_idx)  # Use detect_for_video for tracking
            # Extracting per-bodypart coordinates from results
            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                for feature in positions:
                    idx = self.mapping[feature]
                    lm = landmarks[idx]
                    if dimensions == 2:
                        coords = (lm.x, lm.y)
                    elif dimensions == 3:
                        coords = (lm.x, lm.y, lm.z)
                    positions[feature].append(coords)
            frame_idx += 1
        
        capture.release()
        cv2.destroyAllWindows()
        return positions, metadata

    def _inference(self, input_data: str, dimensions: int = 2, visualize: bool = True) -> KineticBody:

        
        # processing image
        start_time = time.time()
        if self.mode == RunningMode.IMAGE:
            positions, metadata = self._image_inference(input_data, dimensions, visualize)
        elif self.mode == RunningMode.VIDEO:
            positions, metadata = self._video_inference(input_data, dimensions, visualize)

        inference_duration = time.time()-start_time
        print(f"Inference time: {inference_duration:.2f} seconds.")
        body = KineticBody(
            positions=positions, 
            metadata=metadata
            )

        return body

if __name__ == "__main__":

    MODE = "image"
    SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "data", "images", "treadmill_test_img.PNG")
    SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, "data", "videos", "treadmill_test_vid.MOV")
    
    model = POSE(mode = MODE)
    if MODE == "image":
        body = model(SAMPLE_IMAGE_PATH)
    elif MODE == "video":
        body = model(SAMPLE_VIDEO_PATH)