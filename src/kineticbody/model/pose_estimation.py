from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
import mediapipe as mp
from kineticbody.utils.common import read_json
import time
from kineticbody.kinetics.body import KineticBody
from kineticbody.model.proc.position_interpolation import linear_interpolation
from kineticbody.utils.common import read_json
from kineticbody.config.paths import LANDMARK_MAPPING_PATH, POSE_TASK_FILE_DIR
LANDMARK_MAPPING = read_json(LANDMARK_MAPPING_PATH)

def convert_coords_to_cv2(p1, p2, img_width, img_height):
    """ 
    Converts coords with value range  [0,1] into cv2 format (value range defindes by pixels)
    """
    pass


def load_pose_model(
        size:str,
        modality:str,
        lag_reduction:bool
        ):
    
    """ 
    Loads POSE landmarker model.
    """
    modality_options = ["image", "video"]
    model_size_options = ["lite", "heavy"]
    if modality not in modality_options:
        raise ValueError(f"Modality must be in {modality_options}")
    if size not in model_size_options:
        raise ValueError(f"Size must be in {model_size_options}")
    

    # Declaring modality
    if modality == "image":
        model_mode = RunningMode.IMAGE
        # lag reduction is only applicable for video mode
    elif modality == "video":
        # If running in performance mode on video modality,
        # model uses image mode, which decreases tracking delay
        # but surpressing internal estimation smoothing.
        if lag_reduction:
            model_mode = RunningMode.IMAGE
        elif not lag_reduction:
            model_mode = RunningMode.VIDEO


    # constructing model path
    model_path = POSE_TASK_FILE_DIR / f"pose_landmarker_{size}.task"

    settings = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=model_mode,
        num_poses=1,
    )


    return PoseLandmarker.create_from_options(settings)


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

class PoseEstimator(ModelBaseClass):  # Inherit from ModelBaseClass
    def __init__(
            self, 
            modality: str = "video", 
            reduce_lag:bool = True,
            filter:Optional[object] = None,
            size:str="heavy"
            ):
        super().__init__(mode=modality)  
        self.mapping = LANDMARK_MAPPING
        self.filter = filter
        # Variable setting if modality is image but video settings are provided
        if modality == "image":
            # lag reduction is only applicable for video mode
            self.reduce_lag = "not applicable"
            self.filter = None # not applicable for image

        elif modality == "video":
            self.reduce_lag = reduce_lag
            self.filter = filter
            
        # loading model
        self.pose_landmarker = load_pose_model(
            size= size,
            modality=modality,
            lag_reduction=reduce_lag
        )

        # coordinate storer 
        self.body = None


    def _image_inference(self, path:str, dimensions:int = 2) -> Tuple[Dict, Dict]:

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

    def _video_inference(self, path:str, dimensions:int = 2) -> Tuple[Dict, Dict]:

        positions = {k: [] for k in self.mapping}
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            print("Error opening video file.")
            return None

        # Inference looping over frames
        frame_count = 0
        no_pose_count = 0
        while capture.isOpened():

            ret, frame = capture.read()
            if not ret:
                break  # End of video
            
            # Convert frame to mp.Image
            x = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Inference on frame
            if self.reduce_lag:
                # Using default image detection (without smoothing)
                results = self.pose_landmarker.detect(x)
            elif not self.reduce_lag:
                results = self.pose_landmarker.detect_for_video(x, frame_count)  # Use detect_for_video for tracking
            
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
            # if no pose is detected, append np.nan
            elif not results.pose_landmarks:
                no_pose_count += 1
                for feature in positions:
                    if dimensions == 2:
                        positions[feature].append((np.nan, np.nan)) # x,y missing
                    elif dimensions == 3:
                        # positions[feature].append((None, None, None))
                        positions[feature].append((np.nan, np.nan, np.nan)) # x,y,z missing

            # updating frame count
            frame_count += 1                

        capture.release()
        cv2.destroyAllWindows()

        metadata = {
            "mode" : "video",
            "frame_count" : frame_count, 
            "width" : int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height" : int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps" : capture.get(cv2.CAP_PROP_FPS),
            "no_detection_count" : no_pose_count
            }

        return positions, metadata

    def _inference(self, input_data: str, dimensions: int = 2) -> KineticBody:
        
        # processing image
        start_time = time.time()
        if self.mode == "image":
            positions, metadata = self._image_inference(input_data, dimensions)
        elif self.mode == "video":
            positions, metadata = self._video_inference(input_data, dimensions)
        inference_duration = time.time()-start_time

        # Interpolating missing values in positions
        if metadata["no_detection_count"] > 0:
            for lm in positions:
                x = [coords[0] for coords in positions[lm]]
                y = [coords[1] for coords in positions[lm]]
                corrected = []
                x_corrected = linear_interpolation(x, window=2)
                y_corrected = linear_interpolation(y, window=2)
                if len(x_corrected) == len(y_corrected):
                    frame_count = len(x_corrected) 
                    positions[lm] = [
                        [float(x_corrected[i]), float(y_corrected[i])] 
                        for i in range(frame_count)
                        ]


        # Apply filtering if applicable
        if self.filter is not None:
            positions = self.filter(positions)
        
        print(f"Inference time: {inference_duration:.2f} seconds.")
        print(f"Frames with no pose detected: {metadata["no_detection_count"]}")

        body = KineticBody(
            positions=positions, 
            metadata=metadata
            )

        return body







