import os
import sys
from abc import ABC, abstractmethod
import numpy as np 






class ModelBaseClass(ABC):
    def __init__(self, mode: str):
        self.mode = mode

    @abstractmethod
    def _inference(self, input_data):
        """Run inference on `input_data`; subclasses must override."""
        raise NotImplementedError

    def __call__(self, input_data):
        return self._inference(input_data)



#########################################################################################################################
#> Pose Model 
#########################################################################################################################
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# path
MODEL_PATH = "pose_landmarker_lite.task"


class POSE(object):
    super().__init__(self, mode=mode)

    if mode == "image":
        self.mode = RunningMode.IMAGE


    self.settings = options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=self.mode,
        num_poses=1, # -> 1 for image
    )
    self.pose_landmarker = PoseLandmarker.create_from_options(self.settings)

    def _inference(self, input:str):

        # processing image
        if self.mode == "image":
            # img -> np.ndarray 
            x = cv2.imread(input)

        results = self.pose_landmarker.detect(x)
        landmarks = results.pose_landmarks # these are the coordinates 






















mp_image = mp.Image.create_from_file(IMAGE_PATH)



with :
    result = landmarker.detect(mp_image)

if not result.pose_landmarks:
    print("No pose detected.")
    exit()

# Load image for drawing
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]

landmarks = result.pose_landmarks[0]

# Draw points
for i, lm in enumerate(landmarks):
    x_px = int(lm.x * w)
    y_px = int(lm.y * h)
    cv2.circle(image, (x_px, y_px), 5, (0, 0, 255), -1)
    cv2.putText(image, str(i), (x_px + 5, y_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)