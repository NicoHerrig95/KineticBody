import cv2
import mediapipe as mp
import numpy as np 


IMAGE_PATH = "./data/images/treadmill_test_img.PNG"
MODEL_PATH = "pose_landmarker_lite.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Load image for inference
mp_image = mp.Image.create_from_file(IMAGE_PATH)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

if not result.pose_landmarks:
    print("No pose detected.")
    exit()

# Load image for drawing
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]

landmarks = result.pose_landmarks[0]

print(landmarks[0])

#cv2.imshow("Pose", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()