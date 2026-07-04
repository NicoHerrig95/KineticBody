# KineticBody

KineticBody is a Python package for extracting and visualizing human body pose
kinematics from images and videos. It uses MediaPipe Pose Landmarker to estimate
body landmarks, converts those landmarks into a `KineticBody` model, and can
render the estimated skeleton, joints, unilateral body parts, and angles back on
top of the original video.

## Features

- Pose estimation for `.mp4`, `.mov`, `.png`, `.jpg`, and `.jpeg` inputs
- `KineticBody` model export as JSON
- Improved performance compared to the baseline MediaPipe Pose Landmarker by applying lag reduction and Savitzky-Golay filtering for video inference (see below).
- Video visualization from a saved body model
- Docker support for running pose estimation in a reproducible environment

## Demo

The front-view squat example below compares the KineticBody visualization with a
plain MediaPipe visualization.

<table>
  <tr>
    <th>KineticBody</th>
    <th>Plain MediaPipe</th>
  </tr>
  <tr>
    <td>
      <video src="docs/videos/squat_front_view_shortened.mov" controls width="360"></video>
      <br>
      <a href="docs/videos/squat_front_view_shortened.mov">Open KineticBody video</a>
    </td>
    <td>
      <video src="docs/videos/squat_front_view_shortened_mp_only.mov" controls width="360"></video>
      <br>
      <a href="docs/videos/squat_front_view_shortened_mp_only.mov">Open plain MediaPipe video</a>
    </td>
  </tr>
</table>

## Project Structure

```text
KineticBody/
├── docs/
│   └── videos/
├── docker/
│   ├── pose_estimation.dockerfile
│   └── requirements.txt
├── scripts/
│   ├── build_container.bash
│   ├── run_pose_estimation.py
│   ├── run_pose_estimation_container.bash
│   └── visualize_body.py
├── src/
│   └── kineticbody/
│       ├── config/
│       ├── kinetics/
│       ├── model/
│       ├── utils/
│       ├── visualization/
│       └── workflows/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12 is recommended because the Docker image uses `python:3.12-slim`
- `pip`
- `ffmpeg` and OpenCV runtime libraries for video processing
- Docker, optional, for containerized pose estimation

The main Python dependencies are pinned in `requirements.txt` and
`docker/requirements.txt`. They include MediaPipe, OpenCV, NumPy, SciPy, and
PyYAML.

## Setup

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd KineticBody
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies and the package:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If your environment has trouble reading `requirements.txt`, use the ASCII copy
used by the Docker build:

```bash
pip install -r docker/requirements.txt
pip install -e .
```

## Pose Estimation

Run pose estimation with:

```bash
python scripts/run_pose_estimation.py \
  --input-path path/to/input_video.mp4 \
  --output-path outputs/body_model.json
```

The script accepts both dash and underscore variants for its arguments:

```bash
python scripts/run_pose_estimation.py \
  --input_path path/to/input_video.mp4 \
  --output_path outputs/body_model.json
```

### Inputs

Supported input file types are:

- Video: `.mp4`, `.mov`
- Image: `.png`, `.jpg`, `.jpeg`

The script checks that the input file exists before running inference.

### Output

The output is a JSON file containing a serialized `KineticBody` model:

- `positions`: per-frame normalized landmark coordinates
- `metadata`: mode, frame count, input dimensions, and video-specific values
  such as FPS and no-detection count

This JSON file is the input for the visualization workflow.

### Processing Options

`scripts/run_pose_estimation.py` currently enables both processing options in
the script:

```python
LAG_REDUCTION = True
APPLY_FILTER = True
```

Lag reduction changes video inference to use image-mode detection frame by
frame, which reduces tracking delay but bypasses MediaPipe's internal video
smoothing. Filtering applies the configured Savitzky-Golay filter after
landmark extraction.

The default model and filter settings are defined in:

```text
src/kineticbody/model/config/estimator_config.yaml
```

Current defaults:

```yaml
pose_estimator:
  size: heavy
  reduce_lag: True
  filter: True

filter:
  algorithm: SavGol
  window_size: 9
  polynomial_order: 2
```

The pose model files are stored in:

```text
src/kineticbody/model/config/tensorflow/
```

Available model sizes are `lite` and `heavy`.

## KineticBody Class

The `KineticBody` class is the central data model created by the pose estimation
workflow. It stores the detected landmark positions, keeps metadata about the
input, and computes convenient body-part objects for analysis and visualization.

The class is defined in:

```text
src/kineticbody/kinetics/body.py
```

A `KineticBody` instance contains:

- `positions`: normalized MediaPipe landmark coordinates for every detected
  body point across all frames
- `meta`: metadata from the input, including mode, frame count, dimensions, and
  video-specific values when available
- `N_frames`: number of frames represented by the model
- `mode`: input mode, either `image` or `video`
- `joints`: joint coordinates grouped by body part and side
- `head`: head landmarks, including nose, ears, and eyes
- `limbs`: limb vectors computed from pairs of landmarks
- `angles`: joint angles computed from landmark triplets

### Loading a KineticBody Model

Pose estimation returns a `KineticBody` instance directly:

```python
from kineticbody.workflows.run_pose_estimation import run_pose_estimation

body = run_pose_estimation(
    input_path="path/to/input_video.mp4",
    lag_reduction=True,
    apply_filter=True,
)
```

You can also load a previously saved model:

```python
from kineticbody.kinetics.body import KineticBody

body = KineticBody.load_model("outputs/body_model.json")
```

The saved file contains only the raw `positions` and `metadata`. When it is
loaded again, `KineticBody.load_model()` rebuilds the derived `joints`, `head`,
`limbs`, and `angles` objects from those values.

### Accessing Metadata

```python
body.mode        # "video" or "image"
body.N_frames    # number of frames in the model
body.meta        # full metadata dictionary
```

For video inputs, `body.meta` includes values such as `fps`, `width`, `height`,
and `no_detection_count`.

### Accessing Raw Landmarks

Raw landmark coordinates are stored in `body.positions` using MediaPipe-style
landmark names:

```python
right_shoulder = body.positions["RIGHT_SHOULDER"]
left_knee = body.positions["LEFT_KNEE"]
```

Each value is a list with one coordinate per frame. Coordinates are normalized
MediaPipe coordinates, usually in the range `0.0` to `1.0` for `x` and `y`.

```python
frame_idx = 10

right_shoulder_xy = body.positions["RIGHT_SHOULDER"][frame_idx]
x, y = right_shoulder_xy
```

Common raw landmark keys include:

- `NOSE`
- `LEFT_SHOULDER`, `RIGHT_SHOULDER`
- `LEFT_ELBOW`, `RIGHT_ELBOW`
- `LEFT_WRIST`, `RIGHT_WRIST`
- `LEFT_HIP`, `RIGHT_HIP`
- `LEFT_KNEE`, `RIGHT_KNEE`
- `LEFT_ANKLE`, `RIGHT_ANKLE`
- `LEFT_HEEL`, `RIGHT_HEEL`
- `LEFT_FOOT_INDEX`, `RIGHT_FOOT_INDEX`

The full landmark mapping is stored in:

```text
src/kineticbody/kinetics/bodyparts/config/landmark_mapping.json
```

### Accessing Joints

`body.joints` groups landmark coordinates into joint names. Bilateral joints are
accessed with the pattern:

```text
body.joints.<JointName><Side>
```

Use `Right` or `Left` as the side suffix:

```python
right_shoulder = body.joints.ShoulderRight
left_shoulder = body.joints.ShoulderLeft
right_knee = body.joints.KneeRight
left_ankle = body.joints.AnkleLeft
```

Like `body.positions`, each joint value is a list of coordinates over time:

```python
frame_idx = 10

right_knee_xy = body.joints.KneeRight[frame_idx]
x, y = right_knee_xy
```

Available bilateral joints include:

- `ShoulderRight`, `ShoulderLeft`
- `ElbowRight`, `ElbowLeft`
- `WristRight`, `WristLeft`
- `HipRight`, `HipLeft`
- `KneeRight`, `KneeLeft`
- `AnkleRight`, `AnkleLeft`
- `HeelRight`, `HeelLeft`
- `FootIndexRight`, `FootIndexLeft`
- `PinkyRight`, `PinkyLeft`
- `IndexRight`, `IndexLeft`
- `ThumbRight`, `ThumbLeft`

### Accessing Head Landmarks

`body.head` exposes the nose directly and ears/eyes by side:

```python
nose = body.head.Nose
left_eye = body.head.EyeLeft
right_eye = body.head.EyeRight
left_ear = body.head.EarLeft
right_ear = body.head.EarRight

nose_frame_10 = body.head.Nose[10]
```

### Accessing Limbs

`body.limbs` stores vectors as landmark pairs. A bilateral limb is accessed the
same way as a joint:

```python
right_upper_arm = body.limbs.UpperArmRight
left_forearm = body.limbs.ForearmLeft
right_lower_leg = body.limbs.LowerLegRight
```

Each frame contains the start and end coordinate of the limb:

```python
frame_idx = 10

start_xy, end_xy = body.limbs.UpperArmRight[frame_idx]
```

Available bilateral limbs include:

- `UpperArmRight`, `UpperArmLeft`
- `ForearmRight`, `ForearmLeft`
- `UpperLegRight`, `UpperLegLeft`
- `LowerLegRight`, `LowerLegLeft`
- `TorsoRight`, `TorsoLeft`
- `ThumbRight`, `ThumbLeft`
- `IndexRight`, `IndexLeft`
- `PinkyRight`, `PinkyLeft`
- `HeelRight`, `HeelLeft`
- `FootRight`, `FootLeft`

Some limbs are unilateral and do not use a side suffix:

```python
upper_back = body.limbs.UpperBack
hip_line = body.limbs.Hip
```

### Accessing Angles

`body.angles` stores per-frame joint angles in degrees. Angles are accessed with
the same `<AngleName><Side>` pattern:

```python
right_knee_angles = body.angles.KneeRight
left_elbow_angles = body.angles.ElbowLeft
right_hip_angles = body.angles.HipRight
```

Get the angle at a specific frame:

```python
frame_idx = 10

right_knee_angle = body.angles.KneeRight[frame_idx]
```

Currently available angles are:

- `KneeRight`, `KneeLeft`
- `ElbowRight`, `ElbowLeft`
- `HipRight`, `HipLeft`

### Saving a Model

Save the current `KineticBody` model as JSON:

```python
body.save_model("outputs/body_model.json")
```

## Visualization

After generating a body model JSON file, render the pose visualization on top of
the original video:

```bash
python scripts/visualize_body.py \
  --input-video-path path/to/input_video.mp4 \
  --body-model-path outputs/body_model.json \
  --output_path outputs/visualized_video.mp4
```

The visualization script also accepts these aliases:

```bash
python scripts/visualize_body.py \
  --input_path path/to/input_video.mp4 \
  --body_path outputs/body_model.json \
  --output_path outputs/visualized_video.mp4
```

### Visualization Arguments

- `--input-video-path` or `--input_path`: original video used for pose estimation
- `--body-model-path` or `--body_path`: saved `KineticBody` JSON model
- `--output_path`: destination for the rendered video

The script loads the saved model with `KineticBody.load_model()`, opens the
video with OpenCV, draws the configured body visuals on each frame, and writes a
new video file.

By default, the rendered video includes all configured skeleton parts, joints,
unilateral body parts, and angles from:

```text
src/kineticbody/visualization/config/bodyparts_visuals_config.py
```

## Docker Usage

Build the pose estimation image:

```bash
bash scripts/build_container.bash
```

Run pose estimation in the container:

```bash
bash scripts/run_pose_estimation_container.bash \
  path/to/input_video.mp4 \
  outputs/body_model.json
```

The container runner mounts the input file's directory as read-only at `/data`
and mounts the output directory at `/outputs`. It then runs:

```bash
python scripts/run_pose_estimation.py \
  --input-path /data/<input-file> \
  --output-path /outputs/<output-file>
```

## Python API Example

You can also call the workflow directly from Python:

```python
from kineticbody.workflows.run_pose_estimation import run_pose_estimation

body = run_pose_estimation(
    input_path="path/to/input_video.mp4",
    lag_reduction=True,
    apply_filter=True,
)

body.save_model("outputs/body_model.json")
```

Load a saved model:

```python
from kineticbody.kinetics.body import KineticBody

body = KineticBody.load_model("outputs/body_model.json")
```

## Development

Install the package in editable mode:

```bash
pip install -e .
```

Run the scripts from the repository root so local paths and bundled model files
resolve correctly.

Useful files while developing:

- `src/kineticbody/model/pose_estimation.py`: MediaPipe model loading and image/video inference
- `src/kineticbody/workflows/run_pose_estimation.py`: high-level pose estimation workflow
- `src/kineticbody/workflows/rendering.py`: high-level visualization workflow
- `scripts/run_pose_estimation.py`: command-line entry point for model generation
- `scripts/visualize_body.py`: command-line entry point for rendered videos

## Notes

- Coordinates are stored as normalized landmark positions from MediaPipe.
- Missing video detections are filled with `NaN` during inference and then
  interpolated before model creation.
- Filtering is only applied when enabled and is not used for image inference.
- The visualization workflow expects a video input, not a still image.

## License

Add the project license here.
