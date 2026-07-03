from pathlib import Path
import argparse
import os

import cv2

from kineticbody.kinetics.body import KineticBody
from kineticbody.workflows.rendering import visualize_video
from kineticbody.visualization.functions import make_writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a KineticBody model on top of a video."
    )

    parser.add_argument(
        "--input_path",
        "--input-video-path",
        dest="input_video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--body_path",
        "--body-model-path",
        dest="body_model_path",
        type=str,
        required=True,
        help="Path to the saved KineticBody model JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the visualized output video.",
    )

    return parser.parse_args()


def write_video(video, output_path: Path) -> Path:
    os.makedirs(output_path.parent, exist_ok=True)

    writer, actual_output_path = make_writer(
        path=str(output_path),
        fps=video.fps,
        width=video.width,
        height=video.height,
    )

    try:
        for frame in video.frames:
            writer.write(frame)
    finally:
        writer.release()

    return Path(actual_output_path)


def main():
    args = parse_args()

    input_video_path = Path(args.input_video_path)
    body_model_path = Path(args.body_model_path)
    output_path = Path(args.output_path)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video_path}")
    if not body_model_path.exists():
        raise FileNotFoundError(f"Body model does not exist: {body_model_path}")

    body = KineticBody.load_model(str(body_model_path))
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    video = visualize_video(body=body, capture=capture)
    actual_output_path = write_video(video, output_path)

    print(f"Saved visualized video to: {actual_output_path}")


if __name__ == "__main__":
    main()
