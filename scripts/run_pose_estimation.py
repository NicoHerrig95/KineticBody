import os
import sys 
from kineticbody.config.paths import ROOT_DIR
from kineticbody.workflows.run_pose_estimation import run_pose_estimation
import argparse
sys.path.insert(0, ROOT_DIR)
# testing
file_path = "./data/angle_test.mov"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pose estimation on a video."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input video file",
        required=True
    )

    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save positions from KineticBody object.",
        required=True
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.input_path}"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    

    body = run_pose_estimation(
        input_path=args.input_path,
        lag_reduction=True,
        apply_filter=True
    )

    

