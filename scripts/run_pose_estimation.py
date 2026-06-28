from pathlib import Path
import argparse
import os

from kineticbody.workflows.run_pose_estimation import run_pose_estimation




def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pose estimation on a video."
    )

    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        type=str,
        help="Path to the input video file",
    )

    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=str,
        help="Path to save the KineticBody model JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    body = run_pose_estimation(
        input_path=str(input_path),
        lag_reduction=True,
        apply_filter=True,
    )

    # making parent directory
    output_dir = output_path.parent
    print("Output Dir")
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    body.save_model(str(output_path))


if __name__ == "__main__":
    main()
