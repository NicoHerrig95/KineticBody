from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "angle_test.mov"
DEFAULT_TESTING_DIR = PROJECT_ROOT / "testing"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run integration tests for pose estimation and body visualization scripts."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to the test video.",
    )
    parser.add_argument(
        "--testing-dir",
        type=Path,
        default=DEFAULT_TESTING_DIR,
        help="Directory where test outputs and logs are saved.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the scripts under test.",
    )
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> dict:
    started_at = time.time()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=get_test_env(),
        capture_output=True,
        text=True,
    )
    duration_seconds = time.time() - started_at

    log_path.write_text(
        "\n".join(
            [
                f"Command: {' '.join(command)}",
                f"Return code: {result.returncode}",
                f"Duration seconds: {duration_seconds:.2f}",
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )

    return {
        "command": command,
        "return_code": result.returncode,
        "duration_seconds": duration_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log_path": str(log_path),
        "passed": result.returncode == 0,
    }


def get_test_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path("/private/tmp") / "kineticbody_pycache"))
    return env


def assert_file_created(path: Path, description: str) -> None:
    if not path.exists():
        raise AssertionError(f"{description} was not created: {path}")
    if path.stat().st_size == 0:
        raise AssertionError(f"{description} is empty: {path}")


def reset_test_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def assert_valid_json(path: Path, description: str) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{description} is not valid JSON: {path}") from exc


def find_visualized_video(requested_output_path: Path) -> Path:
    candidates = [
        requested_output_path,
        requested_output_path.with_suffix(".avi"),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    raise AssertionError(
        "Visualized video was not created. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def main():
    args = parse_args()

    input_path = args.input_path.resolve()
    testing_dir = args.testing_dir.resolve()
    testing_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Test video does not exist: {input_path}")

    body_model_path = testing_dir / "body_model.json"
    visualized_video_path = testing_dir / "visualized_body.mp4"
    summary_path = testing_dir / "summary.json"
    reset_test_artifacts(
        [
            body_model_path,
            visualized_video_path,
            visualized_video_path.with_suffix(".avi"),
            summary_path,
            testing_dir / "run_pose_estimation.log",
            testing_dir / "visualize_body.log",
        ]
    )

    tests = []

    print("RUNNING TEST FOR run_pose_estimation.py")
    pose_command = [
        args.python,
        "scripts/run_pose_estimation.py",
        "--input-path",
        str(input_path),
        "--output-path",
        str(body_model_path),
    ]
    pose_result = run_command(pose_command, testing_dir / "run_pose_estimation.log")
    tests.append({"name": "run_pose_estimation", **pose_result})
    if pose_result["return_code"] != 0:
        write_summary(summary_path, input_path, body_model_path, None, tests)
        raise RuntimeError(f"run_pose_estimation failed. See {pose_result['log_path']}")
    assert_file_created(body_model_path, "Body model JSON")
    assert_valid_json(body_model_path, "Body model JSON")

    print("RUNNING TEST FOR visualize_body.py")

    visualize_command = [
        args.python,
        "scripts/visualize_body.py",
        "--input-path",
        str(input_path),
        "--model-path",
        str(body_model_path),
        "--output-path",
        str(visualized_video_path),
    ]
    visualize_result = run_command(visualize_command, testing_dir / "visualize_body.log")
    tests.append({"name": "visualize_body", **visualize_result})
    if visualize_result["return_code"] != 0:
        write_summary(summary_path, input_path, body_model_path, None, tests)
        raise RuntimeError(f"visualize_body failed. See {visualize_result['log_path']}")

    actual_visualized_video_path = find_visualized_video(visualized_video_path)
    write_summary(
        summary_path,
        input_path,
        body_model_path,
        actual_visualized_video_path,
        tests,
    )

    print(f"Testing complete. Results saved to: {testing_dir}")
    print(f"Body model: {body_model_path}")
    print(f"Visualized video: {actual_visualized_video_path}")
    print(f"Summary: {summary_path}")


def write_summary(
    summary_path: Path,
    input_path: Path,
    body_model_path: Path,
    visualized_video_path: Path | None,
    tests: list[dict],
) -> None:
    summary = {
        "passed": all(test["passed"] for test in tests),
        "input_video": str(input_path),
        "outputs": {
            "body_model": str(body_model_path),
            "visualized_video": str(visualized_video_path) if visualized_video_path else None,
        },
        "tests": tests,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
