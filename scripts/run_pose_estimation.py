import os
import sys 
from kineticbody.config.paths import ROOT_DIR
from kineticbody.workflows.run_pose_estimation import run_pose_estimation
import argparse
sys.path.insert(0, ROOT_DIR)
# testing
file_path = "./data/angle_test.mov"



if __name__ == "__main__":
    body = run_pose_estimation(
        input_path=file_path,
        lag_reduction=True,
        apply_filter=True
    )

    print(body)

