import os 
import sys
import cv2
from tqdm import tqdm
from bodyscan.utils.common import save_dict_to_json
from bodyscan.model.proc.filtering import SavGol
from bodyscan.model.pose_estimation import PoseEstimator
from bodyscan.config.paths import TMP_DIR
from bodyscan.assets.visualization import (
    visualize_video,
    skeletton_default,
    angles_default,
    joints_default,
    unilaterals_default
)
from dotenv import load_dotenv
load_dotenv()

# Importing rules for different perspectives
from bodyscan.movements.squat.side_view import SIDE_VIEW_RULES

DATA_DIR_CLOUD = os.getenv("DATA_ON_CLOUD")
SIDE_VIEW_PATH = os.path.join(DATA_DIR_CLOUD, "self_filmed_videos_squat/side_view_less_clothes.mov")

RULES = {"side_view" : SIDE_VIEW_RULES}
INPUT_PATHS = {"side_view" : SIDE_VIEW_PATH}
VISUALIZATION_KWARGS = {
        "side_view" : {
            "skeletton" : skeletton_default,
            "joints": joints_default, 
            "unilaterals" : unilaterals_default,
            "angles" : angles_default,
        }
    }


MODEL_KWARGS = {
    "modality" : "video",
    "filter" : SavGol(),
    "reduce_lag" : True
}


def squat_analysis(
        rules:dict,
        input_paths:dict,
        model,
        temp_dir:str,
        visualize = False
    ):
    

    if rules.keys() != input_paths.keys():
        raise ValueError("Rules and input_paths must have the same keys.")

    perspectives = list(input_paths.keys())
    analysis_results = {} # main storer for results
    error_log = {p : {} for p in perspectives}
    os.makedirs(temp_dir, exist_ok=True)


    # -> Main loop
    progress_bar = tqdm(perspectives)
    for p in progress_bar:
        progress_bar.set_description(f"Computing {p}")

        body = None
        analysis_results[p] = {}
        
        try:
        # model inference
            perspective_input_path = input_paths[p]
            body = model(perspective_input_path)
        except Exception as e:
            error_log[p].update({"inference" : e}) 

        ##############################
        # perspective analytics
        ##############################
        
        if body is not None:
            for r in rules[p]:
                rule_name = None
                

                # Kinetic Body Computation
                try:
                    rule = r(body)
                    rule_name = str(rule.name)
                    analysis_results[p][rule_name] = rule()
                except Exception as e:
                    if rule_name is not None:
                        error_log[p].update({rule_name : e})
                    elif rule_name is None:
                        # if error occured before rule instantiation
                        error_log[p].update({r.__name__ : e})

        # Body visualization
        if visualize and (set(perspectives) == visualize.keys()):
            perspective_visualisation_kwargs = visualize[p]
            visualize_video(
                body = body,
                capture=cv2.VideoCapture(perspective_input_path),
                out_path= temp_dir / f"{p}.mp4",
                **perspective_visualisation_kwargs
            )

    return analysis_results, error_log


if __name__ == "__main__":

    analysis_results, error_log = squat_analysis(
        rules = RULES,
        input_paths=INPUT_PATHS,
        model = PoseEstimator(**MODEL_KWARGS),
        temp_dir=TMP_DIR/"testing",
        visualize=VISUALIZATION_KWARGS
    )
    save_dict_to_json(
        analysis_results, 
        TMP_DIR/"analysis_results.json"
        )
    print("-- Analysis successfully executed --")
