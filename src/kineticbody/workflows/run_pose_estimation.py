from kineticbody.model.pose_estimation import PoseEstimator
from kineticbody.kinetics.body import KineticBody
from kineticbody.model.proc.filtering import SavGol
from kineticbody.config.paths import MODEL_CONFIG_PATH
from kineticbody.utils.common import read_yaml
from pathlib import Path

CONFIG = read_yaml(MODEL_CONFIG_PATH)
POSE_MODEL_CONFIG = CONFIG["pose_estimator"]
FILTER_CONFIG = CONFIG["filter"]

FILTER_OPTIONS = {
    "SavGol" : SavGol
}


def get_modality(input_path: str) -> str:
    """
    Determine modality based on file extension.

    Returns:
        "video" for video files
        "image" for image files

    Raises:
        ValueError if the file type is unsupported.
    """

    video_suffix_list = [".mp4", ".mov"]
    image_suffix_list = [".png", ".jpg", ".jpeg"]

    suffix = Path(input_path).suffix.lower()

    if suffix in video_suffix_list:
        return "video"

    if suffix in image_suffix_list:
        return "image"

    raise ValueError(f"Unsupported file type: {suffix}")



def run_pose_estimation(input_path:str,
                        lag_reduction:bool,
                        apply_filter:bool
                        ) -> KineticBody:
    """ 
    Generates a KineticBody object from input.
    """

    modality = get_modality(input_path)
    
    # overwriting config if needed
    POSE_MODEL_CONFIG["reduce_lag"] = lag_reduction
    POSE_MODEL_CONFIG["filter"] = apply_filter

    # instantiating filter
    if POSE_MODEL_CONFIG["filter"]:
        filter_algo = FILTER_CONFIG["algorithm"]
        filter_kwargs = {k:v for k,v in FILTER_CONFIG.items() if k is not "algorithm"}
        filter = FILTER_OPTIONS[filter_algo](**filter_kwargs)

    # instantiating pose estimator
    estimator = PoseEstimator(modality = modality,
                          reduce_lag = lag_reduction,
                          filter = filter,
                          size = POSE_MODEL_CONFIG["size"]
                          )
    
    return estimator(input_path)

    


    


