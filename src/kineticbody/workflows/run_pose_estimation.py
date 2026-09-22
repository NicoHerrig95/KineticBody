from kineticbody.model.pose_estimation import PoseEstimator
from kineticbody.kinetics.body import KineticBody
from kineticbody.model.proc.filtering import SavGol
from kineticbody.config.paths import MODEL_CONFIG_PATH
from kineticbody.utils.common import read_yaml, get_modality
from pathlib import Path

CONFIG = read_yaml(str(MODEL_CONFIG_PATH))
POSE_MODEL_CONFIG = CONFIG["pose_estimator"]
FILTER_CONFIG = CONFIG["filter"]

FILTER_OPTIONS = {
    "SavGol" : SavGol
}



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
        filter_kwargs = {k:v for k,v in FILTER_CONFIG.items()}
        del filter_kwargs["algorithm"]
        print(filter_kwargs)
        filter = FILTER_OPTIONS[filter_algo](**filter_kwargs)

    elif not POSE_MODEL_CONFIG["filter"]:
        filter = None

    # instantiating pose estimator
    estimator = PoseEstimator(modality = modality,
                          reduce_lag = lag_reduction,
                          filter = filter,
                          size = POSE_MODEL_CONFIG["size"]
                          )
    
    return estimator(input_path)

    


    


