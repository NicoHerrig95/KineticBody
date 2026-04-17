import os
import sys 
import numpy as np




def check_joint_angle_range(angles:list, constraints:dict) -> dict:
    """ 
    Logic for checking if angle falls in given constraint range.
    Takes mean value of the 5 frames around minimum angle as reference value, if possible.
    """

    # checking if at least 5 frames lie within angle constaint range
    minimum_angle = min(angles)
    idx_smallest_angle = angles.index(minimum_angle)
    if idx_smallest_angle > 2 and (idx_smallest_angle + 3 < len(angles)):
        idx_start = idx_smallest_angle - 2
        idx_end = idx_smallest_angle + 2
    else:
        idx_start = None
        idx_end = None

    # evaluating movement depth by interior angle
    movement_depth = None
    flexion_types = list(constraints.keys())

    for flexion in flexion_types:
        constraint_max_angle = constraints[flexion]["angle"][0]
        constraint_min_angle = constraints[flexion]["angle"][1]
        if idx_start != None and idx_end != None:
            mean_angle = np.mean(angles[idx_start:idx_end+1])
            if float(constraint_min_angle) <= mean_angle <= float(constraint_max_angle):
                movement_depth = flexion 
        elif idx_start == None or idx_end == None:
            if minimum_angle < constraint_min_angle and minimum_angle > constraint_max_angle:
                movement_depth = flexion


    return {
                "flexion" : movement_depth,
                "flexion_range" : constraints[movement_depth]["angle"],
                "encoding" : constraints[movement_depth]["encoding"],
            }