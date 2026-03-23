import numpy as np

"""  
Utility functions for bodypart computation.
"""

################################################################################
# HELPERS
################################################################################
def get_vector(lm_a, lm_b, positions, N:int) -> list:
    """ 
    Gets a vector from positions by definition of two landmark keys. 
    """
    return [(positions[lm_a][i], positions[lm_b][i]) for i in range(N)]



def get_angle(coord_a:tuple, coord_b:tuple, coord_c:tuple):
    """
    Computes angle from three coordinates (x,y).
    NOTE: Returns the angle for lm_b!
    """

    a = np.array(coord_a)
    b = np.array(coord_b)
    c = np.array(coord_c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle