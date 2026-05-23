from kineticbody.kinetics.bodyparts.base import Bodypart, get_vector

################################################################################
# LIMBS CLASS
################################################################################


BILATERAL_LIMBS = {
    "UpperArm": ("SHOULDER", "ELBOW"),
    "Forearm": ("ELBOW", "WRIST"),
    "UpperLeg": ("HIP", "KNEE"),
    "LowerLeg": ("KNEE", "ANKLE"),
    "Torso": ("SHOULDER", "HIP"),
    "Thumb": ("WRIST", "THUMB"),
    "Index": ("WRIST", "INDEX"),
    "Pinky": ("WRIST", "PINKY"),
    "Heel": ("ANKLE", "HEEL"),
    "Foot": ("HEEL", "FOOT_INDEX")
}

UNILATERAL_LIMBS = {
    "UpperBack" : ("LEFT_SHOULDER","RIGHT_SHOULDER"),
    "Hip" : ("LEFT_HIP", "RIGHT_HIP")
}


class Limbs(Bodypart):
    def __init__(self, positions: dict, N_frames: int):
        super().__init__()
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodypart()

    def _initialize_bodypart(self):
        N_frames = self.N_frames


        for side in ["RIGHT", "LEFT"]:
            # Bilateral Limbs
            for limb_name, (start, end) in BILATERAL_LIMBS.items():
                self._objects.setdefault(limb_name, {})[side] = get_vector(
                    f"{side}_{start}",
                    f"{side}_{end}",
                    self.positions,
                    N_frames
                )

        # Unilateral limbs
        self._objects["UpperBack"] = get_vector(UNILATERAL_LIMBS["UpperBack"][0], UNILATERAL_LIMBS["UpperBack"][1], self.positions, N_frames)
        self._objects["Hip"] = get_vector(UNILATERAL_LIMBS["Hip"][0], UNILATERAL_LIMBS["Hip"][1], self.positions, N_frames)