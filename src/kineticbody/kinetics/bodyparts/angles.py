from kineticbody.kinetics.bodyparts.base import Bodypart, get_angle


################################################################################
# ANGLE CLASS
################################################################################

BILATERAL_ANGLES = {
    "Knee" : ("HIP", "KNEE", "ANKLE"),
    "Elbow" : ("SHOULDER", "ELBOW", "WRIST"),
    "Hip" : ("SHOULDER", "HIP", "KNEE")
    }


class Angles(Bodypart):
    def __init__(self, positions:dict, N_frames):
        super().__init__()
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodypart()

    def _initialize_bodypart(self):

        N_frames = self.N_frames
        for side in ["RIGHT", "LEFT"]:
            # Bilateral Angles
            for angle_name, (start, center, end) in BILATERAL_ANGLES.items():
                self._objects.setdefault(angle_name, {})[side] = [get_angle(
                    coord_a = self.positions[f"{side}_{start}"][i],
                    coord_b = self.positions[f"{side}_{center}"][i],
                    coord_c = self.positions[f"{side}_{end}"][i]
                ) for i in range(N_frames)]