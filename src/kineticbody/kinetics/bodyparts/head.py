from kineticbody.kinetics.bodyparts.base import Bodypart


class Head(Bodypart):

    def __init__(self, positions: dict, N_frames: int):
        super().__init__()
        self.positions = positions
        self.N_frames = N_frames
        self._initialize_bodypart()
    
    def _initialize_bodypart(self):
        # Nose
        self._objects.update({
            "Nose" :  self.positions["NOSE"]
            })
        # Ears & Eyes (midpoint)
        for side in ["LEFT", "RIGHT"]:
            self._objects.setdefault("Ear", {})[side] = self.positions[f"{side}_EAR"]
            self._objects.setdefault("Eye", {})[side] = self.positions[f"{side}_EYE"]



