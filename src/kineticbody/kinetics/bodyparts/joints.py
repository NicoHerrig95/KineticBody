from kineticbody.kinetics.bodyparts.base import Bodypart

################################################################################
# JOINTS CLASS
################################################################################


BILATERAL_JOINTS = {
    'Elbow': 'ELBOW',
    'Wrist': 'WRIST',
    'Pinky': 'PINKY',
    'Index': 'INDEX',
    'Thumb': 'THUMB',
    'Hip': 'HIP',
    'Shoulder': 'SHOULDER',
    'Knee': 'KNEE',
    'Ankle': 'ANKLE',
    'Heel': 'HEEL',
    'FootIndex': 'FOOT_INDEX'
}
class Joints(Bodypart):
    """Stores the body's joints' coordinates."""

    def __init__(self, positions: dict):
        super().__init__()
        self.positions = positions
        self._initialize_bodypart()

    def _initialize_bodypart(self):



        for side in ["RIGHT", "LEFT"]:
            for joint_name, pose_marker_name in BILATERAL_JOINTS.items():
                full_name = f"{side}_{pose_marker_name}"
                self._objects.setdefault(joint_name, {})[side] = self.positions[full_name]
