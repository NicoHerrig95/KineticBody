import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np 
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.common import read_json
import time
import cv2








LANDMARK_MAPPING = read_json(os.path.join(BASE_DIR, "POSE_landmark_mapping.json"))


# NOTE: These are fixed numbers but they should be proportional to width and height
LIMB_THICKNESS = {
    "UPPER_ARM" : 20,
    "FOREARM" : 20,
    "UPPER_LEG": 40,
    "LOWER_LEG" : 30
}
BODY_COLOR = (0, 255, 0) # GREEN



def get_vector(lm_a, lm_b, positions, N:int) -> list:
    """ 
    Gets a vector from positions by definition of two landmark keys. 
    """
    return [(positions[lm_a][i], positions[lm_b][i]) for i in range(N)]


def get_pixel_coordinates(coordinates:tuple, w:int, h:int) -> tuple:
    x = coordinates[0]
    y = coordinates[1]
    return (int(x * w), int(y * h))






class KineticBody(object):
    """  
    Kinetic model of human body
    """

    def __init__(
        self,
        positions: dict, # positions dictionary 
        N_frames: int, # 
        ):


        self.positions = positions
        self.N_frames = N_frames
        # initialising body parts
        self._initialize_bodyparts()

    def _initialize_bodyparts(self):
        # Initialize limb dictionaries
        self._upper_arm = {}
        self._upper_leg = {}
        self._lower_leg = {}
        self._torso = {}
        self._upper_back = {}
        self._hip = {}
        self._forearm = {}
        self._thumb = {}
        self._index = {}
        self._pinky = {}
        self._foot = {}
        self._heel = {}
        N_frames = self.N_frames


        # > Bilateral body parts()
        for side in ["RIGHT", "LEFT"]:
            # Upper Arm
            self._upper_arm[side] = get_vector(
                f"{side}_SHOULDER", 
                f"{side}_ELBOW", 
                self.positions, 
                N_frames
            )
        
            # Forearm
            self._forearm[side] = get_vector(
                f"{side}_ELBOW", 
                f"{side}_WRIST", 
                self.positions,
                N_frames
            )

            # Upper Leg (Hip to Knee)
            self._upper_leg[side] = get_vector(
                f"{side}_HIP", 
                f"{side}_KNEE", 
                self.positions, 
                N_frames
            )
        
            # Lower Leg (Knee to Ankle)
            self._lower_leg[side] = get_vector(
                f"{side}_KNEE", 
                f"{side}_ANKLE", 
                self.positions, 
                N_frames
            )
        
            # Torso (Shoulder to Hip)
            self._torso[side] = get_vector(
                f"{side}_SHOULDER", 
                f"{side}_HIP", 
                self.positions, 
                N_frames
            )


            # ----- HAND ------
            # Thumb
            self._thumb[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_THUMB",
                self.positions,
                N_frames
            )
            # Index 
            self._index[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_INDEX",
                self.positions,
                N_frames
            )
            #Pinky
            self._pinky[side] = get_vector(
                f"{side}_WRIST",
                f"{side}_PINKY",
                self.positions,
                N_frames
            )

            # ----- FOOT ------
            # Heel
            self._heel[side] = get_vector(
                f"{side}_ANKLE",
                f"{side}_HEEL",
                self.positions,
                N_frames
            )

            # Foot (Heel to Foot Index)
            self._foot[side] = get_vector(
                f"{side}_HEEL",
                f"{side}_FOOT_INDEX",
                self.positions,
                N_frames
            )


        # > Unilateral body parts
        # Upper back (left to right shoulder)
        self._upper_back = get_vector(
            "LEFT_SHOULDER", 
            "RIGHT_SHOULDER", 
            self.positions, 
            N_frames
        )

        # Hip 
        self._hip = get_vector(
            "LEFT_HIP",
            "RIGHT_HIP",
            self.positions,
            N_frames
        )

    def visualize(self, background:Optional[np.ndarray] = None):
        """ 
        image -> cv2.imread() output
        """

        if isinstance(background, np.ndarray):
            image = background
            h, w = image.shape[:2]


        elif background is None:
            # black background
            image = np.zeros((800, 800, 3), dtype=np.uint8)
            h, w = image.shape[:2]


        #####################################################
        # Upper Arm
        #####################################################
        # Upper Right Arm
        p1 = get_pixel_coordinates(
            coordinates=self.UpperArmRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.UpperArmRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["UPPER_ARM"])  # Green

        # Upper Left Arm
        p1 = get_pixel_coordinates(
            coordinates=self.UpperArmLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.UpperArmLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["UPPER_ARM"])  # Green

        #####################################################
        # Forearm
        #####################################################
        p1 = get_pixel_coordinates(
            coordinates=self.ForearmRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.ForearmRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["FOREARM"])  # Green

        # Left Forearm
        p1 = get_pixel_coordinates(
            coordinates=self.ForearmLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.ForearmLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["FOREARM"])  # Green

        #####################################################
        # Upper Leg
        #####################################################
        p1 = get_pixel_coordinates(
            coordinates=self.UpperLegRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.UpperLegRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["UPPER_LEG"])  # Blue

        # Upper Left Leg
        p1 = get_pixel_coordinates(
            coordinates=self.UpperLegLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.UpperLegLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["UPPER_LEG"])  # Blue

        #####################################################
        # Lower Leg
        #####################################################
        p1 = get_pixel_coordinates(
            coordinates=self.LowerLegRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.LowerLegRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["LOWER_LEG"])  # Blue

        # Lower Left Leg
        p1 = get_pixel_coordinates(
            coordinates=self.LowerLegLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.LowerLegLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, LIMB_THICKNESS["LOWER_LEG"])  # Blue

        #####################################################
        # Torso
        #####################################################
        # > Areas
        # Torso
        left_shoulder = get_pixel_coordinates(self.UpperBack[0][0], w, h)
        right_shoulder = get_pixel_coordinates(self.UpperBack[0][1], w, h)
        left_hip = get_pixel_coordinates(self.Hip[0][0], w, h)
        right_hip = get_pixel_coordinates(self.Hip[0][1], w, h)

        pts = np.array([
            left_shoulder,      # Top left
            right_shoulder,     # Top right
            right_hip,          # Bottom right
            left_hip            # Bottom left
        ], dtype=np.int32)

        cv2.fillPoly(image, [pts], color=BODY_COLOR) # RED

        #####################################################
        # Hand Features
        #####################################################
        # Thumbs
        p1 = get_pixel_coordinates(
            coordinates=self.ThumbLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.ThumbLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        p1 = get_pixel_coordinates(
            coordinates=self.ThumbRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.ThumbRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        # Indexes
        p1 = get_pixel_coordinates(
            coordinates=self.IndexLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.IndexLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        p1 = get_pixel_coordinates(
            coordinates=self.IndexRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.IndexRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        # Pinkies
        p1 = get_pixel_coordinates(
            coordinates=self.PinkyLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.PinkyLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        p1 = get_pixel_coordinates(
            coordinates=self.PinkyRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.PinkyRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 10)  # Green

        #####################################################
        # Feet
        #####################################################
        # Left Heel
        p1 = get_pixel_coordinates(
            coordinates=self.HeelLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.HeelLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 15)  # Green

        # Right Foot
        p1 = get_pixel_coordinates(
            coordinates=self.HeelRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.HeelRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 15)  # Green

        # Left Foot
        p1 = get_pixel_coordinates(
            coordinates=self.FootLeft[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.FootLeft[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 15)  # Green

        # Right Foot
        p1 = get_pixel_coordinates(
            coordinates=self.FootRight[0][0],
            w=w,
            h=h
        )
        p2 = get_pixel_coordinates(
            coordinates=self.FootRight[0][1],
            w=w,
            h=h
        )
        cv2.line(image, p1, p2, BODY_COLOR, 15)  # Green

        cv2.imshow("Kinetic Body", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    ############################################################## 
    # Properties
    ##############################################################
    @property
    def UpperArmRight(self):
        return self._upper_arm["RIGHT"]

    @property
    def UpperArmLeft(self):
        return self._upper_arm["LEFT"]

    @property
    def ForearmRight(self):
        return self._forearm["RIGHT"]

    @property
    def ForearmLeft(self):
        return self._forearm["LEFT"]

    @property
    def UpperLegRight(self):
        return self._upper_leg["RIGHT"]

    @property
    def UpperLegLeft(self):
        return self._upper_leg["LEFT"]

    @property
    def LowerLegRight(self):
        return self._lower_leg["RIGHT"]

    @property
    def LowerLegLeft(self):
        return self._lower_leg["LEFT"]

    @property
    def TorsoRight(self):
        return self._torso["RIGHT"]

    @property
    def TorsoLeft(self):
        return self._torso["LEFT"]

    @property
    def ThumbLeft(self):
        return self._thumb["LEFT"]
    
    @property
    def ThumbRight(self):
        return self._thumb["RIGHT"]

    @property
    def IndexLeft(self):
        return self._index["LEFT"]
    
    @property
    def IndexRight(self):
        return self._index["RIGHT"]

    @property
    def PinkyLeft(self):
        return self._pinky["LEFT"]
    
    @property
    def PinkyRight(self):
        return self._pinky["RIGHT"]

    @property
    def UpperBack(self):
        return self._upper_back

    @property
    def Hip(self):
        return self._hip

    @property
    def HeelLeft(self):
        return self._heel["LEFT"]

    @property
    def HeelRight(self):
        return self._heel["RIGHT"]


    @property
    def FootLeft(self):
        return self._foot["LEFT"]

    @property
    def FootRight(self):
        return self._foot["RIGHT"]


if __name__ == "__main__":
    pass
