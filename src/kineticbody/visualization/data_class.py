from dataclasses import dataclass
import numpy as np

@dataclass
class Video:
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int