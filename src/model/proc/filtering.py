import os 
import sys 
import numpy as np
from scipy.signal import savgol_filter



class SavGol(object):

    def __init__(self, 
                window_size:int = 9, # recommended 7, 9, 11
                polyorder:int = 2 # recommended 2, 3
                ):
        self.window_size = window_size
        self.polyorder = polyorder


    def __call__(self, positions:dict):
        smoothed_positions = {}
        for lm in positions:
            N = len(positions[lm])

            window_size = min(self.window_size, N)
            x = [positions[lm][i][0] for i in range(N)]
            y = [positions[lm][i][1] for i in range(N)]
            input_array = np.stack([x, y], axis=1)
            # applyging filter
            coords_smooth = savgol_filter(
                input_array,
                window_length=window_size,
                polyorder=self.polyorder,
                axis=0,  # smooth over time
                mode="interp"
            )
            x_smoothed = [coords_smooth[i][0] for i in range(len(coords_smooth))]
            y_smoothed = [coords_smooth[i][1] for i in range(len(coords_smooth))]
            coords_smoothed = [[float(x), float(y)] for x,y in zip(x_smoothed,y_smoothed)]
            smoothed_positions[lm] = coords_smoothed
        return smoothed_positions




