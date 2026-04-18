import os 
import sys 
import numpy as np

def linear_interpolation(arr, window=2):
    """ 
    Linear interpolation (using averages of windows before and after gap)


    Parameters
    ----------
    arr : array-like
        One-dimensional sequence of numeric values. Missing values must be
        represented as `np.nan`.
    window : int, default=2
        Number of neighboring values to consider on each side of a gap when
        computing interpolation anchors for interior gaps.

    Returns
    -------
    np.ndarray
        A float NumPy array of the same length as the input, with eligible
        NaN gaps filled.

    Notes
    -----
    - The function always returns a float array because `np.nan` requires
      floating-point representation.
    - Interior interpolation is performed with `np.interp`.
    - This function operates on 1D arrays only.

    """
    out = np.asarray(arr, dtype=float).copy()
    n = len(out)

    if not np.isnan(out).any():
        return out

    gap_start_idx = None
    gap_end_idx = None
    currently_in_gap = False

    def fill_gap(gap_start_idx, gap_end_idx):
        gap_length = gap_end_idx - gap_start_idx + 1

        left_side = out[max(0, gap_start_idx - window):gap_start_idx]
        right_side = out[gap_end_idx + 1:min(n, gap_end_idx + window + 1)]

        # remove NaNs from context windows
        left_side = left_side[~np.isnan(left_side)]
        right_side = right_side[~np.isnan(right_side)]

        # gap at beginning
        if len(left_side) == 0 and len(right_side) > 0:
            fill_value = right_side[0]
            correction = np.full(gap_length, fill_value)

        # gap at end
        elif len(right_side) == 0 and len(left_side) > 0:
            fill_value = left_side[-1]
            correction = np.full(gap_length, fill_value)

        # interior gap
        elif len(left_side) > 0 and len(right_side) > 0:
            correction = np.interp(
                np.arange(gap_start_idx, gap_end_idx + 1),
                [gap_start_idx - 1, gap_end_idx + 1],
                [np.mean(left_side), np.mean(right_side)]
            )

        # all values are NaN
        else:
            correction = None

        if correction is not None:
            if len(correction) != gap_length:
                raise ValueError("Gap length does not match correction length")
            out[gap_start_idx:gap_end_idx + 1] = correction

    for i in range(n):
        v = out[i]

        if np.isnan(v) and not currently_in_gap:
            currently_in_gap = True
            gap_start_idx = i
            gap_end_idx = i

        elif np.isnan(v) and currently_in_gap:
            gap_end_idx = i

        elif not np.isnan(v) and currently_in_gap:
            fill_gap(gap_start_idx, gap_end_idx)
            currently_in_gap = False
            gap_start_idx = None
            gap_end_idx = None

    # handle trailing gap
    if currently_in_gap:
        fill_gap(gap_start_idx, gap_end_idx)

    return out



