# -*- coding: utf-8 -*-
"""
Contains:
    function:
        windowing          - Slice continuous data into fixed-length overlapping windows    
        group_consecutive  - Group consecutive intergers into separate arrays

Created on Thu Jan  8 14:53:02 2026

@author: Jeyoon Lee
"""
import numpy as np
import pandas as pd
    
def windowing(
        array: np.ndarray,
        sfreq: float,
        window_len: float,
        overlap: float,
        axis: int = -1):
    """
    Slice continuous data into fixed-length overlapping windows.

    Parameters
    ----------
    array : np.ndarray
        Input data array. Example: (n_channels, n_times)
    sfreq : float | int
        sampling frequency in Hz.
    window_len : float | int
        window length in seconds.
    overlap : float | int
        overlapping window in seconds.
    axis : int, optional
        Axis along with time is represented. The default is -1.

    Returns
    -------
    windows : numpy.ndarray
        The windowed data. 
        If input is (n_channels, n_times), output becomes (n_epochs, n_channels, window_size)
    win_start_samp : numpy.ndarray
        Starting sample index of each window (relative to the original array).
    win_start_sec : numpy.ndarray
        Starting time in seconds for each window.
    """

    # Move time axis to the last position
    arr = np.moveaxis(array, axis, -1)
    n_times = arr.shape[-1]

    window_size = int(round(float(window_len) * float(sfreq)))
    overlap_size = int(round(float(overlap) * float(sfreq)))

    if window_size <= 0:
        raise ValueError("window_len is too small (window_size <= 0).")
    if overlap_size < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap_size >= window_size:
        raise ValueError("overlap must be smaller than window_len.")

    step = window_size - overlap_size
    if n_times < window_size:
        raise ValueError(
            f"Data length ({n_times} samples) is shorter than window_size ({window_size})."
        )

    # Compute window start indices
    starts = np.arange(0, n_times - window_size + 1, step, dtype=int)

    # Extract windows
    win_list = [arr[..., s:s + window_size] for s in starts]
    windows = np.stack(win_list, axis=0)

    win_start_samp = starts
    win_start_sec = starts / float(sfreq)

    return windows, win_start_samp, win_start_sec

def group_consecutive(nums: np.ndarray|list)->list[np.ndarray]:
    """
    Example
    ----------
    [1,2,3,5,6,7,9,10] -> [[1,2,3], [5,6,7], [9,10]]
    """
    nums = np.asarray(nums, dtype=int)
    if nums.size == 0:
        return []
    
    nums = np.unique(nums)
    splits = np.where(np.diff(nums) != 1)[0]+1
    
    return np.split(nums, splits)
    

