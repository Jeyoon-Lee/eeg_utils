# -*- coding: utf-8 -*-
"""
eeg_utils/sleep_staging.py

Contains:
    class:
        SleepDetector       - Wraps YASA sleep staging and provides a
                              post-processing step to relabel REM epochs
                              that immediately follow Wake as N1.
    function:
        upsample_hypnogram  - Upsample an epoch-wise hypnogram to a
                              sample-wise array.
"""

import mne
import yasa
import numpy as np
from typing import Any, Literal


class SleepDetector:

    def __init__(self,
                 raw: mne.io.BaseRaw):
        self.raw = raw.copy()

        # output
        self.hypno = None
        self.hypno_fixed = None

    def sleep_staging(self,
                     eeg_name: str = 'C4'):
        sls = yasa.SleepStaging(self.raw, eeg_name=eeg_name)
        self.hypno = np.asarray(sls.predict(), dtype=object)
        self.hypno_fixed = None
        return self

    def relabel_R_after_W(self):
        """Relabel consecutive REM ('R') blocks that immediately follow Wake ('W') as N1."""
        if self.hypno is None:
            raise RuntimeError("Run .sleep_staging() first.")

        hypno = np.array(self.hypno, dtype=object).copy()

        i = 1
        while i < len(hypno):
            if hypno[i-1] == 'W' and hypno[i] == 'R':
                j = i
                while j < len(hypno) and hypno[j] == 'R':
                    hypno[j] = 'N1'
                    j += 1
                i = j
            else:
                i += 1

        self.hypno_fixed = hypno
        return self


def upsample_hypnogram(hypno: np.ndarray,
                       data_len: int,
                       data_fs: float,
                       hypno_len: float = 30.0,
                       fill_None: Literal['UNK', 'before'] = 'UNK',
                       unknown_label: Any = "UNK"):
    """
    Upsample an epoch-wise hypnogram to a sample-wise array.

    Parameters
    ----------
    hypno : np.ndarray
        1D array of sleep stage labels per epoch (e.g., strings like "NREM"/"REM"/"WAKE",
        or numeric codes). Each element corresponds to one epoch.
    data_len : int
        Total number of samples in the target data (e.g., len(raw)).
    data_fs : float
        Sampling frequency (Hz) of the target data.
    hypno_len : float, optional
        Epoch length in seconds used to score the hypnogram. The default is 30.0.
    fill_None : Literal['UNK', 'before'], optional
        How to fill samples after the last available hypnogram epoch when data_len is longer
        than the hypnogram coverage.
        - 'UNK': fill remaining samples with 'UNK'
        - 'before': fill remaining samples with the last available stage label
        The default is 'UNK'.
    unknown_label : Any, optional
        Label/value used when fill_None="UNK". Default is "UNK".
    """
    hypno = np.asarray(hypno)

    samples_per_epoch = int(round(hypno_len * data_fs))
    if samples_per_epoch <= 0:
        raise ValueError("Epoch length * sampling rate resulted in 0 samples. Check inputs.")

    hypno_samples = np.repeat(hypno, samples_per_epoch)
    hypno_upsampled = np.full((data_len,), unknown_label, dtype=object)
    n_copy = min(data_len, hypno_samples.size)
    hypno_upsampled[:n_copy] = hypno_samples[:n_copy]

    if fill_None == 'UNK':
        return hypno_upsampled
    elif fill_None == 'before':
        if n_copy > 0:
            hypno_upsampled[n_copy:] = hypno_upsampled[n_copy-1]
        return hypno_upsampled
    else:
        raise ValueError("'fill_None' should be either 'UNK' or 'before'.")
