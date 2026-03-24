# -*- coding: utf-8 -*-
"""
eeg_utils/eeg_io.py

Contains:
    class:
        EEG_Loader  - Container for EEG data and metadata, with utilities to
                      convert to MNE RawArray.
    function:
        csv_to_raw  - Convenience wrapper that reads a CSV file and returns a
                      fully-annotated mne.io.RawArray.
"""

import numpy as np
import csv
import mne
from config import EEG_CFG
from pathlib import Path
from utils.utils import group_consecutive
from typing import List, Dict, Optional, Sequence

# ====== CONFIG ======
trigger_labels = EEG_CFG.trigger_labels
channel_names = EEG_CFG.channel_names
channel_types = EEG_CFG.channel_types
montage_set = EEG_CFG.montage

signal_loc = EEG_CFG.signal_loc
fs = EEG_CFG.fs


class EEG_Loader:
    """
    Container for EEG data and metadata, with utilities to convert to MNE.

    Attributes
    ----------
    data : np.ndarray
        Signal array with shape (n_ch, n_t) [V].
    fs : float
        Sampling frequency [Hz].
    channel_names : np.ndarray
        Channel name for each row of `data`.
    channel_types : np.ndarray
        Channel type for each row of `data`
        (e.g., 'eeg', 'ecg', 'trigger').
    time : np.ndarray
        Time vector in seconds, shape (n_t,).
    trigger_labels : dict
        Mapping from description (str) to trigger code (int).
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        channel_names: Sequence[str],
        channel_types: Sequence[str],
        trigger_labels: Optional[Dict[str, int]] = None,
    ):
        assert data.ndim == 2, "data must be 2D (n_ch, n_t)."
        assert len(channel_names) == data.shape[0], \
            "Length of channel_names must match number of channels in data."
        assert len(channel_types) == data.shape[0], \
            "Length of channel_types must match number of channels in data."

        self.data = np.asarray(data, dtype=float)
        self.fs = float(fs)
        self.channel_names = np.asarray(channel_names)
        self.channel_types = np.asarray(channel_types)
        self.time = np.arange(self.data.shape[1]) / self.fs
        self.trigger_labels = trigger_labels or {}

    @staticmethod
    def read_csv(f_path: str, delimiter: str = "\t") -> np.ndarray:
        """
        Read a CSV file and return data as a NumPy array (n_ch x n_t).

        Assumes the raw numeric values are in microvolts [µV].

        Parameters
        ----------
        f_path : str
            Path to the CSV file.
        delimiter : str, optional
            Delimiter used in the CSV file (default: tab).

        Returns
        -------
        data : np.ndarray
            Array of shape (n_ch, n_t).
        """
        rows = []
        with open(f_path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for line in reader:
                rows.append(line)

        data = np.array(rows, dtype=float)   # n_t x n_ch [µV]
        data = data.T                        # n_ch x n_t [µV]

        return data

    @staticmethod
    def expand_trigger(trigger: np.ndarray) -> np.ndarray:
        """
        Expand a pulse trigger signal by carrying forward the last
        non-zero value.

        Example
        -------
        [0, 0, 1, 0, 0, 2, 0] -> [0, 0, 1, 1, 1, 2, 2]
        """
        trigger = np.asarray(trigger)
        out = trigger.copy()
        last = 0
        for i, val in enumerate(out):
            if val != 0:
                last = val
            else:
                out[i] = last
        return out

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        f_path: str,
        fs: float,
        signal_loc: Sequence[int],
        channel_names: Sequence[str],
        channel_types: Sequence[str],
        delimiter: str = "\t",
        trigger_labels: Optional[Dict[str, int]] = None,
    ) -> "EEG_Loader":
        """
        Create a EEG_Loader instance from a CSV file.

        Input CSV values are assumed to be in microvolts [µV].
        The returned object stores signals in volts [V].
        """
        raw_data = cls.read_csv(f_path, delimiter=delimiter)  # [µV]

        signal_loc = np.asarray(signal_loc)
        channel_names = np.asarray(channel_names)
        channel_types = np.asarray(channel_types)

        assert len(channel_names) == len(signal_loc), \
            "Length of channel_names must match length of signal_loc."
        assert len(channel_types) == len(signal_loc), \
            "Length of channel_types must match length of signal_loc."

        # Select only the channels of interest
        signals_uV = raw_data[signal_loc, :]  # (n_ch, n_t)

        # Convert µV → V for EEG/ECG; keep trigger as-is
        signals_V = signals_uV.astype(float).copy()
        for i, t in enumerate(channel_types):
            if t in ("eeg", "ecg"):
                signals_V[i, :] *= 1e-6

        rec = cls(
            data=signals_V,
            fs=fs,
            channel_names=channel_names,
            channel_types=channel_types,
            trigger_labels=trigger_labels,
        )
        return rec

    def _add_trigger_annotations(self, raw: mne.io.BaseRaw) -> None:
        """
        Create MNE Annotations for the trigger channel.

        Assumes the original CSV trigger is stored as pulse events
        (single-sample codes), and expands them into continuous blocks
        using `expand_trigger`.
        """
        if not self.trigger_labels:
            print("'trigger_labels' is not defined")
            return

        # find trigger channel index
        trig_idx = np.where(self.channel_types == "trigger")[0]
        if trig_idx.size == 0:
            print("No trigger channel for annotations is founded")
            return

        trig_idx = trig_idx[0] # select the first trigger channel

        trigger_raw = self.data[trig_idx, :]
        trigger_int = trigger_raw.astype(int) # float to int

        trigger = self.expand_trigger(trigger_int)

        onset: List[float] = []
        duration: List[float] = []
        description: List[str] = []

        times = raw.times
        for descr, code in self.trigger_labels.items():
            indices = np.where(trigger == code)[0]
            if indices.size == 0:
                continue

            for g in group_consecutive(indices):
                start_idx, end_idx = int(g[0]), int(g[-1])
                start_t, end_t = times[start_idx], times[end_idx]

                onset.append(start_t)
                duration.append(end_t - start_t)
                description.append(descr)

        if len(onset) == 0:
            return

        my_annot = mne.Annotations(
            onset=onset,
            duration=duration,
            description=description,
        )
        raw.set_annotations(my_annot)

    def to_mne_raw(
        self,
        apply_montage: bool = True,
        add_annotations: bool = False,
        ) -> mne.io.RawArray:
        """
        Convert this recording into an mne.io.RawArray object.

        Parameters
        ----------
        apply_montage : bool
            If True, apply the standard_1020 montage based on channel names.
        add_annotations : bool
            If True, use the trigger channel and self.trigger_labels to
            construct MNE Annotations and attach them to the Raw object.
        Notes
        -----
        - self.data is already in Volts.
        - No additional scaling is applied here.

        Returns
        -------
        raw : mne.io.RawArray
            MNE RawArray containing the signals (in Volts).
        """
        fs = self.fs
        ch_names = list(self.channel_names)
        ch_types_custom = list(self.channel_types)
        data = self.data.copy()  # already V

        # Map custom channel types to MNE-compatible types
        mne_ch_types: List[str] = []
        for t in ch_types_custom:
            if t == "trigger":
                mne_ch_types.append("stim")
            else:
                mne_ch_types.append(t)

        # Trigger channel cast to int (MNE expects integer stim channel)
        for i, t in enumerate(ch_types_custom):
            if t == "trigger":
                data[i, :] = data[i, :].astype(int)

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=mne_ch_types,
        )

        raw = mne.io.RawArray(data, info)

        # Montage (sensor positions)
        if apply_montage:
            montage = mne.channels.make_standard_montage(montage_set)
            raw.set_montage(montage)

        # Trigger → Annotations
        if add_annotations:
            self._add_trigger_annotations(raw)

        return raw


def csv_to_raw(f_path: str,
               save_path: str | None = None):

    f_path = Path(f_path).as_posix()

    temp = EEG_Loader.read_csv(f_path)
    if temp.shape[0] != 32:
        print(f"[Warn] pass {f_path} as the shape of data is different from the preassigned shape")

    loa = EEG_Loader.from_csv(
        f_path=f_path,
        fs=fs,
        signal_loc=signal_loc,
        channel_names=channel_names,
        channel_types=channel_types,
        trigger_labels=trigger_labels
        )
    raw = loa.to_mne_raw(
        apply_montage=True,
        add_annotations=True
        )
    if save_path:
        raw.save(save_path, overwrite=True)
    return raw
