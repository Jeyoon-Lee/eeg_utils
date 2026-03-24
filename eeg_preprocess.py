# -*- coding: utf-8 -*-
"""
eeg_utils/eeg_preprocess.py

Contains:
    class:
        EEG_Preprocessor        - Preprocessing pipeline:
                                  bandpass_filter -> re_referencing ->
                                  mark_bad_channels -> interpolate_bad_channels
                                  -> prepare_ica_raw -> fit_ica -> apply_ica
        BadSegmentAnnotator     - Add 'BAD_*' annotations to Raw objects stored
                                  inside an EEG_Preprocessor.
        EEG_Epocher             - Create fixed-length epochs from a Raw object and
                                  manage epoch-level metadata.
        ECG_Processor           - Detect R-peaks and compute HR/HRV metrics.
    function:
        bad_annot_mask          - Return a boolean mask of BAD-annotated samples.
        preprocess_raw          - End-to-end preprocessing pipeline.
        align_set_eeg_to_orig   - Reorder .set-derived EEG channels to match
                                  the original Raw channel order.
        plot_info               - Plot a summary figure for a preprocessed recording.
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from config import EEG_CFG
from eeg_utils.eeg_io import EEG_Loader
from eeg_utils.sleep_staging import SleepDetector, upsample_hypnogram
from eeg_utils.viz import plot_hypnogram, plot_spectrogram
from utils.utils import group_consecutive, windowing
from mne.preprocessing import ICA, create_ecg_epochs, find_ecg_events
from typing import List, Dict, Optional, Sequence, Any, Literal
from matplotlib.axes import Axes

# ====== CONFIG ======
trigger_labels = EEG_CFG.trigger_labels
trigger_colors = EEG_CFG.trigger_colors
channel_names = EEG_CFG.channel_names
channel_types = EEG_CFG.channel_types
montage_set = EEG_CFG.montage

signal_loc = EEG_CFG.signal_loc
fs = EEG_CFG.fs
staging_ch = EEG_CFG.sleep_staging.target_ch
stage_map = EEG_CFG.sleep_staging.stage_mapping
orig_ref = EEG_CFG.reference


# ============================================================
# Helper function
# ============================================================

def bad_annot_mask(raw: mne.io.BaseRaw):
    """
    True: BAD segments
    False: remained(GOOD) segments
    """
    raw2 = raw.copy()
    times = raw2.times

    ann = raw2.annotations
    is_bad = np.array(["BAD" in d for d in ann.description])
    bad_ann = ann[is_bad]

    total_mask = np.zeros(times.shape, dtype=bool)  # False array
    for onset, dur in zip(bad_ann.onset, bad_ann.duration):
        mask = (times >= onset) & (times < onset + dur)
        total_mask |= mask

    return total_mask


# ============================================================
# Type alias
# ============================================================

SourceName = Literal["orig", "filt", "ref", "marked", "interp", "ica_train", "clean"]


# ============================================================
# Classes
# ============================================================

class EEG_Preprocessor:
    """
    Preprocessing pipeline:
        bandpass_filter -> re_referencing -> mark_bad_channels -> interpolate_bad_channels
        -> prepare_ica_raw -> fit_ica -> apply_ica
    """
    def __init__(
            self,
            raw: mne.io.BaseRaw
            ) -> None:

        self.raw_orig: mne.io.BaseRaw = raw.copy()

        self.raw_filt: Optional[mne.io.BaseRaw] = None
        self.raw_ref: Optional[mne.io.BaseRaw] = None
        self.raw_marked: Optional[mne.io.BaseRaw] = None
        self.raw_interp: Optional[mne.io.BaseRaw] = None

        self.raw_ica_train: Optional[mne.io.BaseRaw] = None
        self.ica: Optional[ICA] = None
        self.raw_clean: Optional[mne.io.BaseRaw] = None

    def _get_raw(self, source: SourceName) -> mne.io.BaseRaw:
        source_map = {
            "orig": self.raw_orig,
            "filt": self.raw_filt,
            "ref": self.raw_ref,
            "marked": self.raw_marked,
            "interp": self.raw_interp,
            "ica_train": self.raw_ica_train,
            "clean": self.raw_clean
            }
        raw = source_map.get(source, None)
        if raw is None:
            raise ValueError(
                f"Source '{source}' is empty.",
                "Make sure you ran the prior step that creates it."
                )
        return raw

    def _set_raw(self, target: SourceName, raw: mne.io.BaseRaw) -> None:
        if target == "orig":
            raise ValueError("target='orig' is not allowed (raw_orig is immutable).")
        if target == "filt":
            self.raw_filt = raw
        elif target == "ref":
            self.raw_ref = raw
        elif target == "marked":
            self.raw_marked = raw
        elif target == "interp":
            self.raw_interp = raw
        elif target == "ica_train":
            self.raw_ica_train = raw
        elif target == "clean":
            self.raw_clean = raw
        else:
            raise ValueError(f"Unsupported target: {target}")

    def bandpass_filter(
            self,
            l_freq: float = 0.5,
            h_freq: float = 50.0,
            notch_freq: Optional[float] = 60.0,
            picks: str | list | None = 'eeg',
            source: SourceName = 'orig',
            target: SourceName = 'filt',
            load_data: bool = True
            ):
        """
        Apply band-pass (+ optional notch) filtering.

        Parameters
        ----------
        l_freq, h_freq : float
            Band-pass filter frequencies.
            The default is 0.5 and 50.0.

        notch_freq : float or None
            Apply a notch filter at this frequency. None = skip.
            The default is 60.
        picks : str | list | None
            Channels to filter.
            The default is 'eeg'.
        inplace : bool
            True → store in self.raw_filt; False → return filtered Raw.
            The default is True.

        Returns
        -------
        self or raw_filt

        """
        base = self._get_raw(source)
        raw_filt = base.copy()
        if load_data:
            raw_filt.load_data()

        raw_filt.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            fir_design='firwin',
            fir_window='hamming',
            phase='zero',
            verbose='CRITICAL'
            )

        if notch_freq is not None:
            raw_filt.notch_filter(freqs=[notch_freq], picks=picks, verbose='CRITICAL')

        self._set_raw(target, raw_filt)
        return self

    def re_referencing(
            self,
            orig_ref: Optional[str] = None,
            drop_orig_ref: bool = False,
            reref: str | list | tuple = 'average',
            source: SourceName = 'filt',
            target: SourceName = 'ref',
            projection: bool = False
            ) -> 'EEG_Preprocessor':
        """
        Re-referencing options:
            - 'average' (default): apply common average reference to EEG.
            - None: do not change the reference.
            - list|tuple of channel names: use these channels as reference.
        """
        base = self._get_raw(source)
        raw_ref = base.copy()

        if orig_ref and drop_orig_ref:
            raw_ref = raw_ref.drop_channels([orig_ref])
        elif orig_ref and (not drop_orig_ref):
            raw_ref.info['bads'] = [orig_ref]

        if reref == 'average':
            raw_ref.set_eeg_reference('average', projection=projection)
        elif reref is None:
            pass  # no re-referencing
        elif isinstance(reref, (list, tuple)):
            raw_ref.set_eeg_reference(ref_channels=list(reref), projection=projection)
        else:
            raise ValueError(f"Unsupported reref option: {reref}")

        self._set_raw(target, raw_ref)
        return self

    def mark_bad_channels(
            self,
            bads: Sequence[str],
            source: SourceName = 'ref',
            target: SourceName = 'marked',
            mode: Literal['replace', 'extend'] = 'replace',
            ensure_exists: bool = True,
            ) -> 'EEG_Preprocessor':
        """
        Mark bad channels by name.

        Parameters
        ----------
        bads : Sequence[str]
            Channel names to mark as bad.
        mode : Literal['replace', 'extend'], optional
            - 'extend': add to existing bad channel list.
            - 'replace' : overwrite existing bad channel list.
            The default is 'replace'.
        ensure_exists : bool, optional
            If True, will raise error if any channel name does not exist.
            The default is True.
        """
        base = self._get_raw(source)
        raw_marked = base.copy()
        bads = list(bads)
        if ensure_exists:
            missing = [ch for ch in bads if ch not in raw_marked.ch_names]
            if missing:
                raise ValueError(f"Bad channels not found in Raw: {missing}")

        if mode == 'replace':
            raw_marked.info['bads'] = bads
        elif mode == 'extend':
            # preserve order, no duplicates
            current = list(raw_marked.info.get("bads", []))
            for ch in bads:
                if ch not in current:
                    current.append(ch)
            raw_marked.info["bads"] = current
        else:
            raise ValueError("mode must be 'replace' or 'extend'.")

        self._set_raw(target, raw_marked)
        return self

    def interpolate_bad_channels(
            self,
            source: SourceName = 'marked',
            target: SourceName = 'interp',
            reset_bads: bool = True,
            method: Literal['spline', 'nearest'] = 'spline',
            ) -> 'EEG_Preprocessor':
        base = self._get_raw(source)
        raw_interp = base.copy()

        if len(raw_interp.info.get('bads', [])) == 0:
            print("[WARN] No bad channels to interpolate (info['bads'] is empty).")
            self._set_raw(target, raw_interp)
            return self

        raw_interp.interpolate_bads(reset_bads=reset_bads, method=method)
        self._set_raw(target, raw_interp)
        return self

    def prepare_ica_raw(
            self,
            l_freq: float = 1.0,
            h_freq: float = 50,
            notch_freq: Optional[float] = 60.0,
            picks: str | list | None = 'eeg',
            source: SourceName = 'interp',
            target: SourceName = 'ica_train',
            load_data: bool = True
            ) -> "EEG_Preprocessor":
        """
        Create a copy of Raw specifically filtered for ICA training.
        """
        base = self._get_raw(source)
        raw_ica = base.copy()
        if load_data:
            raw_ica.load_data()

        raw_ica.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            fir_design='firwin',
            fir_window='hamming',
            phase='zero',
            verbose='CRITICAL'
            )
        if notch_freq is not None:
            raw_ica.notch_filter(freqs=[notch_freq], picks=picks, verbose='CRITICAL')

        self._set_raw(target, raw_ica)
        return self

    def fit_ica(
            self,
            n_components: int | float | None = 0.999999,
            random_state: int = 97,
            method: str = 'fastica',
            picks: str | list | None = 'eeg',
            use_ecg: bool = True,
            use_eog: bool = True,
            ecg_ch_name: str = 'ecg',
            reject_by_annot: bool = True
            ) -> 'EEG_Preprocessor':
        """
        Fit ICA on ICA-preprocessed data and optionally remove ECG-related components.

        Parameters
        ----------
        use_ecg : bool
            Whether to detect/remove ECG-related ICA components.
        """
        if self.raw_ica_train is None:
            raise ValueError("raw_ica_train is empty. Call prepare_ica_raw() first.")

        ica = ICA(
            n_components=n_components,
            random_state=random_state,
            method=method
            )
        ica.fit(self.raw_ica_train, picks=picks,
                reject_by_annotation=reject_by_annot)

        # ECG-based artifact removal
        if use_ecg and (ecg_ch_name in self.raw_ica_train.ch_names):
            try:
                ecg_epochs = create_ecg_epochs(self.raw_ica_train, ch_name=ecg_ch_name)
                ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
                ica.exclude.extend(ecg_inds)
                print(f"[INFO] ECG-related ICA components excluded: {ecg_inds}")
            except Exception as e:
                print(f"[WARN] ECG-based ICA detection failed: {e}")

        if use_eog:
            eog_proxy_candidates = ["Fp1", "Fp2"]
            eog_proxy = [ch for ch in eog_proxy_candidates if ch in self.raw_ica_train.ch_names]

            if len(eog_proxy) > 0:
                try:
                    eog_inds, _ = ica.find_bads_eog(self.raw_ica_train, ch_name=eog_proxy,
                                                             measure='correlation',
                                                             threshold='auto')
                    ica.exclude.extend(eog_inds)
                    print(f"[INFO] EOG-proxy ICA components excluded using {eog_proxy}: {eog_inds}")
                except Exception as e:
                   print(f"[WARN] EOG-proxy ICA detection failed: {e}")
            else:
                print("[WARN] No frontal EEG channels available for EOG-proxy detection.")

        self.ica = ica
        return self

    def apply_ica(
            self,
            source: SourceName = 'interp',
            target: SourceName = 'clean'
            ) -> "EEG_Preprocessor":
        """ Apply ICA to a chosen Raw object. """

        if self.ica is None:
            raise ValueError("ICA is not fitted. Call fit_ica() first.")

        base = self._get_raw(source)
        clean_raw = base.copy()

        self.ica.apply(clean_raw)

        self._set_raw(target, clean_raw)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Return all intermediary and final processing outputs as a dictionary.
        """
        return {
            "raw_orig": self.raw_orig,
            "raw_filt": self.raw_filt,
            "raw_ref": self.raw_ref,
            "raw_marked": self.raw_marked,
            "raw_interp": self.raw_interp,
            "raw_ica_train": self.raw_ica_train,
            "ica": self.ica,
            "raw_clean": self.raw_clean
            }


class BadSegmentAnnotator:
    """ Add 'BAD_*' annotations to Raw objects stored inside an EEG_Preprocessor. """
    def __init__(self, preproc: EEG_Preprocessor):
        """
        Parameters
        ----------
        preproc : EEG_Preprocessor
            An instance of EEG_Preprocessor that manages Raw slots and provides
            `_get_raw(source)` and `_set_raw(target, raw)`.
        """
        self.p = preproc

    @staticmethod
    def _compute_multitaper_bp(raw: mne.io.BaseRaw,
                               freqs: np.ndarray = np.arange(0.5, 4, 0.5),
                               decim: int = 100,
                               freq_band: dict | None = None,
                               picks: str | list | None = 'eeg'):
        """
        Compute a continuous multitaper time-frequency representation (TFR) and
        derive band-power time series.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input Raw object.
        freqs : np.ndarray, optional
            Frequencies used for TFR computation. The default is np.arange(0.5, 4, 0.5).
        decim : int, optional
            Decimation factor applied to the time axis of the TFR (downsampling).
            The default is 100.
        freq_band : dict|None, optional
            Dictionary {band_name: (fmin, fmax)}.
            If None, uses EEG_CFG.freqs_band.
            The default is None.
        picks : str|list|None
            Chnnaels to include. The default is 'eeg'.

        Returns
        -------
        band_power : dict[str, np.ndarray]
            Dictionary of band-power time series.
        times : np.ndarray
            Time vector for the band-power series (seconds).
        """
        n_cycles = freqs / 2
        tfr = raw.compute_tfr(
            method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,
            picks=picks,
            decim=decim
            )

        if freq_band is None:
            from config import EEG_CFG
            freq_band = EEG_CFG.freqs_band

        data = tfr.get_data()  # (n_ch, n_f, n_t)
        band_power = {}
        for band, (fmin, fmax) in freq_band.items():
            band_mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(band_mask):
                continue
            band_power[band] = data[:, band_mask, :].mean(axis=1).mean(axis=0)  # (n_t,)
        times = tfr.times
        return band_power, times

    def annot_bad_power(
            self,
            source: SourceName = 'marked',
            target: SourceName = 'marked',
            band_name: str = 'Delta',
            q: Optional[float] = None,  # 95.0,
            thr: Optional[float] = 1.001e-08,
            min_bad_sec: float = 0.1,
            freqs: np.ndarray = np.arange(0.5, 4.0, 0.5),
            decim: int = 100,
            picks: str | list | None = "eeg",
            ) -> 'BadSegmentAnnotator':
        """
        Annotate BAD segments based on band-power thresholding.

        This method computes a continuous multitaper TFR, derives a band-power time series,
        and marks time regions where the band power exceeds a threshold.

        Parameters
        ----------
        source, target : SourceName, optional
            Input/output Raw slot names in EEG_Preprocessor.
            The default is 'marked'.
        band_name : str, optional
            The band name to threshold (must exist in the computed `band_power` dict).
            The default is 'Delta'.
        q : Optional[float], optional
            Percentile (0-100). The default is None.
        thr : Optional[float], optional
            Fixed threshold on band-power. The default is 1.001e-08.
        min_bad_sec : float, optional
            Minimum duration (seconds) of a BAD annotation to keep.
            The default is 0.1.
        freqs : np.ndarray
            Frequencies used for TFR computation.
        decim : int
            Decimation factor for the TFR time axis.
        picks : str | list | None
            Channels to include. The default is 'eeg'.
        """
        base = self.p._get_raw(source)
        raw = base.copy()
        band_power, times = self._compute_multitaper_bp(raw,
                                                        freqs=freqs,
                                                        decim=decim,
                                                        picks=picks)

        if band_name not in band_power:
            raise ValueError(
                f"{band_name} not in band_power keys: {list(band_power.keys())}"
                )

        vals = band_power[band_name]
        if thr is not None:
            print(f"[INFO] Using fixed threshold: {thr:.3e}")
        elif q is not None:
            thr = np.percentile(vals, q)
            print(f"[INFO] Using {q}th percentile as threshold: {thr:.3e}")
        else:
            raise ValueError("Both thr and q cannot be None.")

        mask = vals > thr
        diff = np.diff(mask.astype(int))
        starts = []
        ends = []
        if mask.size > 0 and mask[0]:
            starts.append(0)
        starts.extend(list(np.where(diff == 1)[0] + 1))
        ends.extend(list(np.where(diff == -1)[0]))
        if mask.size > 0 and mask[-1]:
            ends.append(len(mask) - 1)

        # dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

        onsets = []
        durations = []
        for s, e in zip(starts, ends):
            onset = float(times[s])
            duration = float((times[e] - times[s]))  # + dt)  # TODO: understandable! include last bin width
            if duration >= min_bad_sec:
                onsets.append(onset)
                durations.append(duration)

        if len(onsets) > 0:
            annot = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=[f"BAD_{band_name}"] * len(onsets),
            )
            raw.set_annotations(raw.annotations + annot)
        else:
            print(f"No BAD_{band_name} segments are found.")

        self.p._set_raw(target, raw)
        return self

    def annot_bad_amp(self,
                      source: SourceName = 'marked',
                      target: SourceName = 'marked',
                      reject_amp: float = 280e-6,  # 280 µV
                      window_len: float = 2.0,
                      overlap: float = 0.5,
                      picks: str | list | None = 'eeg',
                      mode: Literal['any', 'nch', 'percent'] = 'any',
                      nch: int = 1,  # mode = 'nch'
                      percent: float = 30.0,  # mode = 'percent'
                      min_duration: float = 0.1,
                      description: str = "BAD_AMP"
                      ):

        base = self.p._get_raw(source)
        raw = base.copy()

        sfreq = float(raw.info['sfreq'])
        data = raw.get_data(picks=picks)
        n_ch, n_times = data.shape
        # windows: (n_win, n_ch, n_t)
        windows, win_start_samp, win_start_sec = windowing(
            array=data,
            sfreq=sfreq,
            window_len=window_len,
            overlap=overlap,
            axis=-1
            )
        ptp = windows.max(axis=-1) - windows.min(axis=-1)  # (n_win, n_ch)
        exceed = ptp > float(reject_amp)

        if mode == 'any':
            bad_win = np.any(exceed, axis=1)
        elif mode == 'nch':
            bad_win = (np.sum(exceed, axis=1) >= int(nch))
        elif mode == 'percent':
            bad_win = (100.0 * np.sum(exceed, axis=1) / float(n_ch) >= float(percent))
        else:
            raise ValueError("mode must be one of  {'any', 'nch', 'percent'}")

        if bad_win.size == 0 or not np.any(bad_win):
            print(f"No {description} segments are found.")
            self.p._set_raw(target, raw)
            return self

        # Convert bad_win mask to continuous time segments
        diff = np.diff(bad_win.astype(int))
        starts = []
        ends = []
        if bad_win[0]:
            starts.append(0)
        starts.extend(list(np.where(diff == 1)[0] + 1))
        ends.extend(list(np.where(diff == -1)[0]))
        if bad_win[-1]:
            ends.append(len(bad_win) - 1)

        # The window spans [start, start+window_len]
        win_size_samp = int(round(window_len * sfreq))
        onsets = []
        durations = []
        for ws, we in zip(starts, ends):
            s_samp = int(win_start_samp[ws])
            e_samp = int(win_start_samp[we] + win_size_samp)  # include end window
            onset = float(s_samp / sfreq)
            dur = float((e_samp - s_samp) / sfreq)

            if dur >= float(min_duration):
                onsets.append(onset)
                durations.append(dur)

        ann = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=[description] * len(onsets),
        )
        raw.set_annotations(raw.annotations + ann)
        self.p._set_raw(target, raw)
        return self


class EEG_Epocher:
    def __init__(self,
                 raw: mne.io.BaseRaw):

        self.raw = raw

        # output
        self.epochs = None
        self.epoch_len = None
        self.overlap = None

    def make_epochs(self,
                    epoch_len: float = 5.0,
                    overlap: float = 2.5,
                    reject_by_annot: bool = True,
                    add_metadata: bool = True,
                    verbose: bool = True):
        """
        Create fixed-length epochs from Raw using MNE.

        Parameters
        ----------
        epoch_len : float, optional
            Epoch length in seconds. The default is 5.0.
        overlap : float, optional
            Overlap in seconds. The default is 2.5.
        reject_by_annot : bool, optional
            If True, MNE will drop epochs overlapping with annotations whose
            description begins with "BAD".
            The default is True.
        add_metadata : bool, optional
            If True, add condition metadata based on Raw annotations.
            The default is True.

        Returns
        -------
        self

        """
        self.epoch_len = float(epoch_len)
        self.overlap = float(overlap)

        events = mne.make_fixed_length_events(
            self.raw,
            id=1,
            duration=self.epoch_len,
            overlap=self.overlap)

        epochs = mne.Epochs(
            self.raw,
            events,
            tmin=0.0,
            tmax=self.epoch_len,
            baseline=None,
            reject_by_annotation=reject_by_annot,
            preload=True
            )
        self.epochs = epochs
        if verbose:
            print(f"Epochs kept: {len(epochs)}")

        if add_metadata:
            self._add_metadata(verbose=verbose)

        return self

    def _add_metadata(self, verbose: bool):
        """
        Add epoch-level metadata for experimental condition.

        This function maps each epoch (by its center time) to a condition label
        based on Raw annotations. It assumes that condition annotations are
        non-overlapping and cover the timeline properly.
        """
        sfreq = float(self.epochs.info['sfreq'])
        ann = self.raw.annotations
        cond_names = [desc for desc in np.unique(ann.description)
                      if (not desc.startswith("BAD_")) and (desc in trigger_labels)]

        epoch_strt_samp = self.epochs.events[:, 0]
        epoch_strt_t = epoch_strt_samp / sfreq
        epoch_center_t = epoch_strt_t + (self.epoch_len / 2)
        meta = pd.DataFrame({
            'epoch_start_sec': epoch_strt_t,
            'epoch_center_sec': epoch_center_t,
            })

        # If no condition annotations exist, store timing only.
        if len(cond_names) == 0:
            print("[WARN] No non-BAD annotations exist for stage labeling.")
            self.epochs.metadata = meta
            return self

        epoch_cond = np.full(len(epoch_center_t), None, dtype=object)
        for on, dur, desc in zip(ann.onset, ann.duration, ann.description):
            if desc in cond_names:
                s = float(on)
                e = float(on + dur)
                m = (epoch_center_t >= s) & (epoch_center_t < e)
                epoch_cond[m] = desc

        meta['stage'] = epoch_cond
        self.epochs.metadata = meta
        if verbose:
            print("[INFO] Added metadata: 'stage'")
        return self

    def add_extra_metadata(self,
                           raw_ch_name: str,
                           meta_col_name: str,
                           stage_map: dict | None = None
                           ):
        """
        Add a new metadata column to ``self.epochs.metadata``

        Parameters
        ----------
        raw_ch_name : str
            Name of the channel in ``self.raw``
        stage_map : dict|None, optional
            stage_map to convert the elements from int to str.
            If None, ``EEG_CFG.sleep_staging.stage_mapping`` is used.
            The default is None.
        meta_col_name : str
            column name to add to ``self.epochs.metadata``
            must not already exist.

        Returns
        -------
        self

        """
        if self.epochs is None or self.epochs.metadata is None:
            raise ValueError("Create epochs (and metadata) first.")

        meta = self.epochs.metadata.copy()
        if meta_col_name in meta.columns:
            raise ValueError(f"{meta_col_name} already exists in epochs.metadata.",
                             "Change 'meta_col_name'.")

        labels_int = np.squeeze(self.raw.get_data(raw_ch_name))
        if stage_map is None:
            stage_map = EEG_CFG.sleep_staging.stage_mapping
        inv_stage_map = {v: k for k, v in stage_map.items()}
        labels_str = np.array([inv_stage_map.get(i, 'UNK') for i in labels_int], dtype=object)

        epoch_center_t = meta["epoch_center_sec"].to_numpy()
        sfreq = float(self.raw.info["sfreq"])
        sample_idx = np.rint(epoch_center_t * sfreq).astype(int)
        sample_idx = np.clip(sample_idx, 0, len(self.raw.times) - 1)

        meta[meta_col_name] = labels_str[sample_idx]
        self.epochs.metadata = meta
        print(f"[INFO] Added new column '{meta_col_name}'")

        return self

    @staticmethod
    def plot_drop_log(epochs,
                      ax: Optional[Axes] = None):
        """ visualize how many epochs were dropped"""

        if epochs is None:
            raise ValueError("Epochs not created yet.")

        drop_log = []
        for i in epochs.drop_log:
            if len(i) == 0:
                drop_log.append("save")
            else:
                drop_log.append(i[0])
        drop_log = pd.Series(drop_log)

        total_len = len(drop_log)
        save_len = (drop_log == 'save').sum()
        remove_len = total_len - save_len

        title = f"{remove_len} of {total_len} epochs removed ({(remove_len/total_len)*100: .1f}%)"
        y_label = "% of epochs removed"

        vc = drop_log.value_counts(dropna=False, normalize=True).mul(100)
        vc.drop(index="save", errors='ignore').plot(kind="bar",
                                                  ax=ax,
                                                  title=title,
                                                  ylabel=y_label,
                                                  rot=0,
                                                  fontsize=10,
                                                  color='lightgray'
                                                  )

        return ax

    @staticmethod
    def plot_epoch_distribution(epochs,
                                ax: Optional[Axes] = None,
                                value: str = 'epoch_start_sec',
                                index: str = 'stage',
                                index_order: Optional[list | tuple] = None,
                                column: str = 'sleep'):
        """
        Plot a stacked bar chart to visualize the distribution of remaining epochs.

        Notes
        -----
        Do not modify the default argument values.
        """
        if epochs.metadata is None:
            raise ValueError("epochs.metadata is None.")

        meta = epochs.metadata.copy()

        required_cols = {value, index, column}
        missing = required_cols - set(meta.columns)
        if missing:
            raise ValueError(
                "Missing required column(s) in epochs.metadata: "
                f"{sorted(missing)}. "
                f"Available columns: {list(meta.columns)}"
            )

        if index == 'stage' and index_order is None:
            stages_present = set(meta[index].dropna().unique())
            index_order = [k for k in trigger_labels.keys()
                           if k in stages_present]
        if index_order:
            meta[index] = pd.Categorical(
                meta[index],
                categories=index_order,
                ordered=True
                )
        meta_pv = meta.pivot_table(values=value,
                                   index=index,
                                   columns=column,
                                   observed=True,
                                   aggfunc='count').fillna(0).astype(int)

        ax = meta_pv.plot(kind='bar', stacked=True,
                          rot=0, ylabel="Epoch Count (n)", ax=ax)
        ax.set_title("Epoch label distribution")
        for c in ax.containers:
            labels = ["" if int(v) == 0 else int(v)
                      for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type="center")
        return ax

    @staticmethod
    def save_epochs(epochs: mne.epochs.BaseEpochs,
                    out_path: str):
        epochs.save(out_path, overwrite=True)
        print(f"Saved: {out_path}")


class ECG_Processor:
    """
    ECG_Processor contains:
        - detect_r_peaks()
            R-peak detection using mne.preprocessing.find_ecg_evnets
        - plot_r_peaks()
        - compute_hr_hrv()
            compute RR intervals, HR (bpm), and RMSSD
    """

    def __init__(self,
                 raw: mne.io.BaseRaw,
                 ecg_ch_name: str = 'ecg'):
        """
        Parameters
        ----------
        raw : mne.io.BaseRaw
            It should contain an ecg channel.
        ecg_ch_name : str, optional
            The default is 'ecg'.
        """
        self.raw = raw
        self.sfreq = raw.info['sfreq']
        self.ecg_ch_name = ecg_ch_name

        # output
        self.ecg_events = None
        self.r_peaks = None
        self.rr_intervals = None
        self.hr_bpm = None
        self.rmssd = None
        self.sdnn = None

    def detect_r_peaks(self,
                       reject_by_annotation: bool = False,
                       event_id: int = 999):
        print("Detecting R-peaks ...")
        ecg_events, _, _ = find_ecg_events(
            self.raw,
            ch_name=self.ecg_ch_name,
            event_id=event_id,
            reject_by_annotation=reject_by_annotation
            )

        self.ecg_events = ecg_events
        self.r_peaks = ecg_events[:, 0]  # shape: (n_r_peaks, 3); [sample_index, 0, event_id=999]
        self.event_id = event_id
        print(f"Detected {len(self.r_peaks)} R-peaks.")

        return self

    def plot_r_peaks(self,
                     start: float = 0.0,
                     duration: float = 10.0,
                     show_ecg_only: bool = False):

        if self.ecg_events is None:
            self.detect_r_peaks()
            print("Running detect_r_peaks() automatically...")
            print(f"Detected {len(self.r_peaks)} R-peaks.")

        picks = None
        if show_ecg_only:
            picks = mne.pick_channels(self.raw.ch_names, include=[self.ecg_ch_name])
        fig = self.raw.plot(
            events=self.ecg_events,
            event_color={self.event_id: 'r'},
            picks=picks
            )
        plt.show()
        return fig

    def compute_hr_hrv(self):
        if self.r_peaks is None:
            self.detect_r_peaks()
            print("Running detect_r_peaks() automatically...")
            print(f"Detected {len(self.r_peaks)} R-peaks.")

        sfreq = self.sfreq
        r = self.r_peaks.astype(float)
        r_sec = r / sfreq

        # RR interval (sec)
        rr_sec = np.diff(r) / sfreq

        # Heart Rate (bpm)
        hr_bpm = 60.0 / rr_sec
        self.hr_bpm = hr_bpm
        # HR timepoints: middle point between 2 R-peaks
        hr_times = (r_sec[1:] + r_sec[:-1]) / 2.0
        self.hr_times = hr_times

        # RMSSD (RR in sec)
        if len(rr_sec) > 1:
            diff_rr = np.diff(rr_sec)
            rmssd = np.sqrt(np.mean(diff_rr**2))
        else:
            rmssd = np.nan
        self.rmssd = rmssd

        return hr_bpm, hr_times, rmssd


# ============================================================
# Pipeline functions
# ============================================================

def preprocess_raw(raw: mne.io.BaseRaw,
                   bads: list,
                   save_raw_path: str,
                   save_epo_path: str):

    ##### Preprocessing #####
    pp = (EEG_Preprocessor(raw)
          .mark_bad_channels(bads, source='orig', target='marked', mode='extend')
          .interpolate_bad_channels(reset_bads=True, source='marked', target='interp')
          )

    (BadSegmentAnnotator(pp)
        .annot_bad_power(source="interp", target="interp", band_name="Delta")
        .annot_bad_amp(source='interp', target='interp')
    )

    pp.prepare_ica_raw(source="interp", target="ica_train", l_freq=1, h_freq=50, notch_freq=60)
    pp.fit_ica(reject_by_annot=True, use_ecg=True, ecg_ch_name="ecg", use_eog=True)
    if len(pp.ica.exclude) > 0:
        pp.apply_ica(source="interp", target="clean")
        raw = pp.raw_clean
        apply_ica = True
    else:
        raw = pp.raw_interp
        apply_ica = False

    ##### Automatic Sleep Scoring #####
    sd = SleepDetector(raw)
    sd.sleep_staging(eeg_name=staging_ch).relabel_R_after_W()
    hyp = sd.hypno_fixed
    hyp_up = upsample_hypnogram(hyp, data_len=raw.get_data().shape[-1],
                                data_fs=fs, hypno_len=30.0, fill_None='before')

    ##### Add a 'sleep' channel in a raw_clean.fif file #####
    sleep_info = mne.create_info(
        ch_names=['sleep'],
        sfreq=raw.info['sfreq'],
        ch_types=['misc']
        )
    hyp_int = np.array([stage_map.get(i, 999) for i in hyp_up], dtype=np.int16)
    hyp_raw = mne.io.RawArray(np.expand_dims(hyp_int, axis=0), sleep_info)
    raw.add_channels([hyp_raw], force_update_info=True)

    ##### Epoching #####
    ep = EEG_Epocher(raw).make_epochs().add_extra_metadata(raw_ch_name='sleep',
                                                           meta_col_name='sleep')
    epochs = ep.epochs

    ##### Save raw_clean.fif & epoch.fif #####
    raw.save(save_raw_path, overwrite=True)
    if len(epochs) == 0:
        print(f"[Warn] cannot generate {save_epo_path} as epochs are empty.")
    else:
        EEG_Epocher.save_epochs(epochs, out_path=save_epo_path)

    return raw, epochs, hyp_up, apply_ica


def align_set_eeg_to_orig(
    raw_set_eeg: mne.io.BaseRaw,
    raw_orig_eeg: mne.io.BaseRaw,
    *,
    strict_names: bool = True,
    ignore_case: bool = False,
) -> mne.io.BaseRaw:
    """
    Validate and (if needed) reorder EEG channels from a .set-derived Raw
    to match the original Raw's EEG channels.

    Checks:
      1) EEG channel names match (optionally case-insensitive)
      2) n_times match
      3) sfreq match
    If names match but order differs, reorders raw_set_eeg to raw_orig_eeg order.

    Parameters
    ----------
    raw_set_eeg : mne.io.BaseRaw
        Raw object loaded from .set and already pick_types(eeg=True) applied.
    raw_orig_eeg : mne.io.BaseRaw
        Original Raw object already pick_types(eeg=True) applied.
    strict_names : bool
        If True, require set EEG names to contain ALL original EEG names.
        If False, allow partial overlap (still enforces same length & sfreq).
    ignore_case : bool
        If True, compare channel names case-insensitively and reorder accordingly.

    Returns
    -------
    raw_set_eeg_aligned : mne.io.BaseRaw
        Copy of raw_set_eeg, reordered to match raw_orig_eeg channel order.

    """
    set_names = list(raw_set_eeg.ch_names)
    orig_names = list(raw_orig_eeg.ch_names)

    # ---------- (1) Names: match + reorder ----------
    if ignore_case:
        set_map = {ch.lower(): ch for ch in set_names}    # lower -> actual
        orig_lower = [ch.lower() for ch in orig_names]

        missing = [ch for ch in orig_lower if ch not in set_map]
        if missing and strict_names:
            raise RuntimeError(
                "EEG channel names mismatch between raw_orig and .set. "
                f"Missing (case-insensitive): {missing}"
            )

        # Build the reorder list in the exact order of orig_names
        reorder_list = []
        for ch in orig_lower:
            if ch in set_map:
                reorder_list.append(set_map[ch])
            elif not strict_names:
                # if partial allowed, just skip missing ones
                continue

    else:
        if set_names != orig_names:
            common = [ch for ch in orig_names if ch in set_names]
            if strict_names and common != orig_names:
                raise RuntimeError("EEG channel names mismatch between raw_orig and .set")
            reorder_list = orig_names[:]  # reorder to orig order
        else:
            reorder_list = orig_names[:]  # already aligned

    # If strict_names=False and we skipped missing channels, we should also
    # ensure orig_eeg is reduced to the same set (optional behavior).
    if not strict_names:
        # Align both to the overlap, in orig order
        overlap = [ch for ch in orig_names if (ch.lower() in {s.lower() for s in set_names})] if ignore_case else \
                  [ch for ch in orig_names if ch in set_names]
        if len(overlap) == 0:
            raise RuntimeError("No overlapping EEG channels between raw_orig and .set.")
        # Reduce reorder_list to overlap only
        if ignore_case:
            set_map = {ch.lower(): ch for ch in set_names}
            reorder_list = [set_map[ch.lower()] for ch in overlap]
        else:
            reorder_list = overlap

    raw_set_eeg_aligned = raw_set_eeg.copy().reorder_channels(reorder_list)

    # ---------- (2) n_times ----------
    if raw_set_eeg_aligned.n_times != raw_orig_eeg.n_times:
        raise RuntimeError(
            f"n_times mismatch: set={raw_set_eeg_aligned.n_times}, orig={raw_orig_eeg.n_times}"
        )

    # ---------- (3) sfreq ----------
    if float(raw_set_eeg_aligned.info["sfreq"]) != float(raw_orig_eeg.info["sfreq"]):
        raise RuntimeError(
            f"sfreq mismatch: set={raw_set_eeg_aligned.info['sfreq']}, orig={raw_orig_eeg.info['sfreq']}"
        )

    return raw_set_eeg_aligned


def plot_info(raw: mne.io.BaseRaw,
              epochs: mne.Epochs,
              hyp_up: np.ndarray,
              apply_ica: bool,
              ID: str | float | None = None,
              p_df: pd.DataFrame | None = None
              ):

    mosaic = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'D', 'E']]
    fig, axes = plt.subplot_mosaic(mosaic, layout='constrained',
                                   figsize=(12, 8),
                                   gridspec_kw={
                                       'width_ratios': [1, 3, 1],
                                       'height_ratios': [1, 2, 2]
                                       })
    axes['A'] = plot_hypnogram(hyp_up, ax=axes['A'],
                               time_axis=True, sfreq=fs)
    tfr = raw.compute_tfr(method='multitaper', freqs=np.arange(0.5, 25, 0.5),
                          picks=staging_ch, output='power',
                          reject_by_annotation=False)
    spect = np.squeeze(tfr.get_data())
    is_bad = bad_annot_mask(raw)
    spect[:, is_bad] = np.nan
    axes['B'] = plot_spectrogram(spect=spect,
                                freqs=tfr.freqs,
                                times=tfr.times,
                                fig=fig,
                                ax=axes['B'],
                                colormap='RdBu_r',
                                normalization=True)
    axes['A'].sharex(axes['B'])
    axes['A'].tick_params(labelbottom=False)
    axes['C'] = EEG_Epocher.plot_drop_log(epochs, ax=axes['C'])
    axes['D'] = EEG_Epocher.plot_epoch_distribution(epochs, ax=axes['D'])
    axes['E'].axis('off')
    if p_df is not None and ID is not None:
        meta_txt = (
            f"Name: {p_df[p_df['ID']==ID]['Name'].item()}\n"
            f"ID: {ID}\n"
            f"Sex: {p_df[p_df['ID']==ID]['Sex'].item()}\n"
            f"Age: {p_df[p_df['ID']==ID]['Age'].item()}\n"
            f"Apply ICA: {apply_ica}"
            )
        axes['E'].text(0.05, 0.95, meta_txt,
                    transform=axes['E'].transAxes,
                    fontsize=13,
                    va='top')
    return fig, axes
