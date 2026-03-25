# -*- coding: utf-8 -*-
"""
eeg_utils/eeg_analysis.py

Contains:
    class:
        PSDAnalyzer             - Power Spectral Density feature extractor for
                                  MNE Raw and Epochs objects.
        FCAnalyzer              - Functional Connectivity analyzer for MNE
                                  Epochs objects.
    function:
        convert2dB              - Convert power values to dB.
        create_mask             - Return a boolean mask for a frequency range.
        safe_div_nan            - Safely divide two arrays (NaN where den==0).
        concat_epochs           - Construct a continuous signal from Epochs.
        apply_FDR               - Apply FDR correction to a results DataFrame.
        extract_mixedlm_result  - Extract MixedLMResults into a long-form
                                  DataFrame.
"""

import numpy as np
import mne
from mne.stats import fdr_correction
import pandas as pd
from fooof import FOOOF
from eeg_utils.config import EEG_CFG
from typing import Tuple, Literal, Optional
from mne_connectivity import (spectral_connectivity_epochs,
                              phase_slope_index,
                              envelope_correlation)

from pactools import Comodulogram

freqs_band = EEG_CFG.freqs_band


# ============================================================
# Helper functions
# ============================================================

def convert2dB(power):
    """
    Convert power to dB, setting 0 or negative values to NaN.
    """
    power = np.asarray(power, dtype=float)
    power[power <= 0] = np.nan

    return 10 * np.log10(power)


def create_mask(freqs: np.ndarray,
                minmax_range: Tuple[float, float]) -> np.ndarray:
    """ Return boolean mask for np.ndarray in [min, max]. """
    mask = (freqs >= minmax_range[0]) & (freqs <= minmax_range[1])
    return mask


def safe_div_nan(num: np.ndarray,
                 den: np.ndarray) -> np.ndarray:
    """ Safely divide two arrays, returning NaN where denominator is zero. """
    out = np.full_like(num, np.nan, dtype=float)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def concat_epochs(epochs: mne.Epochs,
                  overlap_policy: str = "mean"   # "mean" | "first" | "last"
                  ):
    """
    Construct non-overlapped continuous signal from mne.Epochs

    Parameters
    ----------
    epochs : mne.Epochs
    overlap_policy : str, optional
        같은 시간에 값이 여러 개일 경우 처리 방법
        'mean' : 평균
        'first': 먼저 들어온 epoch 값 유지
        'last': 나중 epoch로 덮어쓰기
        The default is "mean".

    Returns
    -------
    x : np.ndaarray
        (n_ch, npt)
    """
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is None")
    for col in ['epoch_start_sec', 'epoch_center_sec']:
        if col not in epochs.metadata.columns:
            raise ValueError(f"epochs.metadata has no '{col}' column")

    sfreq = float(epochs.info['sfreq'])
    data = epochs.get_data()  # (n_ep, n_ch, n_t)
    n_ep, n_ch, n_t = data.shape

    starts_sec = epochs.metadata['epoch_start_sec'].to_numpy(dtype=float)
    t0_sec = float(np.nanmin(starts_sec))  # 기준 시각
    starts_samp = np.rint((starts_sec - t0_sec) * sfreq).astype(int)

    # 전체 길이 추정
    ends_samp = starts_samp + n_t
    total_len = int(np.max(ends_samp))

    x = np.full((n_ch, total_len), np.nan, dtype=float)

    if overlap_policy == "mean":
        acc = np.zeros((n_ch, total_len), dtype=float)
        cnt = np.zeros((total_len,), dtype=float)

        for ei in range(n_ep):
            s = starts_samp[ei]
            e = s + n_t
            acc[:, s:e] += data[ei]
            cnt[s:e] += 1.0

        valid = cnt > 0
        x[:, valid] = acc[:, valid] / cnt[valid]

    elif overlap_policy in ("first", "last"):
        order = range(n_ep) if overlap_policy == "last" else range(n_ep)
        # first의 경우: 이미 채워진 곳은 건너뛰기 위해 mask 필요
        filled = np.zeros((total_len,), dtype=bool)

        for ei in order:
            s = starts_samp[ei]
            e = s + n_t
            if overlap_policy == "last":
                x[:, s:e] = data[ei]
            else:  # first
                seg_mask = ~filled[s:e]
                if np.any(seg_mask):
                    x[:, s:e][:, seg_mask] = data[ei][:, seg_mask]
                    filled[s:e][seg_mask] = True
    else:
        raise ValueError("overlap_policy must be one of: 'mean', 'first', 'last'")

    return x


def apply_FDR(df_out: pd.DataFrame,
              filter: list | str = ['sleep', 'band', 'term'],
              pval_col_name: str = 'pval',
              suffix: str = '_fdr'):

    df = df_out.copy()
    df[f"{pval_col_name}{suffix}"] = np.nan
    df[f"sig{suffix}"] = False

    for _, g in df.groupby(filter):
        gg = g.query("converged == True").dropna(subset=[pval_col_name])
        if gg.shape[0] < 2:  # channal < 2
            continue
        reject, p_fdr = fdr_correction(gg['pval'].to_numpy())
        df.loc[gg.index, f'{pval_col_name}_fdr'] = p_fdr
        df.loc[gg.index, 'sig_fdr'] = reject
    return df


def extract_mixedlm_result(  # TODO 추후에 제거
    mdf,
    meta: dict | None = None,
    *,
    add_ci: bool = True,
    ci_level: float = 0.95,
    include_fit_stats: bool = True,
) -> pd.DataFrame:
    """
    Extract MixedLMResults (statsmodels) into a long-form DataFrame.

    Parameters
    ----------
    mdf : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted model results from md.fit().
    meta : dict, optional
        Metadata to attach to every row (e.g., {"stage":..., "band":..., "ch":...}).
    add_ci : bool
        Whether to include confidence intervals if available.
    ci_level : float
        Confidence level for CI (e.g., 0.95).
    include_fit_stats : bool
        Whether to attach fit-level statistics (nobs, llf, aic, bic, converged).

    Returns
    -------
    pd.DataFrame
        Long-form table with columns:
        [meta..., term, coef, se, p, ci_low, ci_high, (optional fit stats...)]
    """
    if meta is None:
        meta = {}

    # Core series (align by index defensively)
    params = pd.Series(mdf.params, dtype=float)
    bse = pd.Series(getattr(mdf, "bse", np.nan)).reindex(params.index)
    pvals = pd.Series(getattr(mdf, "pvalues", np.nan)).reindex(params.index)

    out = pd.DataFrame({
        "term": params.index.astype(str),
        "coef": params.values,
        "se": bse.values,
        "p": pvals.values,
    })

    # Confidence intervals (if available)
    if add_ci:
        ci_low = np.full(len(out), np.nan, dtype=float)
        ci_high = np.full(len(out), np.nan, dtype=float)
        try:
            alpha = 1.0 - float(ci_level)
            ci = mdf.conf_int(alpha=alpha)
            # conf_int returns DataFrame with same index, two columns
            ci = pd.DataFrame(ci)
            ci = ci.reindex(params.index)
            ci_low = ci.iloc[:, 0].to_numpy(dtype=float)
            ci_high = ci.iloc[:, 1].to_numpy(dtype=float)
        except Exception:
            pass

        out["ci_low"] = ci_low
        out["ci_high"] = ci_high

    # Attach metadata columns (stage/slp/band/ch, etc.)
    # Put meta columns at the front in a stable order
    if meta:
        meta_df = pd.DataFrame([meta] * len(out))
        out = pd.concat([meta_df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    # Fit-level stats (repeated for each term; convenient for filtering later)
    # if include_fit_stats:
    #     def _get_float(attr):
    #         v = getattr(mdf, attr, np.nan)
    #         try:
    #             return float(v)
    #         except Exception:
    #             return np.nan

    #     out["nobs"] = int(getattr(mdf, "nobs", len(getattr(mdf, "model", [])) or len(out)))
    #     out["llf"] = _get_float("llf")
    #     out["aic"] = _get_float("aic")
    #     out["bic"] = _get_float("bic")
    #     out["converged"] = bool(getattr(mdf, "converged", True))

    if include_fit_stats:
        out["nobs"] = int(mdf.nobs)
        out["llf"] = float(getattr(mdf, "llf", np.nan))
        out["aic"] = float(getattr(mdf, "aic", np.nan))
        out["bic"] = float(getattr(mdf, "bic", np.nan))
        out["converged"] = bool(getattr(mdf, "converged", True))

    return out


# ============================================================
# Classes
# ============================================================

class PSDAnalyzer:
    """
    Power Spectral Density (PSD) feature extractor for MNE Raw and Epochs objects
    This class computes the power spectral density(PSD) immediately upon
    initialization, and provides utilities to derive:
        1) Absolute / Relative band power
        2) Aperiodic 1/f parameters using FOOOF
    """
    def __init__(self,
                 inst: mne.io.BaseRaw | mne.Epochs,
                 picks: str | list | None = 'eeg',
                 fmin: float = 0.5,
                 fmax: float = 45.0,
                 method: Literal['multitaper', 'welch'] = 'multitaper',
                 verbose: str = "WARNING"
                 ):
        self.inst = inst
        self.picks = picks
        self.fmin = fmin
        self.fmax = fmax
        self.method = method
        self.verbose = verbose
        if picks is None:
            self.ch_names = list(self.inst.info['ch_names'])
        else:
            if isinstance(picks, str):
                picks = [picks]
            arg = {i: True for i in picks}
            self.ch_names = mne.pick_info(
                self.inst.info,
                sel=mne.pick_types(self.inst.info, **arg)
                ).ch_names

        # immediately compute PSD
        self._compute_psd()

    def _compute_psd(self):
        spec = self.inst.compute_psd(
            method=self.method,
            fmin=self.fmin,
            fmax=self.fmax,
            picks=self.picks,
            verbose=self.verbose
            )
        # for mne.io.BaseRaw, (n_ch, n_f)
        # for mne.Epochs, (n_ep, n_ch, n_f)
        self.psd, self.freqs = spec.get_data(return_freqs=True)
        self.ch_names = list(spec.ch_names)

        if isinstance(self.inst, mne.epochs.BaseEpochs):
            self.psd_mean = np.nanmean(self.psd, axis=0)  # (n_ch, n_f)
        else:
            self.psd_mean = self.psd

    def compute_band_power(
            self,
            bands: dict | None = None,
            cal: Literal['abs', 'norm'] = 'abs') -> pd.DataFrame:
        """
        Compute band power

        Parameters
        ----------
        bands : dict|None, optional
            Dictionary of band definitions, e.g. {'theta': (4,8)}.
            The default is None.
        cal : Literal['abs', 'norm'], optional
            - 'abs' : absolute band power
            - 'norm' : relative band power ( band / total power )
            The default is 'norm'.

        Returns
        -------
        df : pd.DataFrame
            Band power features indexed by channel name.
        """

        if bands is None:
            bands = freqs_band

        out = {}
        for band_name, (bmin, bmax) in bands.items():
            mask = create_mask(self.freqs, (bmin, bmax))

            if cal == 'abs':  # (n_ch, n_f) -> (n_ch,)
                out[band_name] = np.trapezoid(self.psd_mean[:, mask],
                                          self.freqs[mask],
                                          axis=-1
                                          )
            elif cal == 'norm':
                if isinstance(self.inst, mne.epochs.BaseEpochs):
                    bp = np.trapezoid(self.psd[:, :, mask], self.freqs[mask], axis=-1)  # (n_ep, n_ch)
                    tp = np.trapezoid(self.psd, self.freqs, axis=-1)  # (n_ep, n_ch)
                    rel = safe_div_nan(bp, tp).mean(axis=0)  # (n_ch,)
                    out[band_name] = rel
                else:
                    bp = np.trapezoid(self.psd[:, mask], self.freqs[mask], axis=-1)  # (n_ch,)
                    tp = np.trapezoid(self.psd, self.freqs, axis=-1)
                    out[band_name] = safe_div_nan(bp, tp)

        return pd.DataFrame(out, index=self.ch_names)

    def compute_aperiodic(self,
                          f_range: Tuple[float, float] = (2.0, 40.0),
                          fooof_settings: Optional[dict] = None
                          ) -> pd.DataFrame:

        if fooof_settings is None:
            fooof_settings = dict(min_peak_height=0.05, verbose=False)

        mask = create_mask(self.freqs, f_range)
        freqs_fit = self.freqs[mask]

        n_ch = self.psd_mean.shape[0]

        offset = np.full(n_ch, np.nan, dtype=float)
        expo = np.full(n_ch, np.nan, dtype=float)
        r2 = np.full(n_ch, np.nan, dtype=float)
        err = np.full(n_ch, np.nan, dtype=float)
        for ci in range(n_ch):
            tg = self.psd_mean[ci, mask]
            if np.any(tg == 0):
                continue
            fm = FOOOF(**fooof_settings)
            fm.fit(freqs_fit, np.squeeze(tg))

            ap = fm.get_params('aperiodic_params')
            offset[ci] = float(ap[0])
            expo[ci] = float(ap[1])
            r2[ci] = float(fm.r_squared_)
            err[ci] = float(fm.error_)

        ap_df = pd.DataFrame({
            'offset': offset,
            'exponent': expo,
            'r2': r2,
            'error': err
            }, index=self.ch_names)
        return ap_df


class FCAnalyzer:
    def __init__(self,
                 fif_path: str | None = None,
                 epochs: mne.Epochs | None = None,
                 picks: str = 'eeg',
                 ):

        if fif_path is None and epochs is None:
            raise ValueError("Either 'fif_path' or 'epochs' must be provided")
        if fif_path is not None and epochs is None:
            epochs = mne.read_epochs(fif_path, preload=True, verbose="ERROR")

        epochs_pick = epochs.copy().pick(picks)

        self.epochs = epochs
        self.epochs_pick = epochs_pick
        self.ch_names = self.epochs_pick.info['ch_names']
        self.sfreq = float(self.epochs_pick.info['sfreq'])

    @staticmethod
    def metadata_query(epochs: mne.Epochs,
                       query: str):

        if epochs.metadata is None:
            raise ValueError("epochs.metadata is None")
        epochs_query = epochs[query]
        return epochs_query

    def _add_attr(self, name, value):
        setattr(self, name, value)

    def compute_spectral_FC(self,
                            method: str,
                            mode: str,
                            fmin: float | tuple,
                            fmax: float | tuple,
                            faverage: bool,
                            mt_adaptive: bool,
                            query: str | None = None,
                            n_epochs_min=10,
                            verbose="WARNING",
                            **kwargs):
        epo = self.metadata_query(self.epochs_pick, query) if query is not None else self.epochs_pick
        if len(epo) < n_epochs_min:
            raise ValueError(f"Not enough epochs after query (n={len(epo)} < {n_epochs_min})")

        conn = spectral_connectivity_epochs(epo,
                                            method=method,
                                            mode=mode,
                                            sfreq=self.sfreq,
                                            fmin=fmin,
                                            fmax=fmax,
                                            faverage=faverage,
                                            mt_adaptive=mt_adaptive,
                                            verbose=verbose,
                                            **kwargs
                                            )

        self._add_attr(name=method, value=conn)
        return conn

    def compute_PSI(self,
                    fmin: float | tuple,
                    fmax: float | tuple,
                    mode: str = 'multitaper',
                    mt_adaptive: bool = True,
                    query: str | None = None,
                    n_epochs_min=10,
                    verbose="WARNING",
                    **kwargs
                    ):
        epo = FCAnalyzer.metadata_query(self.epochs_pick, query) if query is not None else self.epochs_pick
        if len(epo) < n_epochs_min:
            raise ValueError(f"Not enough epochs after query (n={len(epo)} < {n_epochs_min})")

        conn = phase_slope_index(epo,
                                 sfreq=self.sfreq,
                                 fmin=fmin, fmax=fmax,
                                 mode=mode,
                                 mt_adaptive=mt_adaptive,
                                 verbose=verbose,
                                 **kwargs)
        self.psi = conn
        return conn

    def compute_AEC(self,
                    query: str | None = None,
                    orthogonalize: str | bool = 'pairwise',
                    n_epochs_min=10,
                    **kwargs):
        epo = FCAnalyzer.metadata_query(self.epochs_pick, query) if query is not None else self.epochs_pick
        if len(epo) < n_epochs_min:
            raise ValueError(f"Not enough epochs after query (n={len(epo)} < {n_epochs_min})")

        conn = envelope_correlation(epo, orthogonalize=orthogonalize, **kwargs)

        self.aec = conn
        return conn

    def get_matrix(self,
                   method: str):
        conn = getattr(self, method.lower())
        if hasattr(conn, 'combine'):
            try:
                conn = conn.combine()
            except Exception:
                pass

        data = np.asarray(conn.get_data(output='dense'), dtype=np.float32)
        M = data[:, :, 0]
        np.fill_diagonal(M, np.nan)
        return M

    def conn_to_df(self,
                   method: str,
                   meta: dict | None = None,
                   band: str | list | None = None
                   ):
        """
        Convert a stored connectivity result into an edge-list (ch to ch) DataFrame.

        Parameters
        ----------
        method : str
            Name of the connectivity result stored as an attribute of this instance.
            Examples: 'wpli2_debiased', 'psi', 'aec'.

        meta : dict
            Metadata to prepend as columns. Typical keys:
                {'ID': ..., 'Order': ..., 'stage': ..., 'sleep': ...}.

        band : str|list|None, optional
            Custom labels for the last dimension (frequency-like axis).
            - If provided, its length must match `n_freqs` in the connectivity data.
            - This is required when the connectivity object does not expose `conn.freqs`


        Returns
        -------
        df : pandas.DataFrame
            meta columns + ['ch1', 'ch2', 'freq', 'value']

        """
        if not hasattr(self, method.lower()):
            raise AttributeError(f"'{method.lower()}' not found")

        conn = getattr(self, method.lower())
        if method.lower() in ('psi', 'aec') and band is None:
            raise ValueError(f"'{method}' needs 'band' argument")

        n_ch = len(self.ch_names)

        # If the object supports combine() (epoch/time-resolved), reduce to a single connectivity
        if hasattr(conn, 'combine'):
            try:
                conn = conn.combine()
            except (TypeError, RuntimeError):
                pass

        data = np.asarray(conn.get_data(output='dense'), dtype=np.float32)  # (n_ch, n_ch, n_freqs)

        if data.ndim != 3:
            raise ValueError(f"Expected dense data ndim=3 (n_ch,n_ch,n_freq). Got shape={data.shape}")

        # Determine frequency labels
        if band is not None:
            if isinstance(band, str):
                band = [band]
            if len(band) != data.shape[-1]:
                raise ValueError("band length must match n_freqs")
            freqs = np.asarray(band, dtype=object)
        else:
            freqs = getattr(conn, 'freqs', None)
            if freqs is None:
                raise AttributeError(
                    "no attribute 'freqs'."
                    "Provide 'band' argument for the last axis (n_freqs)."
                    )
            freqs = np.asarray(freqs, dtype=np.float32)

        il, jl = np.tril_indices(n_ch, k=-1)  # lower triangle, PSI: lower, spectral_FC: lower, AEC: both
        vals = data[il, jl, :]  # (n_edges, n_freqs)
        ch1 = [self.ch_names[i] for i in il]
        ch2 = [self.ch_names[j] for j in jl]
        n_edges = len(ch1)

        df = pd.DataFrame({
            'ch1': np.repeat(ch1, len(freqs)),
            'ch2': np.repeat(ch2, len(freqs)),
            'freq': np.tile(freqs, n_edges),
            'value': vals.reshape(-1).astype(np.float32)
            })

        if band is not None:
            df = df.rename(columns={'freq': 'band'})

        if meta is not None:
            for k in reversed(list(meta.keys())):
                df.insert(0, k, meta[k])
        return df

    @staticmethod
    def to_band(df: pd.DataFrame,
                freqs_band: dict | None = None) -> pd.DataFrame:
        out = df.copy()
        if freqs_band is None:
            freqs_band = EEG_CFG.freqs_band
        BANDS = list(freqs_band.keys())
        if np.issubdtype(out["freq"].dtype, np.number):
            freqs = out["freq"].astype(float).to_numpy()
            band = np.full(len(out), None, dtype=object)
            for b, (flo, fhi) in freqs_band.items():
                if b not in BANDS:
                    continue
                band[(freqs >= flo) & (freqs < fhi)] = b
            out["band"] = band
            return out.dropna(subset=["band"])
        else:
            out["band"] = out["freq"].astype(str)
            return out[out["band"].isin(BANDS)]
