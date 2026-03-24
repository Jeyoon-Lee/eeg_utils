# -*- coding: utf-8 -*-
"""
eeg_utils/viz.py

Contains:
    Visualization functions from EEG_analysis_utils.py:
        plot_hypnogram      - Plot a hypnogram (sleep stage time series).
        plot_spectrogram    - Plot a time-frequency spectrogram.
        plot_topo           - Plot an EEG topographic map.
        plot_cluster_topo   - Plot a topographic map with cluster markers.
        plot_FC_topo        - Plot functional connectivity on a topomap.

    Figure-making utilities from figure_making.py:
        significance_star   - Return a significance star string for a p-value.
        add_sig_bar         - Draw a significance bar on an axis.
        _clean_nan          - Remove NaN values from a list of arrays.
        jitter_scatter      - Add jittered scatter points to a plot.
        create_subplot_axes - Create a grid of subplot axes.
        add_sig_bar_axes    - Draw a significance bar using axes coordinates.
        longitudinal_plot   - Box/violin plot for longitudinal comparisons.
        compare_base_plot   - Box/violin plot comparing groups to a baseline.
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as transforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from scipy import stats
from itertools import combinations
import copy
from typing import Optional
from config import EEG_CFG

# Import convert2dB from eeg_analysis (used internally by plot_spectrogram)
from eeg_utils.eeg_analysis import convert2dB


# ============================================================
# Hypnogram / Spectrogram
# ============================================================

def plot_hypnogram(hypno: np.ndarray,
                   time_axis: bool = True,
                   sfreq: Optional[float] = None,
                   ax: Optional[Axes] = None,
                   stage_map: dict = None):
    hypno = np.asarray(hypno)

    if stage_map is None:
        from config import EEG_CFG
        stage_map = EEG_CFG.sleep_staging.stage_mapping
    hypno_int = np.array([stage_map.get(item, None) for item in hypno])

    if time_axis:
        if sfreq is None:
            raise ValueError("'sfreq' must be provided when time_axis=True")
        times = np.arange(len(hypno)) / sfreq
        xlabel = 'Time (seconds)'
    else:
        times = np.arange(len(hypno))
        xlabel = 'Epoch'

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        created_ax = True

    ax.step(times, hypno_int, 'k', where='post', linewidth=1.5)
    ax.set_yticks(list(stage_map.values()))
    ax.set_yticklabels(list(stage_map.keys()))
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, times[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Stage')
    ax.invert_yaxis()
    ax.grid(True, axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("hypnogram")
    if created_ax:
        return fig, ax
    else:
        return ax


def plot_spectrogram(spect: np.ndarray,
                     freqs: np.ndarray,
                     times: np.ndarray,
                     fig: Figure,
                     ax: Axes,
                     colormap: str = 'nipy_spectral',
                     to_dB: bool = True,
                     normalization: bool = True):
    """
    Plot a time-freqeuncy spectrogram.

    Parameters
    ----------
    spect : ndarray, shape (F, T)
        Rows correspond to frequencies and columns correspond to time bins.
    freqs : ndarray
        Array of frequency values (Hz) associated with the spectrogram rows.
    times : ndarray
        Array of time values (s) associated with the spectrogram columns.
    fig : Figure
    ax : Axes
    colormap: str
        The default is 'nipy_spectral'.
    to_dB : bool
        If True, convert the power values (V²/Hz) to dB using 'convert2dB()'.
        The default is True.
    normalization: bool
        If True, apply percentile normalization.
        The default is True.

    Returns
    -------
    ax : Axes

    """
    def _freq_dmean(S):
        mu = np.nanmean(S, axis=1, keepdims=True)
        return S - mu

    if to_dB:
        spect = convert2dB(spect)
        unit = 'dB'
    else:
        unit = 'V²/Hz'

    if normalization:  # percentile normalization
        trimperc = 2.5
        vmin, vmax = np.nanpercentile(spect, [trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    dx = times[1] - times[0]
    dy = freqs[1] - freqs[0]
    extent = [times[0]-dx, times[-1]+dx, freqs[-1]+dy, freqs[0]-dy]

    im = ax.imshow(spect, extent=extent, aspect='auto', cmap=colormap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, label=f'PSD ({unit})', pad=0.01)
    cbar.ax.tick_params(labelsize=5)
    ax.set_xlabel(" Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    ax.invert_yaxis()
    ax.set_title("Spectrogram")
    return ax


# ============================================================
# Topographic maps
# ============================================================

def plot_topo(values, info,
              ch_type: str = 'eeg',
              ax: Axes | None = None,
              vlim=(None, None),
              contours=4,
              cmap='RdBu_r',
              mask=None,
              mask_params: dict | None = None,  # ← None으로
              **kwarg):

    if mask_params is None:
        mask_params = dict(marker='*', markersize=9,
                           markerfacecolor='yellow',
                           markeredgecolor='black',
                           markeredgewidth=0.5)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), layout='constrained')

    im, _ = mne.viz.plot_topomap(values, info,
                         ch_type=ch_type,
                         contours=contours,
                         cmap=cmap,
                         mask=mask,
                         mask_params=mask_params,
                         vlim=vlim,
                         axes=ax,
                         show=False,
                         **kwarg)

    return ax.get_figure(), ax, im


def plot_cluster_topo(values, info,
                      cluster_mask, sig_mask,
                      ch_type: str = 'eeg',
                      ax: Axes | None = None,
                      vlim=(None, None)
                      ):
    fig, ax, im = plot_topo(values, info, ax=ax, ch_type=ch_type,
                        mask=None, vlim=vlim)

    picks = mne.pick_types(info, **{ch_type: True})
    pos = mne.channels.layout._find_topomap_coords(info, picks=picks)

    # add cluster channel markers
    ax.scatter(
        pos[cluster_mask, 0],
        pos[cluster_mask, 1],
        s=35,
        facecolors='none', edgecolors='white',
        linewidth=1.2, zorder=1)

    # add significant cluster channel markers
    ax.scatter(
        pos[sig_mask, 0],
        pos[sig_mask, 1],
        s=80, marker='*',
        facecolor='yellow', edgecolors='black',
        linewidth=1.2, zorder=2
    )
    return fig, ax, im


def plot_FC_topo(M, info,
                 ch_order: list,
                 ch_type: str = 'eeg',
                 threshold: float | None = None,
                 vlim: tuple = (None, None),
                 ax: Axes | None = None,
                 colorbar: bool = True,
                 lw_max=4.0, alpha=0.8,
                 line_strength: bool = True,
                 cmap='RdYlGn'
                 ):
    n_ch = len(ch_order)
    picks = [info['ch_names'].index(c) for c in ch_order]
    pos2d = mne.channels.layout._find_topomap_coords(info, picks=picks)

    # plot the topographic map frame
    fig, ax, _ = plot_topo(np.full(n_ch, np.nan), info, ax=ax,
                           mask=None, contours=0, ch_type=ch_type)

    # colormap / normalization
    fc = M.copy()
    np.fill_diagonal(fc, np.nan)
    valid = fc[~np.isnan(fc)]
    if len(valid) == 0:
        return

    vmin, vmax = vlim[0], vlim[1]
    _vmin = vmin if vmin is not None else valid.min()
    _vmax = vmax if vmax is not None else valid.max()

    norm = Normalize(vmin=_vmin, vmax=_vmax)
    cmap_obj = plt.get_cmap(cmap)

    # draw edges on the frame
    for i in range(1, n_ch):
        for j in range(i):
            val = M[i, j]  # lower triangular matrix

            if threshold:
                if np.isnan(val) or np.abs(val) <= threshold:
                    continue
            else:
                if np.isnan(val):
                    continue

            color = cmap_obj(norm(val))
            lw = lw_max * norm(val) if line_strength else 1.0
            ax.plot(
                [pos2d[i, 0], pos2d[j, 0]],
                [pos2d[i, 1], pos2d[j, 1]],
                color=color, linewidth=lw, alpha=alpha, zorder=2,
            )

    # draw sensor positions
    ax.scatter(pos2d[:, 0], pos2d[:, 1],
               s=20, c='black', zorder=3, edgecolors='white'
               )
    offset = 0.01
    for i, ch in enumerate(ch_order):
        ax.text(pos2d[i, 0], pos2d[i, 1] + offset,
                ch,
                fontsize=6, ha='center', va='bottom',
                zorder=4)

    # add colorbar
    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, format="%.3f")

    return fig, ax


# ============================================================
# Figure-making utilities (from figure_making.py)
# ============================================================

def significance_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def add_sig_bar(ax, x1, x2, y, p_val, h=0.5, lw=1.0):
    ax.plot([x1, x1, x2, x2],
            [y, y+h, y+h, y],
            color='k',
            linewidth=lw)
    sig_val = significance_star(p_val)
    if sig_val == 'n.s.':
        ax.text((x1 + x2)/2, y+h,
                significance_star(p_val),
                ha='center',
                va='bottom')
    else:
        ax.text((x1 + x2)/2, y+h*0.8,
                significance_star(p_val),
                ha='center',
                va='bottom')


def _clean_nan(data_list):
    for li in range(len(data_list)):
        data_list[li] = np.array(data_list[li])[~np.isnan(data_list[li])]
    return data_list


def jitter_scatter(ax, positions, data_list, jitter_scale=0.05,
                   seed=99, s=18, alpha=0.7, edgecolor='k', linewidths=0.4,
                   **kwargs):

    rng = np.random.default_rng(seed)

    for xi, vals in zip(positions, data_list):
        vals = np.asarray(vals)
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            continue

        jitter_x = rng.normal(loc=xi, scale=jitter_scale, size=len(vals))

        ax.scatter(
            jitter_x, vals,
            s=s, alpha=alpha,
            edgecolor=edgecolor,
            linewidths=linewidths,
            **kwargs)
    return ax


def create_subplot_axes(n_plots, n_rows, n_cols, figsize=(12, 8), layout='constrained'):
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize,
                             layout=layout)
    axes = axes.ravel()

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    for extra in range(n_plots, len(axes)):
        axes[extra].set_visible(False)

    return fig, axes


def add_sig_bar_axes(ax, x1, x2, y_ax, p_val,
                     sig_bar_h, sig_lw,
                     sig_text_pad):
    """Draw sig bar using axes coords for y (prevents ylim expansion)."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y0 = y_ax
    y1 = y_ax + sig_bar_h

    ax.plot([x1, x1, x2, x2],
            [y0, y1, y1, y0],
            transform=trans, color="k", lw=sig_lw, clip_on=False)

    ax.text((x1 + x2) / 2, y1 + sig_text_pad, significance_star(p_val),
            transform=trans, ha="center", va="bottom", clip_on=False)


def longitudinal_plot(ax, data, title="", y_label="",
                      x_labels=None, y_start=None, positions=None,
                      color='#c7d6eb', plot_type='box', stat_test=False,
                      plot_jitter=False,
                      sig_lw=1.0,
                      sig_line_h=4,
                      sig_star_h=0.1,
                      compare_list: list | None = None,
                      **kwargs):
    """
    TODO: 나중에 더 상세히 적기

    compare_list: Paired t-test를 할 조합 리스트
     e.g. [(0, 1), (0,1), (0,2), (1,2), ...]
    tuple 내에는 반드시 int, index여야 함.
    """
    x_n = len(data)
    if positions is None:
        positions = np.arange(x_n)

    if x_labels is None:
        x_labels = [f"stage {i+1}" for i in range(x_n)]
    else:
        assert len(x_labels) == x_n, \
            f"x_labels length must be {x_n}."

    if isinstance(color, list):
        if len(color) == 1:
            color = color * x_n
        else:
            assert len(color) == x_n, \
                f"color list length must be 1 or {x_n}."
    else:
        color = [color] * x_n

    clean_data = copy.deepcopy(data)
    clean_data = _clean_nan(clean_data)
    if plot_type == 'box':

        parts = ax.boxplot(
            clean_data,
            positions=positions,
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            **kwargs
            )
        for idx, patch in enumerate(parts['boxes']):
            patch.set_facecolor(color[idx])
            patch.set_alpha(0.8)

    elif plot_type == 'violin':

        parts = ax.violinplot(
            clean_data,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            **kwargs
            )
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color[idx])
            pc.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.3)

    if stat_test is not False:
        if compare_list is None:
            test_list = list(combinations(np.arange(x_n), 2))
        else:
            test_list = compare_list
        if y_start is None:
           ymax = max([np.nanmax(d) if len(d) > 0 else 0 for d in data])
           y_start = ymax + sig_line_h

        for i, (idx1, idx2) in enumerate(test_list):
            tg1, tg2 = np.array(data[idx1]), np.array(data[idx2])
            mask = ~np.isnan(tg1) & ~np.isnan(tg2)
            tg1_clean = tg1[mask]
            tg2_clean = tg2[mask]
            if len(tg1_clean) < 2 or len(tg2_clean) < 2:
                continue
            if stat_test == 'paired_ttest':
                assert len(tg1) == len(tg2), "the data length should be the same for running a paired t-test."
                t_stat, p_val = stats.ttest_rel(tg1_clean, tg2_clean)
            elif stat_test == 'ind_ttest':
                t_stat, p_val = stats.ttest_ind(tg1_clean, tg2_clean)

            y = y_start + i * sig_line_h
            add_sig_bar(ax, idx1, idx2, y, p_val, h=sig_star_h, lw=sig_lw)

    if plot_jitter is True:
        ax = jitter_scatter(ax,
                            positions=positions,
                            data_list=data,
                            jitter_scale=0.05)
    return ax


def compare_base_plot(ax, data, h=0.4, title="", y_label="",
                      x_labels=None, y_start=None, positions=None,
                      color='#c7d6eb', plot_type='box', stat_test=False,
                      baseline=0.0, plot_jitter=False, **kwargs):  # TODO: 나중에 제거
    x_n = len(data)
    if positions is None:
        positions = np.arange(x_n)

    if x_labels is None:
        x_labels = [f"stage {i+1}" for i in range(x_n)]
    else:
        assert len(x_labels) == x_n, \
            f"x_labels length must be {x_n}."

    if isinstance(color, list):
        if len(color) == 1:
            color = color * x_n
        else:
            assert len(color) == x_n, \
                f"color list length must be 1 or {x_n}."
    else:
        color = [color] * x_n

    clean_data = copy.deepcopy(data)
    clean_data = _clean_nan(clean_data)

    if plot_type == 'box':

        parts = ax.boxplot(
            clean_data,
            positions=positions,
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            **kwargs
            )
        for idx, patch in enumerate(parts['boxes']):
            patch.set_facecolor(color[idx])
            patch.set_alpha(0.8)

    elif plot_type == 'violin':

        parts = ax.violinplot(
            clean_data,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            **kwargs
            )
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color[idx])
            pc.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.3)

    # whether the valuse are different from the baseline (1-sample t-test)
    if not stat_test is False:
        if y_start is None:
            ymax = max(np.nanmax(li) if len(li) > 0 else 0 for li in data)
            y_start = ymax + 0.8
        assert isinstance(baseline, (int, float)), \
            f"baseline should be an integer or float for {stat_test}"
        for j, (xi, vals) in enumerate(zip(positions, data)):
            vals = np.asarray(vals)
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            t_stat, p_val = stats.ttest_1samp(vals, baseline)
            y = y_start + j * h * 0.0
            ax.text(xi, y, significance_star(p_val),
                    ha='center', va='bottom', fontsize=10)

    if plot_jitter is True:
        ax = jitter_scatter(ax,
                            positions=positions,
                            data_list=data,
                            jitter_scale=0.05)

    return ax
