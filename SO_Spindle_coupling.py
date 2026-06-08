"""
SO-Spindle Coupling Detection
Based on Schreiner et al. (2021) & Staresina et al. (2015)

입력:
    slp_cat  : mne.concatenate_raws()로 만든 N2 연속 신호
               (BAD_boundary annotation이 이음새에 자동 삽입됨)
    bad_mask : bad_annot_mask(slp_cat) 로 만든 boolean array

사용 예시:
    results = run_so_spindle_coupling(slp_cat, bad_mask, ch_name='Fz')
"""

import numpy as np
import mne
from scipy.signal import hilbert
from scipy.stats import circmean
import matplotlib.pyplot as plt

def bandpass_filter(raw: mne.io.BaseRaw,
                    ch_name: str,
                    l_freq: float,
                    h_freq: float,
                    order: int = 3) -> np.ndarray:
    """
    Butterworth 밴드패스 필터 (bidirectional = zero phase).
    논문 방법과 동일: method='iir', order=3.
    raw는 수정하지 않고 filtered array만 반환.

    Returns
    -------
    filtered : np.ndarray (n_times,)
    """
    return (
        raw.copy()
           .pick(ch_name)
           .filter(l_freq=l_freq, h_freq=h_freq,
                   method='iir',
                   iir_params={'order': order, 'ftype': 'butter'},
                   verbose=False)
           .get_data()[0]
    )

def detect_SOs(raw: mne.io.BaseRaw,
               bad_mask: np.ndarray,
               ch_name: str,
               l_freq: float = 0.16,
               h_freq: float = 1.25,
               min_dur: float = 0.8,
               max_dur: float = 3.0,
               amplitude_percentile: float = 75) -> list[dict]:
    """
    SO detection (Schreiner et al., 2021 기반)

    Parameters
    ----------
    raw          : concatenate_raws()로 만든 N2 신호
    bad_mask     : bad_annot_mask() 결과 (BAD_boundary 포함)
    ch_name      : 분석 채널

    Returns
    -------
    list of dict (각 SO):
        trough_sample, trough_time, peak_sample, peak_time,
        start_sample, end_sample, amplitude
    """
    sfreq = raw.info['sfreq']

    # 1) 밴드패스 필터
    filtered = bandpass_filter(raw, ch_name, l_freq, h_freq)

    # 2) Positive-to-negative zero crossing
    sign = np.sign(filtered)
    sign[sign == 0] = 1
    zero_crossings = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0]

    # 3) 연속 zero crossing 쌍 → 후보 SO
    min_samp = int(min_dur * sfreq)
    max_samp = int(max_dur * sfreq)

    candidates = [
        (zero_crossings[i], zero_crossings[i + 1])
        for i in range(len(zero_crossings) - 1)
        if min_samp <= zero_crossings[i + 1] - zero_crossings[i] <= max_samp
    ]

    if not candidates:
        print(f"[SO] {ch_name}: 후보 없음")
        return []

    # 4) Bad 구간 포함 제외 (BAD_boundary 포함)
    valid = [
        (s, e) for s, e in candidates
        if not bad_mask[s:min(e, len(bad_mask))].any()
    ]
    print(f"[SO] {ch_name}: 후보 {len(candidates)}개 → Bad 제외 후 {len(valid)}개")

    # 5) Amplitude range (peak - trough) 계산
    details = []
    for s, e in valid:
        seg      = filtered[s:e]
        trough_i = int(np.argmin(seg))
        peak_i   = int(np.argmax(seg))
        details.append({
            'start_sample':  s,
            'end_sample':    e,
            'trough_sample': s + trough_i,
            'peak_sample':   s + peak_i,
            'amplitude':     seg[peak_i] - seg[trough_i],
        })

    # 6) 75th percentile threshold
    threshold = np.percentile([d['amplitude'] for d in details], amplitude_percentile)
    SOs = []
    for d in details:
        if d['amplitude'] >= threshold:
            d['trough_time'] = d['trough_sample'] / sfreq
            d['peak_time']   = d['peak_sample']   / sfreq
            SOs.append(d)

    print(f"[SO] {ch_name}: {len(SOs)}개 검출 (threshold={threshold*1e6:.1f} µV)")
    return SOs

def detect_spindles(raw: mne.io.BaseRaw,
                    bad_mask: np.ndarray,
                    ch_name: str,
                    l_freq: float = 12.0,
                    h_freq: float = 16.0,
                    rms_window: float = 0.2,
                    min_dur: float = 0.5,
                    max_dur: float = 3.0,
                    amplitude_percentile: float = 75) -> list[dict]:
    """
    Spindle detection (Staresina et al., 2015 기반)

    Returns
    -------
    list of dict (각 spindle):
        peak_sample, peak_time, start_sample, end_sample, peak_rms
    """
    sfreq    = raw.info['sfreq']
    filtered = bandpass_filter(raw, ch_name, l_freq, h_freq)
    n_times  = len(filtered)

    # RMS sliding window (cumsum으로 효율화)
    hw  = int(rms_window * sfreq) // 2
    cs  = np.concatenate([[0], np.cumsum(filtered ** 2)])
    s_w = np.maximum(np.arange(n_times) - hw, 0)
    e_w = np.minimum(np.arange(n_times) + hw, n_times)
    rms = np.sqrt((cs[e_w] - cs[s_w]) / (e_w - s_w))

    # Bad 구간 NaN → percentile 계산에서 제외
    rms_clean           = rms.copy()
    rms_clean[bad_mask] = np.nan
    threshold           = np.nanpercentile(rms_clean, amplitude_percentile)

    # threshold 초과 구간 추출
    above = np.where(~np.isnan(rms_clean), rms_clean > threshold, False).astype(int)
    diff  = np.diff(above, prepend=0, append=0)
    seg_s = np.where(diff ==  1)[0]
    seg_e = np.where(diff == -1)[0]

    min_samp = int(min_dur * sfreq)
    max_samp = int(max_dur * sfreq)

    spindles = []
    for s, e in zip(seg_s, seg_e):
        if not (min_samp <= e - s <= max_samp):
            continue
        if bad_mask[s:e].any():
            continue
        peak_i = int(np.argmax(rms[s:e]))
        spindles.append({
            'start_sample': s,
            'end_sample':   e,
            'peak_sample':  s + peak_i,
            'peak_time':    (s + peak_i) / sfreq,
            'peak_rms':     float(rms[s + peak_i]),
        })

    print(f"[Spindle] {ch_name}: {len(spindles)}개 검출 (threshold={threshold*1e6:.2f} µV)")
    return spindles

def detect_coupling(SOs: list[dict],
                    spindles: list[dict],
                    raw: mne.io.BaseRaw | None = None,
                    ch_name: str | None = None,
                    up_state_window_deg: float = 180.0,
                    so_l_freq: float = 0.16,
                    so_h_freq: float = 1.25) -> list[dict]:
    """
    각 SO 구간 안에 spindle peak이 있고, 그 spindle peak의 SO phase가
    up-state window(0° = SO peak 중심) 안에 있으면 coupling event로 기록.

    Parameters
    ----------
    up_state_window_deg : float
        Up-state phase window (전체 폭, deg). 0° = SO 위쪽 peak.
            - 360.0  : phase 무시 (SO 전체 사이클, 시간 기반만)
            - 180.0  : ±90° (up-state 절반, 기본값)
            -  90.0  : ±45° (up-state peak 근처만)
            -  ...
    raw, ch_name :
        up_state_window_deg < 360 일 때 SO phase 계산을 위해 필수.
    so_l_freq, so_h_freq : SO band-pass for phase 계산 (기본 0.16-1.25 Hz).

    Returns
    -------
    list of dict (각 coupling event):
        so, spindle, trough_sample, trough_time,
        spindle_peak_time, spindle_peak_phase_deg, time_diff
    """
    if not SOs or not spindles:
        print("[Coupling] SO 또는 Spindle 없음")
        return []

    use_phase_filter = up_state_window_deg < 360.0
    if use_phase_filter and (raw is None or ch_name is None):
        raise ValueError(
            "up_state_window_deg < 360 인 경우 raw, ch_name 필수"
        )

    # SO phase 계산 (Hilbert): 0° = up-state peak, ±180° = trough
    if use_phase_filter:
        filtered    = bandpass_filter(raw, ch_name, so_l_freq, so_h_freq)
        so_phase    = np.angle(hilbert(filtered))
        half_window = np.deg2rad(up_state_window_deg / 2.0)
    else:
        so_phase    = None

    sp_peaks = np.array([sp['peak_sample'] for sp in spindles])

    coupling_events = []
    n_phase_rejected = 0
    for so in SOs:
        in_so = np.where(
            (sp_peaks >= so['start_sample']) &
            (sp_peaks <= so['end_sample'])
        )[0]

        if len(in_so) == 0:
            continue

        # Phase-based filter: spindle peak이 up-state window 안에 있는지
        if use_phase_filter:
            kept = []
            for j in in_so:
                ph = so_phase[spindles[j]['peak_sample']]
                if abs(ph) <= half_window:   # |phase| ≤ half-window  → up-state
                    kept.append(j)
                else:
                    n_phase_rejected += 1
            in_so = kept
            if len(in_so) == 0:
                continue

        best = in_so[np.argmax([spindles[j]['peak_rms'] for j in in_so])]
        sp   = spindles[best]

        sp_phase_deg = (
            float(np.degrees(so_phase[sp['peak_sample']]))
            if use_phase_filter else np.nan
        )

        coupling_events.append({
            'so':                     so,
            'spindle':                sp,
            'trough_sample':          so['trough_sample'],
            'trough_time':            so['trough_time'],
            'spindle_peak_time':      sp['peak_time'],
            'spindle_peak_phase_deg': sp_phase_deg,
            'time_diff':              sp['peak_time'] - so['trough_time'],
        })

    msg = (f"[Coupling] {len(SOs)}개 SO 중 {len(coupling_events)}개 "
           f"({100*len(coupling_events)/len(SOs):.1f}%) coupling 검출")
    if use_phase_filter:
        msg += (f"  |  up-state ±{up_state_window_deg/2:.0f}°,  "
                f"phase-rejected: {n_phase_rejected}")
    print(msg)
    return coupling_events

def extract_coupling_epochs(raw: mne.io.BaseRaw,
                            coupling_events: list[dict],
                            bad_mask: np.ndarray,
                            ch_name: str,
                            tmin: float = -4.0,
                            tmax: float =  4.0) -> tuple[np.ndarray, np.ndarray]:
    """
    SO trough 중심 [tmin, tmax] 구간 epoch 추출.
    Bad 포함 / 경계 밖 epoch은 제외.

    Returns
    -------
    epochs : np.ndarray (n_valid, n_times)
    times  : np.ndarray (n_times,)
    """
    sfreq   = raw.info['sfreq']
    data    = raw.get_data(picks=ch_name)[0]
    n_total = len(data)

    pre   = int(abs(tmin) * sfreq)
    post  = int(tmax      * sfreq)
    times = np.arange(-pre, post) / sfreq

    epochs, rejected = [], 0
    for ev in coupling_events:
        c = ev['trough_sample']
        s, e = c - pre, c + post
        if s < 0 or e > n_total:
            rejected += 1
            continue
        if bad_mask[s:e].any():
            rejected += 1
            continue
        epochs.append(data[s:e])

    print(f"[Epochs] {len(epochs)}개 추출 ({rejected}개 제외)")
    return np.array(epochs), times

def compute_coupling_phase(raw: mne.io.BaseRaw,
                           coupling_events: list[dict],
                           bad_mask: np.ndarray,
                           ch_name: str,
                           l_freq: float = 0.16,
                           h_freq: float = 1.25) -> np.ndarray:
    """
    SO band Hilbert phase에서 spindle peak 시점의 위상 추출.

    Returns
    -------
    phases : np.ndarray (n_valid,)  단위: radians [-π, π]
    """
    filtered = bandpass_filter(raw, ch_name, l_freq, h_freq)
    phase    = np.angle(hilbert(filtered))

    phases = []
    for ev in coupling_events:
        sp_samp = ev['spindle']['peak_sample']
        if bad_mask[sp_samp]:
            continue
        phases.append(phase[sp_samp])

    return np.array(phases)

def summarize_coupling(SOs: list[dict],
                       spindles: list[dict],
                       coupling_events: list[dict],
                       phases: np.ndarray,
                       print_summary: bool = True) -> dict:
    """
    Coupling 결과 요약 통계.

    Returns
    -------
    dict: n_SO, n_spindle, n_coupled, coupled_ratio, MRL, mean_phase_deg
    """
    n_so      = len(SOs)
    n_sp      = len(spindles)
    n_coupled = len(coupling_events)

    if len(phases) > 0:
        mrl            = float(np.abs(np.mean(np.exp(1j * phases))))
        mean_phase_deg = float(np.degrees(
            circmean(phases, low=-np.pi, high=np.pi)
        ))
    else:
        mrl = mean_phase_deg = np.nan

    summary = {
        'n_SO':           n_so,
        'n_spindle':      n_sp,
        'n_coupled':      n_coupled,
        'coupled_ratio':  n_coupled / max(n_so, 1),
        'MRL':            mrl,
        'mean_phase_deg': mean_phase_deg,
    }

    if print_summary:
        print("\n── Coupling Summary ─────────────────")
        print(f"  SO:          {n_so}")
        print(f"  Spindle:     {n_sp}")
        print(f"  Coupled:     {n_coupled}  ({100*summary['coupled_ratio']:.1f}%)")
        print(f"  MRL:         {mrl:.4f}")
        print(f"  Mean phase:  {mean_phase_deg:.1f}°")
        print("─────────────────────────────────────\n")
    return summary

def _spindle_prob_histogram(coupling_events: list[dict],
                            SOs: list[dict],
                            times: np.ndarray,
                            sfreq: float,
                            bin_width: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    """
    SO trough 기준 spindle peak 발생 확률 히스토그램 계산.
    Image 1의 bar plot에 해당.

    Returns
    -------
    bin_centers : np.ndarray  (n_bins,)
    prob        : np.ndarray  (n_bins,)  단위: %
    chance      : float       chance level (%)  = 전체 spindle / 전체 SO / time_range
    """
    tmin, tmax    = times[0], times[-1]
    bins          = np.arange(tmin, tmax + bin_width, bin_width)
    bin_centers   = (bins[:-1] + bins[1:]) / 2

    time_diffs    = np.array([ev['time_diff'] for ev in coupling_events])
    counts, _     = np.histogram(time_diffs, bins=bins)

    n_SO          = max(len(SOs), 1)
    prob          = counts / n_SO * 100        # % per bin

    # chance level: uniform 분포 가정
    chance        = len(coupling_events) / n_SO / len(bin_centers) * 100
    return bin_centers, prob, chance

def _compute_tfr_epochs(epochs: np.ndarray,
                        times: np.ndarray,
                        sfreq: float,
                        freqs: np.ndarray | None = None,
                        n_cycles: float | np.ndarray = 7.0,
                        baseline_tmax: float = -0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    MNE tfr_array_morlet 기반 TFR (z-score normalized).

    Parameters
    ----------
    epochs        : (n_epochs, n_times)
    times         : (n_times,)
    sfreq         : sampling frequency
    freqs         : 분석 주파수 배열. None이면 2~25 Hz (48개)
    n_cycles      : Morlet wavelet cycles. float이면 전 주파수 동일,
                    array이면 주파수별로 다르게 (예: freqs / 2)
    baseline_tmax : z-score baseline 끝 시점 (초). 기본 tmin ~ -0.5s

    Returns
    -------
    tfr_z : np.ndarray (n_freqs, n_times)  z-score
    freqs : np.ndarray (n_freqs,)
    """
    from mne.time_frequency import tfr_array_morlet

    if freqs is None:
        freqs = np.linspace(2, 25, 48)

    # tfr_array_morlet 입력: (n_epochs, n_channels, n_times)
    epochs_3d = epochs[:, np.newaxis, :]   # (n_epochs, 1, n_times)

    # wavelet이 epoch보다 길어지지 않도록 n_cycles 자동 조절
    # wavelet 길이 ≈ 2 * 3 * n_cycles / (2π * freq) * sfreq
    max_cycles = np.floor(freqs * np.pi * epochs.shape[1] / (6 * sfreq))
    if np.isscalar(n_cycles):
        n_cycles_arr = np.minimum(n_cycles, max_cycles).clip(min=3)
    else:
        n_cycles_arr = np.minimum(np.asarray(n_cycles), max_cycles).clip(min=3)

    tfr = tfr_array_morlet(
        epochs_3d,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles_arr,
        output='power',         # (n_epochs, n_channels, n_freqs, n_times)
        verbose=False,
    )
    tfr = tfr[:, 0, :, :]      # (n_epochs, n_freqs, n_times)

    # baseline z-score (baseline: tmin ~ baseline_tmax)
    t_mask  = times < baseline_tmax
    bl      = tfr[:, :, t_mask]                            # (n_ep, n_freqs, n_bl)
    bl_mean = bl.mean(axis=(0, 2), keepdims=True)          # (1, n_freqs, 1) 후 squeeze
    bl_std  = bl.std(axis=(0, 2), keepdims=True)
    bl_std  = np.where(bl_std < 1e-30, 1e-30, bl_std)

    # epoch 평균 후 z-score
    tfr_mean = tfr.mean(axis=0)                            # (n_freqs, n_times)
    tfr_z    = (tfr_mean - bl_mean[0]) / bl_std[0]
    return tfr_z, freqs

def plot_coupling_results(epochs: np.ndarray,
                          times: np.ndarray,
                          coupling_events: list[dict],
                          SOs: list[dict],
                          summary: dict,
                          sfreq: float,
                          ch_name: str = '',
                          save_path: str | None = None) -> None:
    """
        (a) SO trough-locked spindle 발생 확률 히스토그램
            + SO 평균 파형 overlay
            + chance level
        (b) SO trough-locked TFR (Morlet, z-score)
            + SO 평균 파형 overlay
    """
    if len(epochs) == 0:
        print("[Plot] epoch 없음 - 시각화 생략")
        return

    so_mean  = epochs.mean(axis=0) * 1e6   # µV
    so_sem   = epochs.std(axis=0) / np.sqrt(len(epochs)) * 1e6

    # (a) histogram
    bin_centers, prob, chance = _spindle_prob_histogram(
        coupling_events, SOs, times, sfreq, bin_width=0.1
    )

    fig1, ax = plt.subplots(figsize=(7, 4))
    fig1.suptitle(f'SO-Spindle Coupling  |  ch: {ch_name}', fontsize=12)

    # SO 파형 overlay (scaled to fit on top)
    so_scaled = (so_mean - so_mean.min()) / (so_mean.max() - so_mean.min() + 1e-30)
    so_scaled = so_scaled * prob.max() * 1.1 + prob.max() * 0.05
    ax.plot(times, so_scaled, color='black', lw=2, zorder=5)

    # bar histogram
    ax.bar(bin_centers, prob, width=0.09,
           color='lightgray', edgecolor='dimgray', linewidth=0.5, zorder=3)
    ax.errorbar(bin_centers, prob,
                yerr=np.sqrt(prob / 100 * (1 - prob / 100) / max(len(SOs), 1)) * 100,
                fmt='none', ecolor='black', elinewidth=0.8, capsize=2, zorder=4)

    # chance level
    ax.axhline(chance, color='red', ls='-', lw=1.5, alpha=0.8, label=f'Chance ({chance:.1f}%)')

    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set_xlabel('Time (s) relative to SO trough', fontsize=11)
    ax.set_ylabel('Probability (%) of SP center occurrence', fontsize=10)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.set_title(f'n={len(coupling_events)} coupling events  |  '
                 f'coupled ratio={100*summary["coupled_ratio"]:.1f}%', fontsize=10)

    plt.tight_layout()
    if save_path:
        p1 = save_path.replace('.', '_hist.')
        fig1.savefig(p1, dpi=150, bbox_inches='tight')
        print(f"[Plot] 저장: {p1}")
    plt.show()

    # (b) TFR
    print("[TFR] Morlet wavelet 계산 중...")
    tfr_z, freqs = _compute_tfr_epochs(epochs, times, sfreq)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.suptitle(f'SO-Spindle Coupling TFR  |  ch: {ch_name}', fontsize=12)

    im = ax2.imshow(tfr_z,
                    origin='lower',
                    aspect='auto',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='RdBu_r',
                    vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2, label='Power (z-score)')

    # SO 파형 overlay (주파수 축 하단에)
    so_norm = (so_mean - so_mean.min()) / (so_mean.max() - so_mean.min() + 1e-30)
    so_freq = so_norm * (freqs[-1] - freqs[0]) * 0.25 + freqs[0]
    ax2.plot(times, so_freq, color='black', lw=2, zorder=5)

    ax2.axvline(0, color='white', ls='--', lw=1, alpha=0.7)
    ax2.set_xlabel('Time (s) relative to SO trough', fontsize=11)
    ax2.set_ylabel('Frequency (Hz)', fontsize=11)
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(freqs[0], freqs[-1])

    plt.tight_layout()
    if save_path:
        p2 = save_path.replace('.', '_tfr.')
        fig2.savefig(p2, dpi=150, bbox_inches='tight')
        print(f"[Plot] 저장: {p2}")
    plt.show()

# MAIN
def Compute_SO_spindle_coupling(raw: mne.io.BaseRaw,
                                bad_mask: np.ndarray,
                                ch_name: str,
                                tmin: float = -1.5,
                                tmax: float =  1.5,
                                up_state_window_deg: float = 180.0,
                                save_path: str | None = None,
                                plot_results: bool = True,
                                print_summary: bool = True) -> dict:
    """
    SO-Spindle Coupling 전체 파이프라인.

    Parameters
    ----------
    raw       : mne.concatenate_raws()로 만든 N2 신호
    bad_mask  : bad_annot_mask(raw) 결과
    ch_name   : 분석 채널 (예: 'Fz')
    tmin/tmax : epoch 범위 (초) — Image 1,2 기준 ±1.5s
    up_state_window_deg : Up-state phase window (전체 폭, deg, 0° = SO peak).
                          180=±90°(default), 90=±45°, 360=phase 무시.
    save_path : figure 저장 경로 (None이면 저장 안 함)

    Returns
    -------
    dict: SOs, spindles, coupling_events, epochs, times, phases, summary
    """
    if print_summary:
        print(f"\n{'='*50}")
        print(f"  SO-Spindle Coupling  |  ch: {ch_name}")
        print(f"{'='*50}")

    sfreq           = raw.info['sfreq']
    SOs             = detect_SOs(raw, bad_mask, ch_name)
    spindles        = detect_spindles(raw, bad_mask, ch_name)
    coupling_events = detect_coupling(
        SOs, spindles,
        raw=raw, ch_name=ch_name,
        up_state_window_deg=up_state_window_deg,
    )
    epochs, times   = extract_coupling_epochs(raw, coupling_events, bad_mask, ch_name, tmin, tmax)
    phases          = compute_coupling_phase(raw, coupling_events, bad_mask, ch_name)
    summary         = summarize_coupling(SOs, spindles, coupling_events, phases, print_summary=print_summary)
    
    if plot_results:
        plot_coupling_results(epochs, times, coupling_events, SOs,
                              summary, sfreq, ch_name, save_path)

    return {
        'SOs':             SOs,
        'spindles':        spindles,
        'coupling_events': coupling_events,
        'epochs':          epochs,
        'times':           times,
        'phases':          phases,
        'summary':         summary,
    }