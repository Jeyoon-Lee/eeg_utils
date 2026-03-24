# eeg_utils

A personal Python utility package for EEG data loading, preprocessing, analysis, sleep staging, and visualization.

Most modules are built on [MNE-Python](https://mne.tools/stable/index.html).
Sleep staging is powered by [YASA](https://yasa-sleep.org/index.html).

*This is under active development. New features and improvements are continuously being added.

## Structure

```
eeg_utils/
├── eeg_io.py          # EEG data loading and MNE conversion
├── eeg_preprocess.py  # Filtering, re-referencing, ICA, epoching
├── eeg_analysis.py    # PSD, functional connectivity, statistics
├── sleep_staging.py   # Sleep stage detection and hypnogram utilities 
└── viz.py             # Visualization (topomaps, spectrograms, plots)
```

## Modules

### `eeg_io.py`
- `EEG_Loader` — Loads EEG data from CSV, converts to MNE `RawArray`
- `csv_to_raw()` — Helper function for CSV → MNE conversion

### `eeg_preprocess.py`
- `EEG_Preprocessor` — Bandpass/notch filtering, re-referencing, bad channel interpolation, ICA
- `BadSegmentAnnotator` — Marks bad segments based on amplitude/power thresholds
- `EEG_Epocher` — Creates fixed-length epochs with metadata
- `ECG_Processor` — R-peak detection, heart rate, HRV (RMSSD)
- `bad_annot_mask()` — Returns boolean mask for BAD-annotated segments
- `preprocess_raw()` — End-to-end preprocessing pipeline
- `plot_info()` — Summary figure after preprocessing

### `eeg_analysis.py`
- `PSDAnalyzer` — Absolute/relative band power, aperiodic (FOOOF) parameters
- `FCAnalyzer` — Spectral FC (PLV, PLI, wPLI, coherence), PSI, AEC
- `convert2dB()` — Power to dB conversion
- `create_mask()` — Frequency range boolean mask
- `safe_div_nan()` — Division with NaN fallback
- `concat_epochs()` — Reconstruct continuous signal from epochs
- `apply_FDR()` — FDR correction across channel/band groups
- `extract_mixedlm_result()` — Extract statsmodels MixedLM results to DataFrame

### `sleep_staging.py`
- `SleepDetector` — YASA-based sleep staging with post-processing
- `upsample_hypnogram()` — Epoch-level → sample-level hypnogram

### `viz.py`
- `plot_topo()` — MNE topographic map
- `plot_FC_topo()` — FC network overlay on topomap
- `plot_cluster_topo()` — Cluster-based significance topomap
- `plot_spectrogram()` — Time-frequency spectrogram
- `plot_hypnogram()` — Sleep stage hypnogram
- `jitter_scatter()` — Scatter plot with jitter
- `longitudinal_plot()` — Longitudinal group comparison with stats
- `compare_base_plot()` — Group vs. baseline comparison
- `add_sig_bar()` — Significance bar between groups
- `significance_star()` — p-value → significance symbol

## Dependencies

- `mne`, `mne-connectivity`
- `numpy`, `scipy`, `pandas`
- `matplotlib`, `seaborn`
- `yasa`, `fooof`, `pactools`
- `scikit-learn`, `statsmodels`
