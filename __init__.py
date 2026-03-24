# -*- coding: utf-8 -*-
"""
eeg_utils package

Submodules:
    eeg_io          - EEG_Loader class + csv_to_raw() helper function
    eeg_preprocess  - EEG_Preprocessor, BadSegmentAnnotator, EEG_Epocher,
                      ECG_Processor classes + preprocess_raw(),
                      align_set_eeg_to_orig(), plot_info() functions +
                      bad_annot_mask() function
    eeg_analysis    - PSDAnalyzer, FCAnalyzer classes + convert2dB(),
                      create_mask(), safe_div_nan(), concat_epochs(),
                      apply_FDR(), extract_mixedlm_result() functions
    sleep_staging   - SleepDetector class + upsample_hypnogram() function
    viz             - plot_hypnogram(), plot_spectrogram(), plot_topo(),
                      plot_cluster_topo(), plot_FC_topo() and all
                      figure-making utilities
"""
