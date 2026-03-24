# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 15:58:33 2025

@author: Jeyoon Lee
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Any

@dataclass(frozen=True)
class PreprocessConfig:
    bandpass_filt: bool = True
    bandpass_l_freq: float = 0.5
    bandpass_h_freq: float = 50.0
    notch_filt: bool = True
    notch_freq: float = 60.0
    re_referencing = 'average'
    
    ICA_denoising: bool = False  

    reject_amp: float = 280e-6  # Volts
    bad_band: str = "Delta"
    bad_power_thr: float = 1.001e-08


@dataclass(frozen=True)
class EpochConfig:
    length: float = 5.0   # seconds
    overlap: float = 2.5  # seconds


@dataclass(frozen=True)
class SleepDetectionConfig:
    epoch_len: float = 30.0 # seconds
    target_ch: str = 'P4'
    relabel_R_after_W: bool = True
    stage_mapping: dict[str, int] = field(default_factory=lambda: {
        'UNK': -1,
        "W"  : 0, 
        'R'  : 1,
        'N1' : 2,
        'N2' : 3,
        'N3' : 4
        })
        
            
@dataclass(frozen=True)
class EEGConfig:
    fs: int = 125
    montage = "standard_1020"
    reference = 'Pz'
    channel_names: list[str] = field(default_factory=lambda: [
        "Fp1", "F7", "F3", "T3",
        "C3", "Cz", "P3", "O1",
        "Fp2", "F4", "F8", "C4",
        "T4", "P4", "O2", "ecg", "trigger"
    ])
    channel_types: list[str] = field(default_factory=lambda: (["eeg"] * 15 + ["ecg"] + ["trigger"]))
    signal_loc: np.ndarray = field(default_factory=lambda: np.hstack([np.arange(1, 17), 31]))

    trigger_labels: dict[str, int] = field(default_factory=lambda: {
        "baseline": 1,
        "stimulation 1": 2,
        "recovery 1": 3,
        "stimulation 2": 4,
        "recovery 2": 5,
    })
    trigger_colors: dict[int, str] = field(default_factory=lambda: {
        1: "r",
        2: "g",
        3: "b",
        4: "m",
        5: "y",
    })

    freqs_band: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 45),
        "Sigma": (12, 15),
    })
    
    ROI: dict[str, list[str]] = field(default_factory=lambda: {
        'SM': ['C3', 'Cz', 'C4'],
        'FR': ['F4', 'F3', 'Fp1', 'Fp2'],
        'VN': ['O1', 'O2'],
        'VAL': ['T4', 'F8', 'T3', 'F7'],
        'DA': ['P3', 'P4']
        })

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    sleep_staging: SleepDetectionConfig = field(default_factory=SleepDetectionConfig)

@dataclass(frozen=True)
class QuestConfig:
    all_quest_list: list[str] = field(default_factory=lambda: ["IRLS", "PSQI", "ISI", "ESS", "COMPASS31", "BAI", "BDI-2", "PSS"])
    quest_tb_list: list[str] = field(default_factory=lambda: ["PSQI", "ISI", "ESS", "COMPASS31", "BAI", "BDI-2"])
    quest_ec_list: list[str] = field(default_factory=lambda: ["IRLS", "PSQI", "ISI", "ESS", "COMPASS31", "BAI", "BDI-2"])

    quest_rules: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "IRLS": {
            "bins": [-0.99, 0, 10, 20, 30, 40],
            "labels": ["no RLS", "mild", "moderate", "severe", "very severe"],
        },
        "PSQI": {
            "bins": [0, 5, 21],
            "labels": ["normal sleeper", "poor sleeper"],
        },
        "ISI": {
            "bins": [0, 7, 14, 21, 28],
            "labels": ["none", "subthreshold", "moderate", "severe"],
        },
        "ESS": {
            "bins": [0, 10, 24],
            "labels": ["normal", "excessive"],
        },
        "COMPASS31": {
            "bins": [0, 10, 20, 100],
            "labels": ["mild", "moderate", "severe"],
        },
        "BAI": {
            "bins": [0, 7, 15, 25, 63],
            "labels": ["minimal", "mild", "moderate", "severe"],
        },
        "BDI-2": {
            "bins": [0, 13, 19, 28, 63],
            "labels": ["minimal", "mild", "moderate", "severe"],
        },
        "PSS": {
            "bins": [0, 13, 26, 40],
            "labels": ["low", "moderate", "high"],
        },
    })

    abnormal_baseline: dict[str, list[str]] = field(default_factory=lambda: {
        "IRLS": ["mild", "moderate", "severe", "very severe"],
        "PSQI": ["poor sleeper"],
        "ISI": ["moderate", "severe"],
        "ESS": ["excessive"],
        "COMPASS31": ["moderate", "severe"],
        "BAI": ["mild", "moderate", "severe"],
        "BDI-2": ["mild", "moderate", "severe"],
        "PSS": ["moderate", "high"],
    })


EEG_CFG = EEGConfig()
QUEST_CFG = QuestConfig()
