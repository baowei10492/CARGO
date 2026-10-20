"""Constants of the P33 reviewer experiments (docs/P33_REVIEWER_EXPERIMENTS.md), kept free of heavy imports so that the
DiffRegCD environment can use them too."""
from __future__ import annotations

from rs_cd.affine_registration import AffineRanges

# Interpolation control: level 1 of these ranges moves T2 by at most 0.5 px and scales it by at most 0.5 % about the
# crop centre (the per-crop draw of the paper's levels, shrunk), so the raw input is resampled like levels 1-3 are,
# without their misalignment.
INTERP_RANGES = AffineRanges(0.0, 0.985, 1.015, 1.5, 3)
INTERP_LEVEL = 1

# S2Looking (Shen et al., Remote Sensing 2021): never used to train the registration module.
S2LOOKING = {"name": "S2Looking", "dataset_root": "/root/autodl-tmp/rs_cd_work/data/S2Looking",
             "manifest": "manifests/datasets/s2looking.csv", "split": "test", "tile_size": 1024, "mask_threshold": 127}
S2LOOKING_DETECTOR_POINTER = "outputs/p33_s2looking_detector/CHECKPOINT"

# Registration modules retrained without change-aware weights or without change supervision (pointers to run dirs).
RETRAINED_LOCALIZERS = {
    "v2r_conf_only": "outputs/p33_localizer_conf_only/LATEST",
    "v2r_no_change_label": "outputs/p33_localizer_no_change_label/LATEST",
}
