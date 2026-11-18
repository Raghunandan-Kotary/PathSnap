"""Shared paths and constants for the PathSnap repo.

Raw data is NOT committed (87 MB). Point PATHSNAP_DATA at the folder that holds
`train/` and `student_test/`; the default assumes the layout used during development
(`<repo>/../Student folder`).

The evaluator-testing partition is never available here and must never be referenced.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT.parent / "data" / "raw"))

TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"
PARTITION_DIRS = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}

# Distances are computed in a per-trip local projection (see geo.py). The trips are spread over
# the whole world, so there is deliberately NO single global projected CRS here.

GLOBAL_SEED = 42  # one seed for the whole team's harness; every run also logs it explicitly

EDA_OUT = REPO_ROOT / "eda" / "outputs"
EXPERIMENT_LOGS = REPO_ROOT / "experiments" / "logs"
TUNING_DIR = REPO_ROOT / "tuning"
MODELS_DIR = REPO_ROOT / "models"
