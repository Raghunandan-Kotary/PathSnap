"""Hand-computed checks for the metric definitions and the harness guards.

    python experiments/tests/test_harness.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from person2.experiments.harness import load_table, run_experiment  # noqa: E402
from person2.experiments.metrics import candidate_metrics, observation_metrics  # noqa: E402
from person2.experiments.models import GAUSSIAN_NB  # noqa: E402


def test_top1_and_miss_rate():
    pred = pd.DataFrame({
        "trip_id": ["t"] * 6, "obs_idx": [0, 0, 1, 1, 2, 2], "arc_id": [10, 11, 20, 21, 30, 31],
        "label": [0, 1, 1, 1, 0, 0], "proba": [0.9, 0.8, 0.2, 0.7, 0.6, 0.4],
    })
    obs = pd.DataFrame({"trip_id": "t", "obs_idx": [0, 1, 2, 3], "n_candidates": [2, 2, 2, 0], "n_positive": [1, 2, 0, 0]})
    m = observation_metrics(pred, obs)
    assert m["top1_acc"] == 1 / 3, m             # obs0 wrong, obs1 right, obs2 wrong
    assert m["top1_acc_hit_only"] == 0.5, m      # only obs0 and obs1 contain a positive
    assert m["candidate_miss_rate"] == 0.5, m    # obs2 and obs3 (no candidates) out of 4
    assert observation_metrics(pred, None)["candidate_miss_rate"] == 1 / 3  # candidates-only scope


def test_tie_break_is_deterministic():
    pred = pd.DataFrame({"trip_id": ["t", "t"], "obs_idx": [0, 0], "arc_id": [7, 3], "label": [0, 1], "proba": [0.5, 0.5]})
    assert observation_metrics(pred)["top1_acc"] == 1.0  # smaller arc_id (3, label 1) wins the tie


def test_candidate_metrics():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.2, 0.6])  # threshold .5 -> pred 1,0,0,1
    m = candidate_metrics(y, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert abs(m["macro_f1"] - 0.5) < 1e-12 and abs(m["pos_f1"] - 0.5) < 1e-12
    assert abs(m["log_loss"] - np.mean(-np.log([0.9, 0.4, 0.8, 0.4]))) < 1e-9


def _toy_table(n_trips=10, rows=60, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_trips):
        n = rows
        x = rng.normal(size=(n, 3))
        y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
        parts.append(pd.DataFrame({
            "trip_id": f"{i:08d}", "obs_idx": np.arange(n) // 3, "arc_id": np.arange(n),
            "label": y, "f_a": x[:, 0], "f_b": x[:, 1], "f_c": x[:, 2],
            "partition": "student_test" if i >= 8 else "train", "fold": -1 if i >= 8 else i % 4,
        }))
    return pd.concat(parts, ignore_index=True)


def test_schema_guards():
    d = Path(tempfile.mkdtemp())
    t = _toy_table()
    t.to_parquet(d / "ok.parquet", index=False)
    load_table(d / "ok.parquet")
    bad = t.copy()
    bad.loc[bad.trip_id == "00000000", "partition"] = np.where(np.arange((bad.trip_id == "00000000").sum()) % 2, "train", "student_test")
    bad.to_parquet(d / "bad.parquet", index=False)
    try:
        load_table(d / "bad.parquet")
    except ValueError:
        pass
    else:
        raise AssertionError("a trip split across partitions must be rejected")
    ev = t.copy()
    ev.loc[ev.trip_id == "00000009", "partition"] = "evaluator_test"
    ev.to_parquet(d / "ev.parquet", index=False)
    try:
        load_table(d / "ev.parquet")
    except ValueError:
        pass
    else:
        raise AssertionError("an evaluator_test partition must be rejected")


def test_harness_alignment_and_no_nan():
    d = Path(tempfile.mkdtemp())
    t = _toy_table()
    t.to_parquet(d / "t.parquet", index=False)
    df, obs, cols = load_table(d / "t.parquet")
    res = run_experiment(GAUSSIAN_NB, df, cols, obs, smoke=True, log_dir=d / "logs", verbose=False)
    oof = pd.read_parquet(res["run_dir"] / "oof_proba_train.parquet")
    tst = pd.read_parquet(res["run_dir"] / "proba_student_test.parquet")
    tr, te = df[df.partition == "train"], df[df.partition == "student_test"]
    assert not oof.proba.isna().any() and not tst.proba.isna().any()
    assert (oof[["trip_id", "obs_idx", "arc_id"]].to_numpy() == tr[["trip_id", "obs_idx", "arc_id"]].to_numpy()).all()
    assert (tst[["trip_id", "obs_idx", "arc_id"]].to_numpy() == te[["trip_id", "obs_idx", "arc_id"]].to_numpy()).all()
    assert len(res["metrics"]["cv_per_fold_macro_f1"]) == 4


def test_promote_and_load_reproduces_logged_probabilities():
    import subprocess

    from person2.models.load import load_model

    d = Path(tempfile.mkdtemp())
    t = _toy_table()
    t.to_parquet(d / "t.parquet", index=False)
    df, obs, cols = load_table(d / "t.parquet")
    res = run_experiment(GAUSSIAN_NB, df, cols, obs, smoke=True, save_model=True, log_dir=d / "logs", verbose=False)
    out = d / "final"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "models" / "save_final.py"), "--run-dir", str(res["run_dir"]),
                    "--out-dir", str(out)], check=True, capture_output=True)
    model = load_model(out)
    test = df[df.partition == "student_test"]
    stored = pd.read_parquet(res["run_dir"] / "proba_student_test.parquet").proba.to_numpy()
    assert np.allclose(model.predict_proba(test), stored, atol=1e-9)
    # ground truth must not be able to influence a prediction: scrambling `label` changes nothing
    scrambled = test.assign(label=1 - test.label)
    assert np.allclose(model.predict_proba(scrambled), stored, atol=1e-9)
    try:
        model.predict_proba(test.drop(columns=[cols[0]]))
    except KeyError:
        pass
    else:
        raise AssertionError("missing feature column must raise")
    # a smoke run must not be promotable to models/final
    r = subprocess.run([sys.executable, str(root / "models" / "save_final.py"), "--run-dir", str(res["run_dir"])],
                       capture_output=True)
    assert r.returncode != 0, "a smoke run must not be promotable to models/final"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
