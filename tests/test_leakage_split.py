"""Tests for Step 4 leakage + split (synthetic, no files)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate
from src.preprocessing.preprocess import parse_dates, qa_fill, add_wind_season, add_lags
from src.preprocessing.leakage_checks import target_check, dup_check, overlap_check, suspicious_check
from src.data.split import time_split, cv_folds

def _proc(n=40):
    df = parse_dates(generate(n_days=n, seed=1))
    df = qa_fill(add_lags(add_wind_season(df)))
    return df

def test_target_and_dup_checks():
    df = _proc()
    t = target_check(df)
    assert any("OK" in x or "corr" in x.lower() or "banned" in x.lower() for x in t)
    d = dup_check(df)
    assert any("duplicate rows: 0" in x for x in d)

def test_time_split_never_random():
    df = _proc()
    p = time_split(df, 0.70, 0.15, 0.15, "train_val_test")
    assert set(p) == {"train", "val", "test"}
    assert p["train"]["date"].max() <= p["val"]["date"].min()
    assert p["val"]["date"].max() <= p["test"]["date"].min()
    o = overlap_check(p["train"], p["test"])
    assert any("(OK)" in x for x in o)

def test_cv_grouped_train_only():
    df = _proc()
    p = time_split(df, 0.70, 0.15, 0.15, "train_val_test")
    f = cv_folds(p["train"], n=5)
    assert set(f.unique()) <= set(range(5))
    s = suspicious_check(p["train"])
    assert len(s) > 0
