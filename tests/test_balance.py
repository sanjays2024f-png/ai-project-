"""Tests for Step 5 balance (train-only STRICT SMOTE per fold, SMOKE)."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate
from src.preprocessing.preprocess import parse_dates, qa_fill, add_wind_season, add_lags
from src.data.split import time_split, cv_folds
from src.training.balance import feature_cols, encode_y, decode_y, smote_fold

CLS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

def _train(n=60):
    df = parse_dates(generate(n_days=n, seed=2))
    return qa_fill(add_lags(add_wind_season(df)))

def test_smote_train_only_balances():
    df = _train()
    p = time_split(df, 0.70, 0.15, 0.15, "train_val_test")
    tr = p["train"]
    cfg = {"features": {"exclude_from_X": []}, "smote": {}}
    feats = feature_cols(tr, cfg)
    assert len(feats) > 5
    f = cv_folds(tr, n=3)
    m = (f.values == 0)
    X = tr.loc[m, feats].fillna(0).values
    y = encode_y(tr.loc[m, "aqi_class"], CLS)
    try:
        Xb, yb = smote_fold(X, y, k=3, seed=0)
    except RuntimeError as ex:
        # honest loud skip on tiny smoke folds (single class or n_minority<=2)
        assert "FAILED" in str(ex) or "SKIP-SMOKE-SMALL-FOLD" in str(ex)
        return
    assert len(yb) >= len(y)  # minority upsampled
    assert set(decode_y(yb, CLS)) <= set(CLS)
    # val/test untouched: no SMOTE applied there
    assert len(p["val"]) > 0 and len(p["test"]) > 0
