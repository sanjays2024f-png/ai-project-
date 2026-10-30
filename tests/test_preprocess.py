"""Tests for Step 3 preprocess (uses synthetic generator, no files needed)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate
from src.preprocessing.preprocess import parse_dates, qa_fill, clip, add_wind_season, add_lags, fit_norm, apply_norm

def _df(n=30):
    df = generate(n_days=n, seed=0)
    return parse_dates(df)

def test_qa_fill_removes_nan():
    df = _df()
    assert df["aod"].isna().sum() > 0  # synthetic gaps exist
    f = qa_fill(df)
    assert f.isna().sum().sum() == 0

def test_wind_season_flags():
    df = add_wind_season(_df())
    assert {"wind_speed", "wind_dir", "is_winter", "is_monsoon", "is_diwali"} <= set(df.columns)
    assert (df["wind_speed"] >= 0).all()
    assert df["wind_dir"].between(0, 360).all()

def test_lags_and_train_only_norm():
    df = add_lags(add_wind_season(_df()))
    for c in ["pm25_lag1", "pm25_lag3", "pm25_lag7", "aod_roll3", "no2_roll7"]:
        assert c in df.columns, f"missing {c}"
    cut = df["date"].quantile(0.70)
    tr = df[df["date"] <= cut]
    st = fit_norm(tr, ["aod", "no2", "temp"])
    out = apply_norm(df, st)
    # train slice of normalised cols ~ mean 0
    m = out.loc[out["date"] <= cut, "aod_n"].mean()
    assert abs(m) < 0.05
