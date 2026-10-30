"""Step 3: preprocess india_dataset.csv (QA masks, gap-fill, lags, wind, flags, norm).
Fit normaliser on TRAIN time slice only. Run:
  python src/preprocessing/preprocess.py --in data/raw/synthetic/india_dataset.csv --out data/processed/india_proc.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

LAG_COLS = ["pm25", "aod", "no2", "temp", "rain", "fire_count"]
ROLL_WINS = [3, 7]
LAGS = [1, 3, 7]
CLIP_Q = (0.005, 0.995)  # quantile clip per numeric col

def load_cfg(p: str) -> dict:
    """Load YAML config (fails loudly if PyYAML missing or path bad)."""
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date col -> datetime, sort by cell then date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["cell_id", "date"]).reset_index(drop=True)

def qa_fill(df: pd.DataFrame) -> pd.DataFrame:
    """QA gaps: NaN = cloud/snow mask. Time-interp per cell, then zone-day median."""
    df = df.copy()
    num = df.select_dtypes("number").columns.tolist()
    # 1) time interpolation within each cell (limit 3 days both directions)
    df[num] = df.groupby("cell_id")[num].transform(
        lambda g: g.interpolate("linear", limit=3, limit_direction="both"))
    # 2) MERRA-2 fallback ~ zone median (real pipeline uses M2T1NXAER/SLV grids)
    for c in num:
        if df[c].isna().any():
            zm = df.groupby("zone")[c].transform("median")
            df[c] = df[c].fillna(zm).fillna(df[c].median())
    return df

def clip(df: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric outliers to quantile range (per column)."""
    df = df.copy()
    for c in df.select_dtypes("number").columns:
        if c in ("cell_id", "lat", "lon", "elevation", "population"):
            continue
        lo, hi = df[c].quantile(CLIP_Q[0]), df[c].quantile(CLIP_Q[1])
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            df[c] = df[c].clip(lo, hi)
    return df

def add_wind_season(df: pd.DataFrame, diwali: list[str] | None = None) -> pd.DataFrame:
    """Wind speed/dir + winter/monsoon/Diwali flags."""
    df = df.copy()
    df["wind_speed"] = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
    df["wind_dir"] = (np.degrees(np.arctan2(df["wind_u"], df["wind_v"])) + 360) % 360
    m = df["date"].dt.month
    df["is_winter"] = m.isin([11, 12, 1, 2]).astype(int)
    df["is_monsoon"] = m.isin([6, 7, 8, 9]).astype(int)
    dw = set(diwali or ["2024-11-01"])
    ds = set(pd.to_datetime(list(dw)).strftime("%Y-%m-%d"))
    df["is_diwali"] = df["date"].dt.strftime("%Y-%m-%d").isin(ds).astype(int)
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lags 1/3/7d + rolling means 3/7d per cell (sorted)."""
    df = df.copy()
    for c in LAG_COLS:
        if c not in df.columns:
            continue
        g = df.groupby("cell_id")[c]
        for L in LAGS:
            df[f"{c}_lag{L}"] = g.shift(L)
        for w in ROLL_WINS:
            df[f"{c}_roll{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    # backfill early rows per cell with cell median then global median
    lagcols = [c for c in df.columns if "_lag" in c or "_roll" in c]
    for c in lagcols:
        df[c] = df.groupby("cell_id")[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    return df

def fit_norm(train: pd.DataFrame, cols: list[str]) -> dict:
    """Mean/std on TRAIN only. Returns {col: [mean, std]}."""
    st: dict = {}
    for c in cols:
        mu, sd = float(train[c].mean()), float(train[c].std() or 1.0)
        st[c] = [mu, sd if sd > 1e-9 else 1.0]
    return st

def apply_norm(df: pd.DataFrame, st: dict) -> pd.DataFrame:
    """Apply z-score using train stats (new col <c>_n)."""
    df = df.copy()
    for c, (mu, sd) in st.items():
        if c in df.columns:
            df[c + "_n"] = (df[c] - mu) / sd
    return df

NORM_COLS = ["aod", "no2", "so2", "co", "bc", "dust", "temp", "rh",
             "wind_u", "wind_v", "blh", "rain", "fire_count", "ndvi",
             "lst", "wind_speed", "wind_dir"]

def run(ind: str, outd: str, cfgp: str = "config/settings.yaml") -> pd.DataFrame:
    """Full Step 3 pipeline. Returns processed df, writes CSV + norm stats."""
    global LAGS
    cfg = load_cfg(cfgp)
    feat = (cfg.get("features", {}) or {})
    diw = ((cfg.get("split", {}) or {}).get("diwali_dates")
           or (cfg.get("seasons", {}) or {}).get("diwali_dates")
           or cfg.get("diwali_dates")
           or ["2024-11-01"])
    lags = feat.get("lags", LAGS)
    LAGS = lags  # honour config
    df = pd.read_csv(ind, encoding="utf-8")
    df = parse_dates(df)
    df = qa_fill(df)   # NaN -> time interp -> zone median (MERRA-2 fallback)
    df = clip(df)      # quantile outlier clip
    df = add_wind_season(df, diwali=diw)
    df = add_lags(df)
    # train-only normaliser: earliest 70% dates = pseudo-train slice
    cut = df["date"].quantile(0.70)
    tr = df[df["date"] <= cut]
    cols = [c for c in NORM_COLS if c in df.columns]
    st = fit_norm(tr, cols)
    df = apply_norm(df, st)
    o = Path(outd)
    o.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(o, index=False, encoding="utf-8")
    sp = o.parent / "norm_stats.json"
    sp.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"preprocess: {len(df)} rows x {len(df.columns)} cols -> {o}")
    print(f"norm stats ({len(st)} cols, train<={cut.date()}) -> {sp}")
    print("NaN left:", int(df.isna().sum().sum()))
    return df

def main() -> None:
    """CLI: python src/preprocessing/preprocess.py --in data/raw/synthetic/india_dataset.csv --out data/processed/india_proc.csv"""
    ap = argparse.ArgumentParser(description="Preprocess india dataset (Step 3)")
    ap.add_argument("--in", dest="ind", default="data/raw/synthetic/india_dataset.csv")
    ap.add_argument("--out", dest="outd", default="data/processed/india_proc.csv")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.ind, a.outd, a.config)

if __name__ == "__main__":
    main()
