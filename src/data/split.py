"""Step 4b: time-based train/val/test split + unseen-district hold-out + 10-fold CV plan.
NEVER random: latest months = test. Then StratifiedGroupKFold(grouped by district)
on TRAINING data only (fold ids saved). Supports train_test mode via config.
Run: python src/data/split.py --in data/processed/india_proc.csv --outdir data/processed
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def load_cfg(p: str) -> dict:
    """Load YAML config (fails loudly if missing)."""
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def time_split(df: pd.DataFrame, tr: float, va: float, te: float, mode: str) -> dict[str, pd.DataFrame]:
    """Split by date quantiles: earliest=train, latest=test. Returns dict of dfs."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if mode == "train_test":
        c = df["date"].quantile(1 - te)
        return {"train": df[df["date"] <= c], "test": df[df["date"] > c]}
    c1 = df["date"].quantile(tr)
    c2 = df["date"].quantile(tr + va)
    return {"train": df[df["date"] <= c1],
            "val": df[(df["date"] > c1) & (df["date"] <= c2)],
            "test": df[df["date"] > c2]}

def cv_folds(train: pd.DataFrame, n: int = 10, group: str = "district",
             target: str = "aqi_class", seed: int = 42) -> pd.Series:
    """StratifiedGroupKFold ids on TRAIN only. FAILS loudly if sklearn missing."""
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    ids = pd.Series(-1, index=train.index)
    for i, (_, vi) in enumerate(sg.split(train, train[target], train[group])):
        ids.iloc[vi] = i
    return ids

def run(ind: str, outdir: str, cfgp: str = "config/settings.yaml") -> dict[str, Path]:
    """Write train/val/test CSVs + fold ids + hold-out. Returns paths."""
    cfg = load_cfg(cfgp)
    sp = cfg.get("split", {}) or {}
    mode = sp.get("mode", "train_val_test")
    tr, va, te = float(sp.get("train_frac", 0.70)), float(sp.get("val_frac", 0.15)), float(sp.get("test_frac", 0.15))
    nfold = int(sp.get("n_folds", 10))
    grp = sp.get("group_col", "district")
    holds = list(sp.get("holdout_districts", ["Leh", "Kavaratti", "Itanagar"]))
    df = pd.read_csv(ind, encoding="utf-8")
    parts = time_split(df, tr, va, te, mode)
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for k, d in parts.items():
        p = o / f"{k}.csv"
        d.to_csv(p, index=False, encoding="utf-8")
        paths[k] = p
    # unseen-district hold-out carved from TEST (or full data if missing there)
    test = parts.get("test", df)
    ho = test[test["district"].isin(holds)]
    if len(ho) == 0:
        ho = df[df["district"].isin(holds)]
    hp = o / "holdout.csv"
    ho.to_csv(hp, index=False, encoding="utf-8")
    paths["holdout"] = hp
    # 10-fold CV ids on TRAINING only
    train = parts["train"]
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    folds = cv_folds(train, n=nfold, group=grp if grp in train.columns else train.columns[0],
                     target=tgt if tgt in train.columns else train.columns[0])
    fp = o / "cv_folds.csv"
    pd.DataFrame({"row_id": train.index, "fold": folds.values}).to_csv(fp, index=False, encoding="utf-8")
    paths["folds"] = fp
    for k, p in paths.items():
        print(f"split {k}: {len(pd.read_csv(p, encoding='utf-8'))} rows -> {p}")
    print(f"holdout districts {holds}: {len(ho)} rows; folds: {nfold} (group={grp}, train-only)")
    return paths

def main() -> None:
    """CLI: python src/data/split.py --in data/processed/india_proc.csv --outdir data/processed"""
    ap = argparse.ArgumentParser(description="Time-based split + hold-out + CV (Step 4b)")
    ap.add_argument("--in", dest="ind", default="data/processed/india_proc.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.ind, a.outdir, a.config)

if __name__ == "__main__":
    main()
