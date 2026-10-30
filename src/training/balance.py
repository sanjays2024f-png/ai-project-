"""Step 5: SMOTE (imblearn) on TRAINING data only, inside each CV fold.
Never on validation or test. Logs class counts before/after per fold.
STRICT: requires imbalanced-learn + scikit-learn; fails loudly if missing.
SMOTE applies ONLY to classification targets, never to regression (pm25)
and never to validation/test. For graph tasks where SMOTE would break
spatial structure, use class-weighted loss instead (see train.py).
Run: python src/training/balance.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --outdir results/reports
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.require import need
import numpy as np
import pandas as pd

need("sklearn")
need("imblearn")
need("yaml")

def load_cfg(p: str) -> dict:
    """Load YAML config; FAILS loudly if PyYAML missing or file bad."""
    import yaml
    return yaml.safe_load(Path(p).read_text()) or {}

def feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Numeric _n cols (train-normed) + lags/flags; exclude targets/ids."""
    excl = set((cfg.get("features", {}) or {}).get("exclude_from_X", []))
    excl |= {"pm25", "aqi_class", "disease_risk", "date", "district", "state", "zone", "fold"}
    cols = [c for c in df.columns if c not in excl]
    # prefer normalised + engineered numeric cols
    num = df[cols].select_dtypes("number").columns.tolist()
    return num

def encode_y(s: pd.Series, classes: list[str]) -> np.ndarray:
    """Map class labels -> ints via config order."""
    m = {c: i for i, c in enumerate(classes)}
    return s.map(m).fillna(-1).astype(int).values

def decode_y(a: np.ndarray, classes: list[str]) -> list[str]:
    """Ints -> labels via config order."""
    return [classes[int(i)] for i in a]

def smote_fold(Xtr: np.ndarray, ytr: np.ndarray, k: int = 5, seed: int = 42,
               strategy: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """STRICT SMOTE on one training-fold classification target. No fallback.

    Raises RuntimeError (FAILED, not silent) if imblearn missing or fold has
    a single class. Never call on regression targets or val/test.
    """
    from imblearn.over_sampling import SMOTE
    Xtr = np.asarray(Xtr, dtype=float)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    uniq, cnt = np.unique(ytr, return_counts=True)
    if len(uniq) < 2:
        raise RuntimeError(f"FAILED: SMOTE needs >=2 classes in fold, got {dict(zip(uniq, cnt))}")
    # SMOTE needs k < n_minority; tiny smoke folds may have 1-2 samples -> honest skip with reason
    if int(cnt.min()) <= 2:
        raise RuntimeError(f"SKIP-SMOKE-SMALL-FOLD: minority n={int(cnt.min())} <=2; "
                           f"SMOTE needs >=3 samples. Use full 180-day data (real run).")
    kk = int(min(k, max(1, cnt.min() - 1)))
    sm = SMOTE(k_neighbors=kk, random_state=seed, sampling_strategy=strategy)
    out = sm.fit_resample(Xtr, ytr)
    return out[0], out[1]

def run(trainp: str, foldp: str, outdir: str, cfgp: str = "config/settings.yaml") -> Path:
    """Apply SMOTE per CV fold on TRAIN only; writes class-count log + sample balanced fold."""
    cfg = load_cfg(cfgp)
    sm = cfg.get("smote", {}) or {}
    k = int(sm.get("k_neighbors", 5))
    strat = sm.get("sampling_strategy", "auto")
    seed = int(cfg.get("seed", 42))
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    classes = list(cfg.get("aqi_classes", ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]))
    tr = pd.read_csv(trainp)
    fo = pd.read_csv(foldp)
    assert len(tr) == len(fo), f"train/folds row mismatch {len(tr)} vs {len(fo)}"
    folds = fo["fold"].values
    feats = feature_cols(tr, cfg)
    log: dict = {"features": feats, "per_fold": {}, "method": "SMOTE",
                 "target": tgt, "target_type": "classification (aqi_class) ONLY; never regression/val/test",
                 "note": "train-only, inside each CV fold; val/test untouched"}
    demX, demy = None, None
    for f in sorted(set(folds.tolist())):
        m = folds == f
        Xtr = tr.loc[m, feats].fillna(0).values
        ytr = encode_y(tr.loc[m, tgt], classes)
        before = dict(Counter(decode_y(ytr, classes)))
        Xb, yb = smote_fold(Xtr, ytr, k=k, seed=seed + int(f),
                            strategy=strat if strat != "auto" else "auto")
        after = dict(Counter(decode_y(yb, classes)))
        log["per_fold"][str(f)] = {"n_before": int(m.sum()), "n_after": int(len(yb)),
                                   "before": before, "after": after}
        print(f"fold {f}: {before} -> {after}")
        if demX is None:
            demX, demy = Xb, yb
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    lp = o / "smote_log.json"
    lp.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    dp = o / "train_balanced_demo_fold0.csv"
    ddf = pd.DataFrame(demX, columns=feats)
    ddf[tgt] = decode_y(demy, classes)
    ddf.to_csv(dp, index=False, encoding="utf-8")
    print(f"SMOTE train-only, {len(log['per_fold'])} folds; val/test untouched -> {lp}, demo {dp}")
    return lp

def main() -> None:
    """CLI: python src/training/balance.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --outdir results/reports"""
    ap = argparse.ArgumentParser(description="SMOTE train-only per fold (Step 5)")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--folds", default="data/processed/cv_folds.csv")
    ap.add_argument("--outdir", default="results/reports")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.train, a.folds, a.outdir, a.config)

if __name__ == "__main__":
    main()

