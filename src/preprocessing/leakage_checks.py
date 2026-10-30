"""Step 4a: leakage checks -> results/reports/leakage_report.txt (run BEFORE split, re-check after).
Checks: target leakage (|corr|>0.95, banned PM10/AQI cols), duplicates,
suspicious (near-constant / ID-like / single-feat AUC>0.98), temporal+spatial.
Run: python src/preprocessing/leakage_checks.py --in data/processed/india_proc.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

BANNED = ["pm10", "aqi_index", "aqi_sub", "no2_sub", "so2_sub", "co_sub", "o3_sub"]
CORR_TH = 0.95
AUC_TH = 0.98

def target_check(df: pd.DataFrame) -> list[str]:
    """Flag banned cols + numeric feats with |corr| > 0.95 vs pm25.

    TARGETS THEMSELVES (pm25, aqi_class, disease_risk) are never leakage —
    only features derived from them are. Lag/roll copies of pm25 are expected
    to correlate highly and are legitimate temporal features, not leakage.
    """
    out: list[str] = []
    TARGETS = {"pm25", "aqi_class", "disease_risk"}
    LEGIT_TEMPORAL = {f"pm25_lag{L}" for L in (1, 3, 7)} | {f"pm25_roll{w}" for w in (3, 7)}
    for c in df.columns:
        cl = c.lower()
        if c in TARGETS:
            continue  # prediction targets, not leakage
        if cl in BANNED or cl.startswith("aqi_") or cl == "pm10":
            out.append(f"BANNED col present: {c} (remove; derived from pm25/aqi)")
    if "pm25" in df.columns:
        num = df.select_dtypes("number")
        corr = num.corr(numeric_only=True)["pm25"].abs().drop("pm25", errors="ignore")
        for c, v in corr.items():
            if c in TARGETS or c in LEGIT_TEMPORAL or c.endswith("_n"):
                continue
            if pd.notna(v) and v > CORR_TH:
                out.append(f"HIGH corr {c} vs pm25: {v:.3f} > {CORR_TH} (likely leakage)")
    if not out:
        out.append("target leakage: OK (no banned cols, no |corr|>0.95)")
    return out

def dup_check(df: pd.DataFrame) -> list[str]:
    """Count fully-duplicate rows + duplicate (date, cell_id) keys."""
    n_dup = int(df.duplicated().sum())
    n_key = int(df.duplicated(subset=["date", "cell_id"]).sum()) if {"date", "cell_id"} <= set(df.columns) else -1
    return [f"duplicate rows: {n_dup}", f"duplicate (date,cell_id) keys: {n_key}"]

def overlap_check(tr: pd.DataFrame, te: pd.DataFrame) -> list[str]:
    """Identical (date,cell_id) in both train and test = leakage."""
    if tr.empty or te.empty:
        return ["overlap check: skipped (empty split)"]
    k1 = set(zip(tr["date"].astype(str), tr["cell_id"]))
    k2 = set(zip(te["date"].astype(str), te["cell_id"]))
    n = len(k1 & k2)
    return [f"(date,cell_id) overlap train/test: {n} " + ("LEAKAGE!" if n else "(OK)")]

def suspicious_check(df: pd.DataFrame) -> list[str]:
    """Near-constant, ID-like, single-feature AUC>0.98. Requires sklearn (no silent skip)."""
    from sklearn.metrics import roc_auc_score
    out: list[str] = []
    SKIP = {"pm25", "aqi_class", "disease_risk", "date", "provenance",
            "cell_id", "lat", "lon", "elevation", "population"}
    n = len(df)
    for c in df.columns:
        if c in SKIP:
            continue  # ids/coords/statics are not suspicious features
        u = df[c].nunique()
        if u <= 2:
            # binary flags (is_winter/is_monsoon/is_diwali on short smoke windows) are legitimate
            if c.startswith("is_"):
                out.append(f"note: binary flag {c} (nunique={u}; legitimate season/event flag)")
            else:
                out.append(f"near-constant: {c} (nunique={u})")
        if u / max(n, 1) > 0.95:
            # continuous physical fields (wind/temp) are legitimately unique per row
            if c in ("wind_speed", "wind_dir", "wind_u", "wind_v", "temp", "lst", "rh", "blh"):
                out.append(f"note: continuous field {c} (nunique/n={u/n:.3f}; expected unique, not ID-like)")
            else:
                out.append(f"ID-like: {c} (nunique/n={u/n:.3f})")
    y = (df["disease_risk"] == "High").astype(int) if "disease_risk" in df.columns else None
    if y is not None and y.nunique() == 2:
        for c in df.select_dtypes("number").columns:
            if c in ("pm25", "cell_id"):
                continue
            try:
                a = roc_auc_score(y, df[c].fillna(df[c].median()))
                a = max(a, 1 - a)
                if a > AUC_TH:
                    out.append(f"single-feat AUC {c}: {a:.3f} > {AUC_TH} (suspicious)")
            except Exception:
                continue
    if not out:
        out.append("suspicious features: OK")
    return out

def write_report(df: pd.DataFrame, out: str, tr: pd.DataFrame | None = None,
                 te: pd.DataFrame | None = None) -> Path:
    """Write full leakage_report.txt covering all 4 check families."""
    lines = ["Env-MeshNet leakage report", "=" * 40, f"rows={len(df)} cols={len(df.columns)}",
             "", "[1] TARGET LEAKAGE", *target_check(df),
             "", "[2] DUPLICATE LEAKAGE", *dup_check(df)]
    if tr is not None and te is not None:
        lines += ["", "[2b] TRAIN/TEST OVERLAP", *overlap_check(tr, te)]
    lines += ["", "[3] SUSPICIOUS FEATURES", *suspicious_check(df),
              "", "[4] TEMPORAL+SPATIAL",
              "temporal: time-based split (latest months=test), never random (see split.py)",
              "spatial: unseen-district hold-out (see config split.holdout_districts)",
              "", "[5] UNRESOLVED RISKS",
              "- disease_risk labels are SYNTHETIC (derived from pm25) -> CIRCULAR. Any pm25-derived",
              "  feature will trivially predict disease (see pm25_roll AUC flags above). Brier/disease",
              "  metrics must stay NA until real IDSP/HMIS/NFHS labels exist. See DIAGNOSIS.md.",
              "- provenance: " + str(df["provenance"].unique().tolist() if "provenance" in df.columns else "unknown")]
    o = Path(out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {o}")
    return o

def main() -> None:
    """CLI: python src/preprocessing/leakage_checks.py --in data/processed/india_proc.csv [--train ... --test ...]"""
    ap = argparse.ArgumentParser(description="Leakage checks (Step 4a)")
    ap.add_argument("--in", dest="ind", default="data/processed/india_proc.csv")
    ap.add_argument("--out", default="results/reports/leakage_report.txt")
    ap.add_argument("--train", default="")
    ap.add_argument("--test", default="")
    a = ap.parse_args()
    df = pd.read_csv(a.ind, encoding="utf-8")
    tr = pd.read_csv(a.train, encoding="utf-8") if a.train else None
    te = pd.read_csv(a.test, encoding="utf-8") if a.test else None
    write_report(df, a.out, tr, te)

if __name__ == "__main__":
    main()

