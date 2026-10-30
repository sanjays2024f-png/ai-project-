"""Finalize model_diagnosis.json: merge measured balance/leakage/pipeline notes."""
import json
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
REP = ROOT / "results" / "reports"
def main():
    d = json.loads((MET / "model_diagnosis.json").read_text(encoding="utf-8"))
    bal = pd.read_csv(REP / "dataset_balance_report.csv", encoding="utf-8")
    summ = {}
    for tgt in bal["target"].unique():
        sub = bal[bal["target"] == tgt].sort_values("count")
        mn, mx = sub.iloc[0], sub.iloc[-1]
        summ[tgt] = {"minority": {"class": str(mn["class"]), "count": int(mn["count"]), "pct": float(mn["percent"])},
                     "majority": {"class": str(mx["class"]), "count": int(mx["count"]), "pct": float(mx["percent"])},
                     "IR": round(float(mx["count"] / mn["count"]), 2)}
    d["balance_summary"] = summ
    d["balance_verdict"] = "IMBALANCED: aqi_class IR=32.8 (Good 1.64pct vs Very Poor 53.91pct); disease_risk IR=11.1 synthetic circular"
    d["fit_honest"] = {"rule": "overfit if honest-train minus val gap>0.10; underfit if both<0.50; else good",
        "honest_train_f1_cv3": 0.6231, "val_f1": 0.6228, "gap": 0.0003, "status": "good",
        "note": "Honest numbers from results/diagnosis.json (3-fold cross_val_predict; RF probe synthetic smoke). This file's train=1.0 is train-on-train memorization artifact - DO NOT USE."}
    d["cv_note"] = "Repo results/cv_metrics.csv is STALE/INVALID (all 0.0, 2 folds, mesh-stub). Prior live log shows real 10-fold crashed at fold5 xgb single-class fold - honest loud failure from imbalance plus small grouped folds. Rerun train.py after xgb fold fix."
    d["temporal"] = {"train": "2024-08-01 to 2024-12-04", "val": "2024-12-05 to 2024-12-31", "test": "2025-01-01 to 2025-01-27", "overlap": 0}
    d["leakage_measured"] = {"dups": 0, "dup_keys": 0, "banned": "none", "corr95": "none (excl legit lags/rolls/_n)", "disease": "CIRCULAR confirmed build_csv.py:82; Brier NA until real labels", "holdout": "Leh/Kavaratti/Itanagar"}
    d["pipeline_code"] = {"smote": "train-only per-fold; graph uses class-weighted CE (no SMOTE on graph) - OK", "norm": "PARTIAL: pseudo-70pct slice fit; production must split-then-fit on train.csv", "feats": "train-derived numeric excl targets/ids - OK"}
    d["recommendations"] = ["Keep train-only SMOTE for baselines; class-weighted CE for graph.", "Fit=good on synthetic probe; no arch change.", "Fix preprocess to split-then-fit norm on train.csv.", "Fix xgb single-class fold crash before full CV.", "All results synthetic_generator_v1 smoke-only, not real-world."]
    (MET / "model_diagnosis.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("FINALIZED keys:", sorted(d.keys()))
if __name__ == "__main__":
    main()
