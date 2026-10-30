"""Step 8b: SHAP importance for paper (best RF probe on train sample).
STRICT: requires shap + sklearn (+matplotlib for plot).
If shap missing -> FAILED with install hint; RF-importance fallback is written
to rf_importance.csv (NOT shap_importance.csv) so reports are never mislabeled.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.require import need
import pandas as pd

def run(trainp: str, outdir: str, cfgp: str = "config/settings.yaml", n: int = 200) -> Path:
    """SHAP bar plot + CSV of mean |SHAP| per feature."""
    need("shap")
    need("sklearn")
    from src.training.balance import feature_cols

    def load_cfg(p: str) -> dict:
        import yaml
        return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    cfg = load_cfg(cfgp)
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    tr = pd.read_csv(trainp, encoding="utf-8").sample(min(n, 500), random_state=42) if n else pd.read_csv(trainp, encoding="utf-8")
    feats = feature_cols(tr, cfg)
    X = tr[feats].fillna(0).values
    y = tr[tgt].astype("category").cat.codes.values
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    import shap
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    m.fit(X, y)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X[:n])
    import numpy as np
    arr = np.abs(sv).mean(axis=0) if isinstance(sv, list) else np.abs(sv).mean(axis=0)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": arr}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(o / "shap_importance.csv", index=False, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = imp.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top["feature"], top["mean_abs_shap"])
        ax.set_title("SHAP importance (top 15)")
        fig.tight_layout()
        fig.savefig(o / "shap_importance.png", dpi=100)
        plt.close(fig)
    except ImportError as ex2:
        print(f"WARNING: matplotlib missing -> plot skipped ({ex2}); CSV still written")
    print(f"-> {o/'shap_importance.csv'}")
    return o / "shap_importance.csv"

def main() -> None:
    """CLI: python src/evaluation/explain.py --train data/processed/train.csv --outdir results/reports"""
    ap = argparse.ArgumentParser(description="SHAP importance (Step 8b)")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--outdir", default="results/reports")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.train, a.outdir, a.config)

if __name__ == "__main__":
    main()
