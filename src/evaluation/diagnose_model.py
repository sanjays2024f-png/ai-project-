"""Step 7: over/underfit diagnosis + correction + retrain hint.
STRICT: requires sklearn + yaml + matplotlib (plot optional with warning).
Saves learning curves + diagnosis.json + corrected config. Run:
  python src/evaluation/diagnose_model.py --train data/processed/train.csv --val data/processed/val.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.require import need
need("sklearn")
need("yaml")
import numpy as np
import pandas as pd

OVERFIT_GAP = 0.10  # F1 gap threshold per spec
UNDERFIT_TH = 0.50  # both low threshold

def load_cfg(p: str) -> dict:
    """Load YAML config (fails loudly)."""
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def decide(train_f1: float, val_f1: float) -> str:
    """Classify fit: overfit if gap>10%, underfit if both low, else good."""
    if train_f1 - val_f1 > OVERFIT_GAP:
        return "overfit"
    if train_f1 < UNDERFIT_TH and val_f1 < UNDERFIT_TH:
        return "underfit"
    return "good"

def recommend(status: str) -> dict:
    """Correction recipe per spec (overfit vs underfit)."""
    if status == "overfit":
        return {"dropout": 0.5, "weight_decay": 1e-3, "early_stopping_patience": 3,
                "hidden_dim": 32, "n_mesh_blocks": 1, "noise_augmentation": True,
                "label_smoothing": 0.1, "epochs": 30,
                "note": "dropout+wd+early-stop+fewer-params+noise+label-smooth"}
    if status == "underfit":
        return {"dropout": 0.1, "weight_decay": 1e-5, "early_stopping_patience": 10,
                "hidden_dim": 128, "n_mesh_blocks": 3, "noise_augmentation": False,
                "label_smoothing": 0.0, "epochs": 80,
                "note": "wider/deeper+more-epochs+less-reg"}
    return {"note": "good fit; no change"}

def quick_scores(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], tgt: str,
                 classes: list[str]) -> tuple[float, float, list[float], list[float]]:
    """Honest RF probe: 5-fold CV macro-F1 on train (never train-on-train) + val F1.

    Old code predicted Xtr[:n] with a model fit on Xtr[:n] -> memorisation (1.0).
    Now: cross_val_predict on each prefix (honest), final val via full-train fit.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.training.train import f1_acc
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    m = {c: i for i, c in enumerate(classes)}
    Xtr = tr[feats].fillna(0).values
    ytr = tr[tgt].map(m).fillna(-1).astype(int).values
    Xva = va[feats].fillna(0).values
    yva = va[tgt].map(m).fillna(-1).astype(int).values
    if (ytr < 0).any() or (yva < 0).any():
        raise RuntimeError("FAILED: unknown AQI label in diagnose probe")
    curve_tr: list[float] = []
    curve_va: list[float] = []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        n = max(30, int(len(Xtr) * frac))
        Xs, ys = Xtr[:n], ytr[:n]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            p_tr = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                                     Xs, ys, cv=cv)
        except ValueError as ex:
            raise RuntimeError(f"FAILED: CV probe needs >=3 classes per prefix: {ex}")
        m_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m_full.fit(Xs, ys)
        p_va = m_full.predict(Xva)
        f_tr, _ = f1_acc(ys, p_tr)
        f_va, _ = f1_acc(yva, p_va)
        curve_tr.append(round(f_tr, 4))
        curve_va.append(round(f_va, 4))
    return curve_tr[-1], curve_va[-1], curve_tr, curve_va

def run(trainp: str, valp: str, outdir: str, cfgp: str = "config/settings.yaml", metrics_subdir: str = "metrics", fig_subdir: str = "figures") -> Path:
    """Diagnose fit, save curves + diagnosis.json + corrected config."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.training.balance import feature_cols
    cfg = load_cfg(cfgp)
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    classes = list(cfg.get("aqi_classes", ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]))
    tr = pd.read_csv(trainp, encoding="utf-8")
    va = pd.read_csv(valp, encoding="utf-8") if Path(valp).exists() else tr.sample(frac=0.15, random_state=42)
    feats = [c for c in feature_cols(tr, cfg) if c != "provenance"]
    if "provenance" in tr.columns:
        assert (tr["provenance"] == "synthetic_generator_v1").all(), "diagnose expects synthetic smoke input"
    tr_f1, va_f1, c_tr, c_va = quick_scores(tr, va, feats, tgt, classes)
    status = decide(tr_f1, va_f1)
    rec = recommend(status)
    o = Path(outdir)
    mo, fo = o / metrics_subdir, o / fig_subdir
    mo.mkdir(parents=True, exist_ok=True); fo.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [20, 40, 60, 80, 100]
        plt.figure()
        plt.plot(xs, c_tr, marker="o", label="train F1")
        plt.plot(xs, c_va, marker="s", label="val F1")
        plt.xlabel("train data %")
        plt.ylabel("macro F1")
        plt.title(f"Learning curve ({status}): train={tr_f1:.3f} val={va_f1:.3f}")
        plt.legend()
        plt.savefig(fo / "learning_curve.png", dpi=100)
        plt.close()
    except ImportError as ex:
        print(f"WARNING: matplotlib missing -> curve not plotted ({ex}); install matplotlib==3.9.2")
    import sklearn as _sk, torch as _th, platform as _pf
    diag = {"train_f1": tr_f1, "val_f1": va_f1, "gap": round(tr_f1 - va_f1, 4),
            "status": status, "correction": rec, "curves": {"train": c_tr, "val": c_va},
            "model_identity": "RF probe with 3-fold cross_val_predict on train (honest; diagnosis only; paper model = Env-MeshNet pytorch-geomesh)",
            "data": {"n_train": int(len(tr)), "n_val": int(len(va))},
            "provenance": "synthetic_generator_v1 (smoke only)",
            "versions": {"python": _pf.python_version(), "sklearn": _sk.__version__,
                         "torch": _th.__version__, "pandas": pd.__version__},
            "warnings": ["train scores are CV-based (not train-on-train); synthetic disease labels are circular; Brier unavailable"]}
    dp = mo / "diagnosis.json"
    dp.write_text(json.dumps(diag, indent=1, ensure_ascii=False), encoding="utf-8")
    cp = Path("config") / "config_corrected.yaml"
    import yaml
    nc = json.loads(json.dumps(cfg))
    nc.setdefault("model", {}).update({k: v for k, v in rec.items() if k != "note"})
    cp.write_text(yaml.safe_dump(nc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"diagnose: train_F1={tr_f1:.3f} val_F1={va_f1:.3f} gap={tr_f1-va_f1:.3f} -> {status}")
    print(f"correction: {rec.get('note')}")
    print(f"-> {dp}, {fo/'learning_curve.png'}, {cp}")
    print("Retrain: python src/training/train.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --config config/config_corrected.yaml")
    return dp

def main() -> None:
    """CLI: python src/evaluation/diagnose_model.py --train ... --val ... --outdir results"""
    ap = argparse.ArgumentParser(description="Over/underfit diagnosis (Step 7)")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--val", default="data/processed/val.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.train, a.val, a.outdir, a.config)

if __name__ == "__main__":
    main()
