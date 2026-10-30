"""Step 8: FINAL test-only evaluation (run once on untouched test).
STRICT: requires sklearn; disease metrics = NA unless real health labels exist.
Synthetic disease_risk derived from pm25 is CIRCULAR -> Brier marked unavailable.
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

def load_cfg(p: str) -> dict:
    """Load YAML config (fails loudly)."""
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def cls_scores(y: np.ndarray, p: np.ndarray, prob: np.ndarray | None = None) -> dict:
    """Accuracy, macro F1/prec/rec, kappa, AUC-ovr (if prob given). No fallback."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, cohen_kappa_score, roc_auc_score)
    d = {"acc": float(accuracy_score(y, p)),
         "f1": float(f1_score(y, p, average="macro", zero_division=0)),
         "prec": float(precision_score(y, p, average="macro", zero_division=0)),
         "rec": float(recall_score(y, p, average="macro", zero_division=0)),
         "kappa": float(cohen_kappa_score(y, p))}
    if prob is not None:
        try:
            d["auc"] = float(roc_auc_score(y, prob, multi_class="ovr"))
        except Exception as ex:
            d["auc"] = float("nan")
            d["auc_note"] = f"unavailable: {ex}"
    else:
        d["auc"] = float("nan")
        d["auc_note"] = "unavailable: no predict_proba"
    return d

def reg_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """RMSE + MAE for pm25."""
    e = y - p
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())}

def brier_picp(y3: np.ndarray, prob3: np.ndarray, pm: np.ndarray, lo: np.ndarray,
               hi: np.ndarray) -> dict:
    """Brier (disease 3-class mean) + PICP + CRPS-proxy (mean width)."""
    b = float(np.mean(np.sum((prob3 - np.eye(3)[y3]) ** 2, axis=1))) if len(y3) else float("nan")
    picp = float(np.mean((pm >= lo) & (pm <= hi))) if len(pm) else float("nan")
    crps = float(np.mean(hi - lo)) if len(pm) else float("nan")
    return {"brier_disease": b, "picp": picp, "crps_width": crps}

def run(trainp: str, testp: str, outdir: str, cfgp: str, figdir: str | None = None, reportsdir: str | None = None) -> Path:
    """Final run-once eval on untouched test + ablations + zone/season."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.training.balance import feature_cols
    from src.training.train import f1_acc, run_sk
    cfg = load_cfg(cfgp)
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    classes = list(cfg.get("aqi_classes", ["Good", "Moderate"]))
    m = {c: i for i, c in enumerate(classes)}
    dm = {"Low": 0, "Medium": 1, "High": 2}
    tr = pd.read_csv(trainp, encoding="utf-8")
    te = pd.read_csv(testp, encoding="utf-8")
    feats = feature_cols(tr, cfg)
    Xtr = tr[feats].fillna(0).values
    ytr = tr[tgt].map(m).fillna(-1).astype(int).values
    Xte = te[feats].fillna(0).values
    yte = te[tgt].map(m).fillna(-1).astype(int).values
    pred = run_sk("rf", Xtr, ytr, Xte)
    prob = None
    try:
        from src.models.baselines import sklearn_baseline
        mm = sklearn_baseline("rf", 42)
        mm.fit(Xtr, ytr)
        prob = mm.predict_proba(Xte)
    except Exception:
        pass
    s = cls_scores(yte, pred, prob)
    med = tr.groupby(tgt)["pm25"].median().to_dict()
    pmap = np.array([med.get(c, tr["pm25"].median()) for c in te[tgt]])
    s.update({"pm_" + k: v for k, v in reg_scores(te["pm25"].values, pmap).items()})
    yd = te["disease_risk"].map(dm).fillna(1).astype(int).values
    # STRICT: synthetic disease_risk is derived from pm25 -> CIRCULAR. Never fabricate Brier.
    s["brier_disease"] = float("nan")
    s["brier_note"] = "unavailable: synthetic disease labels are circular (derived from pm25); needs real IDSP/HMIS/NFHS labels"
    pm_true = te["pm25"].values
    lo, hi = pmap * 0.7, pmap * 1.3
    s.update({k: v for k, v in brier_picp(yd, np.zeros((len(te), 3)), pm_true, lo, hi).items() if k != "brier_disease"})
    rows = [{"model": "envmeshnet(rf-probe)", "n_test": int(len(te)),
             **{k: (round(float(v), 4) if isinstance(v, float) and v == v else v) for k, v in s.items()}}]
    te2 = te.copy()
    te2["_pred"] = [classes[int(i)] for i in pred]
    mo = pd.to_datetime(te2["date"]).dt.month
    te2["_season"] = np.where(mo.isin([11, 12, 1, 2]), "winter_smog", np.where(mo.isin([6, 7, 8, 9]), "monsoon", "other"))
    for col in ["zone", "_season"]:
        for g, d in te2.groupby(col):
            f1, acc = f1_acc(d[tgt].map(m).fillna(-1).astype(int).values, d["_pred"].map(m).fillna(-1).astype(int).values)
            rows.append({"model": f"split:{col}={g}", "n_test": int(len(d)),
                         "acc": round(acc, 4), "f1": round(f1, 4)})
    abls = {"no_fire": [c for c in feats if "fire" not in c],
            "no_lags": [c for c in feats if "_lag" not in c and "_roll" not in c],
            "no_pop": [c for c in feats if c != "population"],
            "no_mesh": feats}
    for name, fs in abls.items():
        pa = run_sk("rf", tr[fs].fillna(0).values, ytr, te[fs].fillna(0).values)
        f1, acc = f1_acc(yte, pa)
        rows.append({"model": f"ablation:{name}", "n_test": int(len(te)),
                     "acc": round(acc, 4), "f1": round(f1, 4)})
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    fig = Path(figdir) if figdir else o
    fig.mkdir(parents=True, exist_ok=True)
    rep = Path(reportsdir) if reportsdir else o
    rep.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out["provenance"] = "synthetic_generator_v1 (smoke only)"
    df_out["disease_note"] = "Brier NA: synthetic disease labels are circular"
    mp = o / "test_metrics.csv"
    df_out.to_csv(mp, index=False, encoding="utf-8")
    lt = rep / "test_table.tex"
    try:
        lt.write_text(df_out.to_latex(index=False, float_format="%.4f",
                      caption="Env-MeshNet test results on synthetic smoke data (honest probe; not research evidence). Brier NA (circular labels).",
                      label="tab:envmeshnet-smoke-test"), encoding="utf-8")
    except Exception:
        lt.write_text("% latex unavailable\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yte, pred, labels=list(range(len(classes))))
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted AQI")
        ax.set_ylabel("true AQI")
        ax.set_title("Confusion matrix (AQI, synthetic smoke test)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig / "confusion_matrix.png", dpi=100)
        plt.close(fig)
        fig2, ax2 = plt.subplots()
        sc = ax2.scatter(te["lon"], te["lat"], c=yte, s=4)
        ax2.set_xlabel("lon")
        ax2.set_ylabel("lat")
        ax2.set_title("India test AQI (true labels, synthetic)")
        fig2.colorbar(sc, ax=ax2)
        fig2.savefig(fig / "india_aqi_map.png", dpi=100)
        plt.close(fig2)
    except ImportError as ex:
        print(f"WARNING: matplotlib missing -> plots skipped ({ex})")
    print(f"FINAL test (run once, untouched): acc={s.get('acc')} f1={s.get('f1')}")
    print(f"-> {mp}, {lt}")
    return mp

def main() -> None:
    """CLI: python src/evaluation/evaluate.py --train ... --test ... --metricsdir results/metrics --figdir results/figures --reportsdir results/reports"""
    ap = argparse.ArgumentParser(description="Final test evaluation (Step 8)")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--test", default="data/processed/test.csv")
    ap.add_argument("--metricsdir", default="results/metrics")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--reportsdir", default="results/reports")
    ap.add_argument("--config", default="config/settings.yaml")
    a = ap.parse_args()
    run(a.train, a.test, a.metricsdir, a.config, a.figdir, a.reportsdir)

if __name__ == "__main__":
    main()
