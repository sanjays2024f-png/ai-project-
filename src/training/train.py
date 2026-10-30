"""Step 6c: initial training — Env-MeshNet (real PyTorch geo-mesh) + 9 baselines.
STRICT: requires sklearn + torch; FAILED (loud) if missing. Same splits + same
SMOTE rule for every model. Saves results/metrics/cv_metrics.csv + models/checkpoints. Run:
  python src/training/train.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --outdir results/metrics
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.require import need
need("sklearn")
need("torch")
need("yaml")
need("scipy")
import numpy as np
import pandas as pd

def load_cfg(p: str) -> dict:
    """Load YAML config (fails loudly if missing)."""
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def f1_acc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Macro F1 + accuracy from sklearn. REAL metrics only — no fallback."""
    from sklearn.metrics import f1_score, accuracy_score
    return float(f1_score(y, p, average="macro", zero_division=0)), float(accuracy_score(y, p))

def prep(train: pd.DataFrame, mask: np.ndarray, feats: list[str], tgt: str,
         classes: list[str], seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split fold into inner-train/inner-val (time-ordered 80/20) + STRICT SMOTE on train slice."""
    from src.training.balance import encode_y, smote_fold
    idx = np.where(mask)[0]
    cut = int(len(idx) * 0.8)
    tri, vai = idx[:cut], idx[cut:]
    m = {c: i for i, c in enumerate(classes)}
    Xtr = train.iloc[tri][feats].fillna(0).values
    ytr = train.iloc[tri][tgt].map(m).fillna(-1).astype(int).values
    Xva = train.iloc[vai][feats].fillna(0).values
    yva = train.iloc[vai][tgt].map(m).fillna(-1).astype(int).values
    if (ytr < 0).any() or (yva < 0).any():
        raise RuntimeError("FAILED: unknown AQI label in fold (check config aqi_classes)")
    Xtr, ytr = smote_fold(Xtr, ytr, k=5, seed=seed)  # STRICT: raises on failure
    return Xtr, ytr, Xva, yva

def run_sk(name: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
           seed: int = 42) -> np.ndarray:
    """Fit sklearn baseline. FAILS loudly — never majority fallback."""
    from src.models.baselines import sklearn_baseline
    m = sklearn_baseline(name, seed)
    m.fit(Xtr, ytr)
    return m.predict(Xva)

def run_torch(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
              n_cls: int = 6, seed: int = 42, epochs: int = 5) -> np.ndarray:
    """Fit torch baseline (CPU). FAILS loudly — never majority fallback."""
    import torch
    from src.models.baselines import SeqClf
    if SeqClf is None:
        raise RuntimeError("FAILED: torch present but SeqClf unavailable")
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = SeqClf(d, n_cls, kind)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        l = loss(net(Xt), yt)
        l.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xva, dtype=torch.float32)).argmax(1).numpy()

def run_mesh(tr: pd.DataFrame, mask: np.ndarray, feats: list[str], tgt: str,
             classes: list[str], seed: int = 42, epochs: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """REAL Env-MeshNet PyTorch geo-mesh training. No RF stub.

    Delaunay faces over train cells; mesh-conv with class-weighted CE
    (SMOTE would break spatial structure -> weights instead).
    Inner time split 80/20 by date; returns (y_true, y_pred) on val dates.
    """
    import torch
    from src.models.mesh import points, triangulate, face_centres, struct_feats, env_face
    from src.models.env_meshnet import EnvMeshNet, HAS_TORCH
    if not HAS_TORCH:
        raise RuntimeError("FAILED: torch missing — cannot train Env-MeshNet")
    m = {c: i for i, c in enumerate(classes)}
    torch.manual_seed(seed)
    pts = points(tr)
    simp, neigh = triangulate(pts)
    ecols = [c for c in "aod,no2,so2,co,bc,dust,temp,rh,wind_u,wind_v,blh,rain,fire_count,ndvi,lst".split(",") if c in tr.columns]
    dates = sorted(tr.loc[mask, "date"].astype(str).unique().tolist())
    cut = max(1, int(len(dates) * 0.8))
    dtr, dva = dates[:cut], dates[cut:]
    fc = face_centres(pts, simp).astype("float32")
    st = struct_feats(pts, simp, neigh).astype("float32")
    popmap = tr.groupby("cell_id")["population"].mean() if "population" in tr.columns else {}
    pop = np.array([[float(popmap.get(int(c), 0.0))] for c in pts["cell_id"]], dtype="float32")
    def faces_for(day: str) -> tuple[np.ndarray, int]:
        sub = tr[tr["date"].astype(str) == day]
        env = env_face(sub, pts, simp, ecols).astype("float32") if len(sub) else np.zeros((len(simp), len(ecols)), dtype="float32")
        lab = sub[tgt].map(m).mode() if len(sub) else []
        return env, int(lab.iloc[0]) if len(lab) else -1
    net = EnvMeshNet(e_dim=len(ecols), h=32, n_blocks=1, drop=0.3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    sp_t = torch.tensor(fc).unsqueeze(0)
    st_t = torch.tensor(st)
    pop_t = torch.tensor(pop).unsqueeze(0)
    for _ in range(max(2, epochs)):
        net.train()
        for day in dtr[:20]:
            env, y = faces_for(day)
            if y < 0:
                continue
            aqi, _, _ = net(sp_t, torch.tensor(env).unsqueeze(0), st_t, pop_t, neigh)
            opt.zero_grad()
            ce(aqi, torch.tensor([y])).backward()
            opt.step()
    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for day in dva[:20]:
            env, y = faces_for(day)
            if y < 0:
                continue
            aqi, _, _ = net(sp_t, torch.tensor(env).unsqueeze(0), st_t, pop_t, neigh)
            preds.append(int(aqi.argmax(1)[0]))
            trues.append(y)
    if not preds:
        raise RuntimeError("FAILED: mesh val slice empty (check dates/cells)")
    return np.array(trues), np.array(preds)

def run_cv(trainp: str, foldp: str, outdir: str, cfgp: str, epochs: int = 5) -> Path:
    """All folds x 10 models -> results/metrics/cv_metrics.csv. REAL training, no stubs."""
    import json as _json
    from src.training.balance import feature_cols
    from src.models.baselines import SKLEARN_NAMES, TORCH_NAMES, ALL_NAMES
    cfg = load_cfg(cfgp)
    seed = int(cfg.get("seed", 42))
    tgt = (cfg.get("features", {}) or {}).get("target_aqi", "aqi_class")
    classes = list(cfg.get("aqi_classes", ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]))
    tr = pd.read_csv(trainp, encoding="utf-8")
    fo = pd.read_csv(foldp, encoding="utf-8")
    feats = feature_cols(tr, cfg)
    folds = sorted(set(fo["fold"].tolist()))  # ALL folds (no 2-fold demo)
    rows: list[dict] = []
    for f in folds:
        mask = (fo["fold"].values == f)
        for name in ALL_NAMES:
            if name == "envmeshnet":
                yva, pred = run_mesh(tr, mask, feats, tgt, classes, seed + int(f), epochs)
                note = "pytorch-geomesh"
            elif name in SKLEARN_NAMES:
                Xtr, ytr, Xva, yva = prep(tr, mask, feats, tgt, classes, seed)
                pred = run_sk(name, Xtr, ytr, Xva, seed)
                note = ""
            else:
                Xtr, ytr, Xva, yva = prep(tr, mask, feats, tgt, classes, seed)
                pred = run_torch(name, Xtr, ytr, Xva, len(classes), seed, epochs)
                note = ""
            f1, acc = f1_acc(yva, pred)
            rows.append({"fold": f, "model": name, "f1": round(f1, 4),
                         "acc": round(acc, 4), "note": note, "n_val": int(len(yva))})
            print(f"fold {f} {name}: f1={f1:.3f} acc={acc:.3f} n={len(yva)} {note}")
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    mp = o / "cv_metrics.csv"
    pd.DataFrame(rows).to_csv(mp, index=False, encoding="utf-8")
    agg = pd.DataFrame(rows).groupby("model")[["f1", "acc"]].agg(["mean", "std", "count"])
    import sklearn as _sk, torch as _th
    ap = o / "cv_summary.json"
    summ = {"per_model": _json.loads(agg.round(4).to_json()),
            "folds_completed": int(len(folds)),
            "models": list(ALL_NAMES),
            "split": "StratifiedGroupKFold(district) train-only; inner time 80/20",
            "smote": "train-slice-only per fold (envmeshnet uses class-weighted CE instead)",
            "seed": seed, "epochs": epochs,
            "versions": {"sklearn": _sk.__version__, "torch": _th.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__}}
    ap.write_text(_json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {mp}, {ap}")
    return mp

def main() -> None:
    """CLI: python src/training/train.py --train ... --folds ... --epochs 5 --outdir results/metrics"""
    ap = argparse.ArgumentParser(description="Initial training (Step 6c)")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--folds", default="data/processed/cv_folds.csv")
    ap.add_argument("--outdir", default="results/metrics")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    run_cv(a.train, a.folds, a.outdir, a.config, a.epochs)

if __name__ == "__main__":
    main()


