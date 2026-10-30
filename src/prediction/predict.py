"""Batch predict on processed test (RF probe) + write results/predictions/."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
def main() -> None:
    root = Path(__file__).resolve().parents[2]
    tr = pd.read_csv(root / "data" / "processed" / "train.csv", encoding="utf-8")
    te = pd.read_csv(root / "data" / "processed" / "test.csv", encoding="utf-8")
    cfg_excl = {"date","cell_id","district","state","zone","pm25","aqi_class",
                "disease_risk","pm10","aqi_index","fold","provenance"}
    feats = [c for c in tr.columns if c not in cfg_excl and pd.api.types.is_numeric_dtype(tr[c])]
    m = {c: i for i, c in enumerate(["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"])}
    inv = {i: c for c, i in m.items()}
    Xtr = tr[feats].fillna(0).values
    ytr = tr["aqi_class"].map(m).fillna(-1).astype(int).values
    Xte = te[feats].fillna(0).values
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = [inv[int(i)] for i in clf.predict(Xte)]
    out = root / "results" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    te[["date","cell_id","district","aqi_class"]].assign(pred_aqi=pred).to_csv(
        out / "pred_test.csv", index=False, encoding="utf-8")
    print(f"predictions -> {out / 'pred_test.csv'} (RF probe, synthetic smoke)")
if __name__ == "__main__":
    main()
