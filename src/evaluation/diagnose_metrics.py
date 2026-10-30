"""Measured metrics: train/val/test + CV5 + curves + CMs (venv python)."""
import json
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
AQI = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
EXCL = {"date", "cell_id", "district", "state", "zone", "pm25", "aqi_class",
        "disease_risk", "pm10", "aqi_index", "fold", "provenance"}

def main():
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    tr = pd.read_csv(PROC / "train.csv", encoding="utf-8")
    va = pd.read_csv(PROC / "val.csv", encoding="utf-8")
    te = pd.read_csv(PROC / "test.csv", encoding="utf-8")
    feats = [c for c in tr.columns if c not in EXCL and pd.api.types.is_numeric_dtype(tr[c])]
    print(f"FEATS n={len(feats)}")
    m = {c: i for i, c in enumerate(AQI)}
    Xtr = tr[feats].fillna(0).values
    ytr = tr["aqi_class"].map(m).fillna(-1).astype(int).values
    Xva = va[feats].fillna(0).values
    yva = va["aqi_class"].map(m).fillna(-1).astype(int).values
    Xte = te[feats].fillna(0).values
    yte = te["aqi_class"].map(m).fillna(-1).astype(int).values
    assert (ytr >= 0).all() and (yva >= 0).all() and (yte >= 0).all(), "unknown label"
    print(f"TEMPORAL train {tr['date'].min()}->{tr['date'].max()} val {va['date'].min()}->{va['date'].max()} test {te['date'].min()}->{te['date'].max()}")
    k1 = set(zip(tr["date"].astype(str), tr["cell_id"])); k2 = set(zip(te["date"].astype(str), te["cell_id"]))
    print("OVERLAP_TR_TE", len(k1 & k2))
    print("DUP_PROC", int(pd.read_csv(PROC / "india_proc.csv", encoding="utf-8").duplicated().sum()))
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    out = {"model_identity": "RF probe (sklearn, diagnosis only). Paper model = Env-MeshNet pytorch-geomesh.",
           "provenance": "synthetic_generator_v1 (SMOKE ONLY)", "labels": AQI, "n_feats": len(feats)}
    for nm, Xx, yx in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        p = clf.predict(Xx)
        acc = float(accuracy_score(yx, p))
        prec = float(precision_score(yx, p, average="macro", zero_division=0))
        rec = float(recall_score(yx, p, average="macro", zero_division=0))
        f1 = float(f1_score(yx, p, average="macro", zero_division=0))
        cm = confusion_matrix(yx, p, labels=list(range(len(AQI)))).tolist()
        print(f"{nm}: acc={acc:.4f} prec_macro={prec:.4f} rec_macro={rec:.4f} f1_macro={f1:.4f} n={len(yx)}")
        print(f"  cm={cm}")
        rep = classification_report(yx, p, labels=list(range(len(AQI))), target_names=AQI, zero_division=0, output_dict=True)
        out[nm] = {"n": int(len(yx)), "acc": round(acc, 4), "prec_macro": round(prec, 4),
                   "rec_macro": round(rec, 4), "f1_macro": round(f1, 4), "cm": cm,
                   "per_class": {c: {"prec": round(rep[c]["precision"], 4), "rec": round(rep[c]["recall"], 4),
                                     "f1": round(rep[c]["f1-score"], 4), "support": int(rep[c]["support"])} for c in AQI}}
    return out, (Xtr, ytr, Xva, yva)

def part2(out, data):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Xtr, ytr, Xva, yva = data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), Xtr, ytr, cv=cv)
    out["cv5"] = {"acc": round(float(accuracy_score(ytr, p_cv)), 4),
                  "f1_macro": round(float(f1_score(ytr, p_cv, average="macro", zero_division=0)), 4)}
    print("CV5", out["cv5"])
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    c_tr, c_va = [], []
    for frac in fracs:
        n = max(30, int(len(Xtr) * frac))
        Xs, ys = Xtr[:n], ytr[:n]
        cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            ptr = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), Xs, ys, cv=cv3)
            c_tr.append(round(float(f1_score(ys, ptr, average="macro", zero_division=0)), 4))
        except Exception as ex:
            c_tr.append(None)
            print("curve-train fail", ex)
        mf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        mf.fit(Xs, ys)
        pv = mf.predict(Xva)
        c_va.append(round(float(f1_score(yva, pv, average="macro", zero_division=0)), 4))
    print("CURVES", c_tr, c_va)
    out["curves"] = {"x_pct": [20, 40, 60, 80, 100], "train_f1_cv3": c_tr, "val_f1": c_va}
    gap = round(out["train"]["f1_macro"] - out["val"]["f1_macro"], 4)
    status = "overfit" if (out["train"]["f1_macro"] - out["val"]["f1_macro"]) > 0.10 else ("underfit" if out["train"]["f1_macro"] < 0.50 and out["val"]["f1_macro"] < 0.50 else "good")
    out["gap_train_val"] = gap
    out["status"] = status
    print("STATUS", status, "GAP", gap)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [20, 40, 60, 80, 100]
    plt.figure()
    plt.plot(xs, [v if v is not None else float("nan") for v in c_tr], marker="o", label="train F1 (3-fold CV)")
    plt.plot(xs, c_va, marker="s", label="val F1")
    plt.xlabel("train data %")
    plt.ylabel("macro F1")
    plt.title(f"Learning curve ({status}): train={out['train']['f1_macro']:.3f} val={out['val']['f1_macro']:.3f}")
    plt.legend()
    lp = FIG / "learning_curve_measured.png"
    plt.savefig(lp, dpi=100)
    plt.close()
    print("PLOT", lp)
    out["learning_curve_png"] = lp.name
    (RES / "model_diagnosis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WROTE model_diagnosis.json")
    return out

if __name__ == "__main__":
    o, d = main()
    part2(o, d)
