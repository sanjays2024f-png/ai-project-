"""Balance + leakage measurement (venv python). No yaml needed."""
import pandas as pd
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "synthetic" / "india_dataset.csv"
RES = ROOT / "results" / "reports"
AQI = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BPS = [30, 60, 90, 120, 250]
PROC = ROOT / "data" / "processed"

def ctable(s, name):
    vc = s.value_counts(dropna=False)
    n = len(s)
    return [{"target": name, "class": str(k), "count": int(v),
             "percent": round(100 * int(v) / n, 2)} for k, v in vc.items()]

def main():
    df = pd.read_csv(DATA, encoding="utf-8")
    rows = []
    rows += ctable(df["aqi_class"], "aqi_class")
    rows += ctable(df["disease_risk"], "disease_risk")
    def binpm(x):
        for bp, c in zip(BPS, AQI):
            if x <= bp:
                return c
        return AQI[-1]
    rows += ctable(df["pm25"].apply(binpm), "pm25_binned")
    bal = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    bal.to_csv(RES / "dataset_balance_report.csv", index=False, encoding="utf-8")
    print(f"TOTAL rows={len(df)} cols={len(df.columns)}")
    print(bal.to_string(index=False))
    print("DUP_ROWS", int(df.duplicated().sum()))
    print("DUP_KEYS", int(df.duplicated(subset=["date", "cell_id"]).sum()))
    banned = [c for c in df.columns if c.lower() in ("pm10", "aqi_index") or c.lower().startswith("aqi_")]
    print("BANNED", banned if banned else "none")
    num = df.select_dtypes("number")
    corr = num.corr(numeric_only=True)["pm25"].abs().drop("pm25", errors="ignore")
    legit = {f"pm25_lag{L}" for L in (1, 3, 7)} | {f"pm25_roll{w}" for w in (3, 7)}
    high = {c: round(float(v), 3) for c, v in corr.items()
            if c not in legit and not c.endswith("_n") and pd.notna(v) and v > 0.95}
    print("CORR_GT_95", high if high else "none")
    print("PM25 mean=%.2f median=%.2f min=%.2f max=%.2f std=%.2f" % (
        df["pm25"].mean(), df["pm25"].median(), df["pm25"].min(), df["pm25"].max(), df["pm25"].std()))
    print("PROV", df["provenance"].unique().tolist() if "provenance" in df.columns else "nocol")
    for p in [PROC / "train.csv", PROC / "val.csv", PROC / "test.csv"]:
        if p.exists():
            d = pd.read_csv(p, encoding="utf-8", usecols=["date"])
            print(p.name, len(d), d["date"].min(), "->", d["date"].max())

if __name__ == "__main__":
    main()
