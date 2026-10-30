"""Step 2b: refresh data/external/stations.csv (OpenAQ + CPCB + data.gov.in).
Dedup rule: name similarity (rapidfuzz>=85) AND distance<300m -> same station.
Run daily from project root: python scripts/update_stations.py. Needs OPENAQ_API_KEY in .env.
CPCB has no open API -> keeps seed + merges data/raw/cpcb*.csv if present.
"""
from __future__ import annotations
import math, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.require import need
need("pandas")
import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a: object, **k: object) -> bool:  # type: ignore
        print("WARNING: python-dotenv missing -> .env not loaded; pip install python-dotenv==1.0.1")
        return False

SEED = Path("data/external/stations.csv")
RAW = Path("data/raw/cpcb")
def km(a: float, b: float, c: float, d: float) -> float:
    """Haversine km between (a,b) and (c,d)."""
    r = math.radians
    h = math.sin(r(c-a)/2)**2 + math.cos(r(a))*math.cos(r(c))*math.sin(r(d-b)/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(h))
def norm(s: str) -> str:
    """Lowercase, strip CPCB suffixes/spaces for fuzzy match."""
    s = s.lower()
    s = re.sub(r"[-,]?\s*cpcb\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()
def fetch_openaq(limit: int = 1000) -> pd.DataFrame:
    """Fetch OpenAQ v3 locations for India (bbox). Empty if no key/offline."""
    load_dotenv()
    key = os.getenv("OPENAQ_API_KEY", "")
    if not key:
        print("OPENAQ: no key -> skip"); return pd.DataFrame()
    import requests
    url = "https://api.openaq.org/v3/locations"
    h = {"X-API-Key": key}
    p = {"bbox": "68.0,6.0,97.5,37.5", "limit": 100, "page": 1}
    rows: list[dict] = []
    try:
        for pg in range(1, limit // 100 + 1):
            p["page"] = pg
            r = requests.get(url, headers=h, params=p, timeout=30)
            if r.status_code != 200: break
            js = r.json().get("results", [])
            if not js: break
            for it in js:
                co = it.get("coordinates") or {}
                rows.append({"station_id": f"OA-{it.get('id')}",
                 "name": it.get("name", ""), "district": it.get("locality", ""),
                 "state": "", "lat": co.get("latitude"), "lon": co.get("longitude"),
                 "source": "OpenAQ"})
    except Exception as ex:
        print(f"OPENAQ fetch failed ({ex}) -> skip")
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"])
def merge_dedup(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge with rule: fuzzy name>=85 AND dist<0.3km -> duplicate, keep base. Requires rapidfuzz."""
    from rapidfuzz import fuzz  # fails loudly if missing (no silent non-dedup merge)
    out = base.copy() if len(base) else pd.DataFrame(
        columns=["station_id","name","district","state","lat","lon","source"])
    for _, r in new.iterrows():
        dup = False
        for i, b in out.iterrows():
            if km(float(r.lat), float(r.lon), float(b.lat), float(b.lon)) > 0.3:
                continue
            if fuzz.ratio(norm(str(r["name"])), norm(str(b["name"]))) >= 85:
                dup = True; break
        if not dup:
            out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    return out.drop_duplicates(subset=["station_id"])
def main() -> None:
    """CLI: python scripts/update_stations.py [--limit 500]"""
    import argparse
    ap = argparse.ArgumentParser(description="Refresh station registry")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", type=str, default="data/external/stations.csv")
    a = ap.parse_args()
    base = pd.read_csv(SEED, encoding="utf-8") if SEED.exists() else pd.DataFrame()
    print(f"seed: {len(base)} stations")
    oa = fetch_openaq(limit=a.limit)
    print(f"openaq: {len(oa)} stations")
    merged = merge_dedup(base, oa)
    for f in sorted(RAW.glob("cpcb*.csv")):  # manual CPCB dumps (UTF-8)
        try:
            c = pd.read_csv(f, encoding="utf-8")
            merged = merge_dedup(merged, c)
            print(f"merged {f}: {len(c)} rows")
        except Exception as ex:
            print(f"skip {f}: {ex}")
    o = Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(o, index=False, encoding="utf-8")
    print(f"Wrote {len(merged)} stations -> {o}")
if __name__ == "__main__":
    main()

