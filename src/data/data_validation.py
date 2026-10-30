"""Validate dataset schema + provenance. Fails loudly on mismatch."""
from __future__ import annotations
import pandas as pd
EXPECTED = ["date","cell_id","lat","lon","district","state","zone","aod","no2",
 "so2","co","bc","dust","temp","rh","wind_u","wind_v","blh","rain","fire_count",
 "ndvi","lst","elevation","population","pm25","aqi_class","disease_risk","provenance"]
def validate(df: pd.DataFrame, expect_synthetic: bool = True) -> None:
    """Raise RuntimeError on schema/provenance violation."""
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise RuntimeError(f"FAILED: schema missing columns {missing}")
    if "provenance" in df.columns and expect_synthetic:
        vals = set(df["provenance"].unique().tolist())
        if vals != {"synthetic_generator_v1"}:
            raise RuntimeError(f"FAILED: expected synthetic provenance, got {vals}")
def main() -> None:
    from src.data.load_dataset import load_raw_synthetic
    df = load_raw_synthetic()
    validate(df)
    print(f"VALID rows={len(df)} provenance=synthetic_generator_v1")
if __name__ == "__main__":
    main()
