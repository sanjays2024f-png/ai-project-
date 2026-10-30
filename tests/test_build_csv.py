"""Tests for Step 1 synthetic generator (SMOKE ONLY — never research evidence)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate, aqi

EXPECTED_COLS = ["date","cell_id","lat","lon","district","state","zone","aod","no2",
"so2","co","bc","dust","temp","rh","wind_u","wind_v","blh","rain","fire_count",
"ndvi","lst","elevation","population","pm25","aqi_class","disease_risk","provenance"]

def test_schema_and_breakpoints():
    df = generate(n_days=10, seed=42)
    assert list(df.columns) == EXPECTED_COLS
    assert set(df.zone) >= {"IGP","South","Himalaya","Islands","Northeast"}
    # synthetic smoke only; real-data schema identical + provenance must read 'nasa_*'
    assert (df["provenance"] == "synthetic_generator_v1").all()
    # CPCB breakpoints
    assert aqi(30) == "Good" and aqi(60) == "Satisfactory"
    assert aqi(90) == "Moderate" and aqi(120) == "Poor"
    assert aqi(250) == "Very Poor" and aqi(251) == "Severe"
    # bbox whole India
    assert df.lat.between(6.0, 37.5).all() and df.lon.between(68.0, 97.5).all()
    # known test districts present (Leh, Port Blair, Kavaratti, Itanagar etc.)
    for d in ["Leh","Port Blair","Kavaratti","Itanagar","Maheswaram","Bhuj","Srinagar","Kanyakumari"]:
        assert d in set(df.district), f"missing {d}"
