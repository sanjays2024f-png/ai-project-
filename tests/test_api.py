"""API endpoint test (TestClient; requires fastapi + httpx)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_api_air_returns_card_schema():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    r = c.get("/api/air", params={"q": "Leh"})
    assert r.status_code == 200
    j = r.json()
    for k in ["place", "pm25", "aqi_category", "source", "nearest_station",
              "distance_km", "confidence", "updated_at", "disease_risk",
              "advice", "forecast"]:
        assert k in j, f"missing {k}"
    assert len(j["forecast"]) == 3

def test_api_latlon_inside_india_never_errors():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    r = c.get("/api/air", params={"lat": 11.62, "lon": 92.93})
    assert r.status_code == 200 and "pm25" in r.json()

