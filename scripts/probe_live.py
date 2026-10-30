"""Probe the FastAPI app in-process (no network) and dump plain-ASCII verification. Run from project root: python scripts/probe_live.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from src.api.main import app

c = TestClient(app)
r = c.get("/api/air", params={"q": "Leh"})
print("API_STATUS:", r.status_code)
card = r.json()
print("PLACE:", card["place"])
print("AQI:", card["aqi_category"], "PM:", card["pm25"])
print("SRC:", card["source"], "| CONF:", card["confidence"])
print("NEAR:", card["nearest_station"], card["distance_km"])
print("RISK:", card["disease_risk"])
h = c.get("/")
print("HTML_STATUS:", h.status_code)
body = h.text
print("HAS_TITLE:", "Env-MeshNet" in body and "India Air Quality" in body)
print("HAS_PLACEHOLDER:", "Enter a city, district, or state" in body)
print("HAS_UNIT:", "g/m" in body)
print("LIVE_OK")
