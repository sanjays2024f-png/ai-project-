"""Tests for Step 9 app (offline: station/PIN/satellite paths only). All English UTF-8."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.main import answer, aqi, value_for, BBOX

CASES = ["BTM Layout, Bengaluru - CPCB", "BTM Layout", "Bengaluru", "560068",
         "Maheswaram Ranga Reddy", "Leh", "Port Blair", "Kavaratti", "Itanagar",
         "Kanyakumari", "Bhuj", "Srinagar"]

def test_required_locations_return_cards():
    for q in CASES:
        r = answer(q, None, None)
        assert "pm25" in r and "aqi_category" in r, f"{q} -> {r}"
        assert "source" in r and "forecast" in r and len(r["forecast"]) == 3
        assert r["source"] in ("Measured", "Interpolated", "Satellite estimate")

def test_every_india_point_returns_value():
    pts = [(28.6, 77.2), (12.9, 77.6), (34.1, 77.5), (11.6, 92.9),
           (10.5, 72.6), (27.1, 93.6), (8.0, 77.5), (23.2, 69.6), (19.0, 72.8)]
    for la, lo in pts:
        r = answer(None, la, lo)
        assert "pm25" in r, f"{la},{lo} -> {r}"
        assert 68.0 <= lo <= 97.5 and 6.0 <= la <= 37.5

def test_aqi_breakpoints():
    assert aqi(30) == "Good" and aqi(60) == "Satisfactory" and aqi(251) == "Severe"

def test_no_mojibake_in_responses():
    """No encoding artifacts in API output; units render as single-encoded UTF-8."""
    bad = ["Ã©", "Ã³", "�", "â€"]
    for q in ["Bengaluru", "Leh", "560068", "Kavaratti", "Port Blair", "Delhi"]:
        r = answer(q, None, None)
        blob = str(r)
        for token in bad:
            assert token not in blob, f"mojibake {token!r} in {q}: {blob[:200]}"
    r = answer("Kanyakumari", None, None)
    assert r.get("source") == "Satellite estimate"
    assert r.get("confidence") in ("High", "Medium", "Low")

def test_location_search_keeps_requested_places_separate():
    results = {q: answer(q, None, None) for q in ["Vellore", "Leh", "Bengaluru"]}
    for q, result in results.items():
        assert result.get("place") == q
        assert "pm25" in result and len(result["forecast"]) == 3
    assert len({result["place"] for result in results.values()}) == 3

def test_unsupported_location_does_not_return_another_city():
    result = answer("__definitely_not_an_indian_place__", None, None)
    assert "pm25" not in result
    assert "suggestions" in result
    assert result.get("place") is None

def test_missing_station_records_are_explicit():
    pm, source, nearest, distance = value_for(12.92, 79.13, [])
    assert isinstance(pm, float)
    assert source == "Satellite estimate"
    assert nearest == "none"
    assert distance == 999.0

def test_vellore_district_reference_places_resolve_separately():
    places = ["Vellore", "Katpadi", "Gudiyatham", "Pernampattu",
              "K V Kuppam", "Anaikkattu"]
    results = [answer(place, None, None) for place in places]
    assert [r["place"] for r in results] == places
    assert all(r["district"] == "Vellore" for r in results)
    assert all(r["state"] == "Tamil Nadu" for r in results)
    assert all(r["source"] == "Satellite estimate" for r in results)
    assert len({(r["latitude"], r["longitude"]) for r in results}) == len(places)

def test_vellore_district_alias_and_pin_resolution():
    assert answer("pernambut", None, None)["place"] == "Pernampattu"
    assert answer("632101", None, None)["place"] == "Anaikkattu"
