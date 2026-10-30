"""Honest geographic test: index composition, disambiguation, and no silent swaps.

Uses actual reference records only — no invented coordinates or results.
Existing app/API tests must keep passing unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.geography import (load_states, load_districts, search_offline,
                               coverage_summary, load_combined_index)
from src.api.main import answer


def test_all_states_indexed():
    states = load_states()
    names = {r["state"] for r in states}
    assert len(names) == 36, f"expected 28 states + 8 UTs, got {len(names)}"
    for required in ["Maharashtra", "Uttar Pradesh", "Tamil Nadu", "Assam",
                     "Ladakh", "Lakshadweep", "Puducherry", "Delhi"]:
        assert required in names, f"missing {required}"


def test_district_names_cover_all_states():
    districts = load_districts()
    per_state: dict[str, int] = {}
    for r in districts:
        per_state[r["state"]] = per_state.get(r["state"], 0) + 1
    assert len(per_state) == 36, f"districts missing for states: {per_state.keys()}"
    # Spot-check actual reference coverage across regions
    idx = {(r["state"], r["district"]) for r in districts}
    for s, d in [("Maharashtra", "Thane"), ("Uttar Pradesh", "Varanasi"),
                 ("Tamil Nadu", "Vellore"), ("Assam", "Kamrup Metropolitan"),
                 ("Jammu and Kashmir", "Srinagar"), ("Ladakh", "Leh")]:
        assert (s, d) in idx, f"missing district {d}, {s}"


def test_offline_search_exact_partial_alias_pin():
    assert search_offline("Mumbai")[0]["place"] == "Mumbai"
    assert search_offline("mumbai")[0]["place"] == "Mumbai"
    assert search_offline("Mumb")[0]["place"] == "Mumbai"
    assert search_offline("bombay")[0]["place"] == "Mumbai"
    assert search_offline("632101")[0]["place"] == "Anaikkattu"
    # duplicate district name shows both states for disambiguation
    aur = search_offline("Aurangabad")
    states = {r["state"] for r in aur}
    assert "Bihar" in states and "Maharashtra" in states


def test_state_district_filters_narrow_duplicates():
    mh = search_offline("Aurangabad", state="Maharashtra")
    br = search_offline("Aurangabad", state="Bihar")
    assert mh and mh[0]["state"] == "Maharashtra"
    assert br and br[0]["state"] == "Bihar"


def test_name_only_district_has_no_fake_coordinates():
    hits = search_offline("Thane")
    assert hits, "Thane should be indexed by name"
    thane = [h for h in hits if h["place"] == "Thane"]
    assert thane, f"Thane record missing: {hits}"
    # Honest: no invented coordinates — assessment must not use this directly
    assert thane[0].get("lat") is None


def test_answer_uses_offline_gazetteer_without_network():
    r = answer("Mumbai", None, None)
    assert r.get("place") == "Mumbai"
    assert r.get("state") == "Maharashtra"
    assert "pm25" in r


def test_answer_never_returns_wrong_city():
    r = answer("__definitely_not_an_indian_place__", None, None)
    assert "pm25" not in r
    assert "suggestions" in r
    assert r.get("place") is None
    r2 = answer("Thane", None, None)
    # Thane is name-only offline; with no network it must explain, not swap city
    if "pm25" in r2:
        assert r2.get("district") == "Thane" or r2.get("place") == "Thane"


def test_coverage_summary_is_honest():
    cov = coverage_summary()
    assert cov["states_uts"] == 36
    assert cov["districts_indexed"] >= 700
    assert cov["districts_with_coordinates"] < cov["districts_indexed"]
    assert cov["stations"]["stations_indexed"] >= 1
    assert "model_inputs" in cov
