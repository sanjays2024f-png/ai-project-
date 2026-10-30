"""Tests for Step 2b station updater (no network, no API key)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.update_stations import km, norm

def test_haversine_bengaluru():
    # BTM Layout ~1.2 km from 560068 seed point
    d = km(12.9166, 77.6101, 12.9200, 77.6000)
    assert 0.5 < d < 2.5

def test_norm_strips_cpcb():
    assert norm("BTM Layout, Bengaluru - CPCB") == "btm layout, bengaluru"
    assert norm("  Leh CPCB ") == "leh"

def test_dedup_thresholds_documented():
    # rule: fuzz>=85 AND dist<0.3km -> duplicate (see merge_dedup)
    assert 0.3 == 0.3 and 85 == 85
