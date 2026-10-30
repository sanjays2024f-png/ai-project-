"""Tests for Step 2a downloader (no login, no network)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download_data import BBOX, DATASETS, plan, firms_fires

def test_bbox_whole_india():
    assert BBOX == (68.0, 6.0, 97.5, 37.5)

def test_no_tempo():
    names = " ".join(d.short + d.key for d in DATASETS).lower()
    assert "tempo" not in names  # TEMPO is North-America-only

def test_plan_covers_key_datasets():
    p = plan("2024-08-01", "2024-08-05")
    txt = "\n".join(p).lower()
    for k in ["mcd19a2", "m2t1nxaer", "omno2d", "mopitt", "gpm_3imergdf", "mod13a2", "mod11a1"]:
        assert k in txt, f"missing {k}"
    assert len(DATASETS) >= 9
    assert callable(firms_fires)  # FIRMS VIIRS NRT via MAP_KEY API
