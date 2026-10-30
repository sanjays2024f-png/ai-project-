"""Tests for Step 7 diagnose (pure rules, no heavy fit)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.diagnose_model import decide, recommend, OVERFIT_GAP, UNDERFIT_TH

def test_decide_thresholds():
    assert OVERFIT_GAP == 0.10 and UNDERFIT_TH == 0.50
    assert decide(0.90, 0.70) == "overfit"   # gap 0.20 > 10%
    assert decide(0.80, 0.75) == "good"      # gap 5%
    assert decide(0.40, 0.35) == "underfit"  # both low
    assert decide(0.60, 0.58) == "good"

def test_recommend_matches_spec():
    o = recommend("overfit")
    assert o["dropout"] >= 0.5 and o["hidden_dim"] <= 32 and o["label_smoothing"] > 0
    u = recommend("underfit")
    assert u["hidden_dim"] >= 128 and u["epochs"] >= 80 and u["dropout"] <= 0.1
    assert recommend("good")["note"].startswith("good")
