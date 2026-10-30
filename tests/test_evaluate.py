"""Tests for Step 8 evaluate (metric helpers only, fast)."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate import cls_scores, reg_scores, brier_picp

def test_cls_reg_helpers():
    y = np.array([0, 1, 2, 1, 0, 2])
    p = np.array([0, 1, 2, 1, 0, 1])
    s = cls_scores(y, p)
    assert 0 <= s["f1"] <= 1 and 0 <= s["acc"] <= 1 and "kappa" in s
    assert s["f1"] < 1.0 and s["acc"] < 1.0  # imperfect input -> never 1.0
    r = reg_scores(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert r["rmse"] >= r["mae"] >= 0
    b = brier_picp(np.array([0, 2]), np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
                   np.array([30.0, 90.0]), np.array([20.0, 80.0]), np.array([40.0, 100.0]))
    assert b["picp"] == 1.0 and "brier_disease" in b  # Brier validated separately (NA if synthetic)
