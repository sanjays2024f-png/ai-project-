"""Grouped CV plumbing (thin wrapper over data.split.cv_folds)."""
from __future__ import annotations
import pandas as pd
from src.data.split import cv_folds
def make_folds(train: pd.DataFrame, n: int = 10, group: str = "district",
               target: str = "aqi_class", seed: int = 42) -> pd.Series:
    """StratifiedGroupKFold ids on TRAIN only. Fails loudly."""
    g = group if group in train.columns else train.columns[0]
    t = target if target in train.columns else train.columns[0]
    return cv_folds(train, n=n, group=g, target=t, seed=seed)
