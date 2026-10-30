"""Central project-root path resolution (pathlib, no cwd dependence)."""
from __future__ import annotations
from pathlib import Path
def ROOT() -> Path:
    """Project root = envmeshnet/ (two levels above src/utils)."""
    return Path(__file__).resolve().parents[2]
def CONFIG() -> Path: return ROOT() / "config" / "settings.yaml"
def MODEL_CONFIG() -> Path: return ROOT() / "config" / "model_config.yaml"
def RAW_SYNTH() -> Path: return ROOT() / "data" / "raw" / "synthetic" / "india_dataset.csv"
def PROC_DIR() -> Path: return ROOT() / "data" / "processed"
def EXT_STATIONS() -> Path: return ROOT() / "data" / "external" / "stations.csv"
def METRICS_DIR() -> Path: return ROOT() / "results" / "metrics"
def REPORTS_DIR() -> Path: return ROOT() / "results" / "reports"
def FIGURES_DIR() -> Path: return ROOT() / "results" / "figures"
def CHECKPOINTS_DIR() -> Path: return ROOT() / "models" / "checkpoints"
def FRONTEND() -> Path: return ROOT() / "frontend" / "index.html"
