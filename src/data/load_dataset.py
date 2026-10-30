"""Load canonical datasets by project-root paths (pathlib, no cwd dependence)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
def _root() -> Path:
    return Path(__file__).resolve().parents[2]
def raw_synthetic() -> Path:
    return _root() / "data" / "raw" / "synthetic" / "india_dataset.csv"
def processed(name: str) -> Path:
    return _root() / "data" / "processed" / name
def load_raw_synthetic() -> pd.DataFrame:
    p = raw_synthetic()
    if not p.exists():
        raise RuntimeError(f"FAILED: missing synthetic dataset {p}; generate with: python src/data/build_csv.py")
    return pd.read_csv(p, encoding="utf-8")
def load_processed(name: str) -> pd.DataFrame:
    p = processed(name)
    if not p.exists():
        raise RuntimeError(f"FAILED: missing processed file {p}; run preprocessing pipeline first")
    return pd.read_csv(p, encoding="utf-8")
def main() -> None:
    df = load_raw_synthetic()
    print(f"rows={len(df)} cols={len(df.columns)} -> {raw_synthetic()}")
if __name__ == "__main__":
    main()
