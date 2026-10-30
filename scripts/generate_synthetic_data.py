"""Entry: (re)generate synthetic india_dataset.csv (SMOKE ONLY)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate
def main() -> None:
    out = Path("data/raw/synthetic/india_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_days=180, seed=42)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {out} (synthetic_generator_v1)")
if __name__ == "__main__":
    main()
