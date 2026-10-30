"""Entry: preview NASA queries (dry-run) — real downloads need .env creds."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download_data import search_download
import argparse
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-01"); ap.add_argument("--end", default="2024-08-05")
    ap.add_argument("--out", default="data/raw/nasa"); ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    search_download(a.start, a.end, Path(a.out), dry=not a.real)
if __name__ == "__main__":
    main()
