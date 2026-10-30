"""List remaining root + sweep stale refs, log to logs/sweep.txt."""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parent
out = []
rows = sorted([p.name for p in ROOT.iterdir() if p.is_file()])
out.append(f"root-files={len(rows)}")
out += [" - " + r for r in rows]
STALE = [r"configs/config\.yaml(?!_corrected)", r"src/(balance|train|diagnose|evaluate|leakage|preprocess|split|build_csv|download|mesh|explain)\.py",
         r"app/main", r"app/static", r"results/(model_diagnosis|diagnosis|cv_metrics|cv_summary|test_metrics|dataset_balance|leakage_report|test_table|smote_log|run_smoke|learning_curve|confusion_matrix|india_aqi_map)\.",
         r"data/india_dataset\.csv", r'(?<![\w/])data/stations\.csv',
         r"results/checkpoints", r"from (balance|train|diagnose|evaluate|leakage|preprocess|split|build_csv|download|mesh|explain|require|main|update_stations) import",
         r"from models\.(baselines|envmeshnet) import", r"\"results\"", r"'results'"]
hits = []
for p in sorted(ROOT.rglob("*.py")):
    if ".venv" in p.parts or "__pycache__" in p.parts:
        continue
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith("_") or rel in ("_cleanup2.py", "_sweep.py"):
        continue
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        continue
    for i, ln in enumerate(lines, 1):
        for pat in STALE:
            if re.search(pat, ln):
                s = ln.strip()
                hits.append(f"{rel}:{i}: {pat} :: {s[:130]}")
out.append(f"stale-hits={len(hits)}")
out += hits
(ROOT / "logs" / "sweep.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
