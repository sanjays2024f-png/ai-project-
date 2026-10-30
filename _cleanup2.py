"""Cleanup root clutter (log to logs/cleanup.txt to avoid self-lock)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "cleanup.txt"
LOG.parent.mkdir(parents=True, exist_ok=True)
out = []
# setup_env.bat fix (idempotent)
p = ROOT / "scripts" / "setup_env.bat"
t = p.read_text(encoding="utf-8")
if "run_all.py" in t:
    p.write_text(t.replace("run_all.py", "run_pipeline.py"), encoding="utf-8")
    out.append("setup_env.bat -> run_pipeline.py")
else:
    out.append("setup_env.bat already ok")
import os
# collect victims (NOT this script, NOT the log file)
victims = []
for pat in ["_*.py", "_*.txt", "_*.log"]:
    for q in ROOT.glob(pat):
        if q.name in ("_cleanup2.py",):
            continue
        victims.append(q)
for q in ["run_diag2.py", "run_diagnosis.py", "diag_m2.py", "audit_tree.py",
          "diag_balance.py", "diag_metrics.py", "regen_plot.py", "finalize_json.py"]:
    f = ROOT / q
    if f.exists():
        victims.append(f)
for v in sorted(set(victims)):
    try:
        v.unlink()
        out.append(f"del {v.name}")
    except Exception as ex:
        out.append(f"KEEP {v.name}: {ex}")
# legacy dirs/files
import shutil
for d in ["app", "configs"]:
    dd = ROOT / d
    if dd.exists():
        shutil.rmtree(dd)
        out.append(f"rmdir {d}/")
d2 = ROOT / "results" / "diag2_out.txt"
if d2.exists():
    shutil.move(str(d2), str(ROOT / "logs" / "diag2_out.txt"))
    out.append("results/diag2_out.txt -> logs/diag2_out.txt")
ds = ROOT / "results" / "diag_stdout.txt"
if ds.exists():
    ds.unlink()
    out.append("del results/diag_stdout.txt")
rc = ROOT / "results" / "checkpoints"
if rc.exists() and not any(rc.iterdir()):
    shutil.rmtree(rc)
    out.append("rmdir results/checkpoints/")
for k in ["data/raw/nasa", "data/raw/cpcb", "data/interim", "data/external",
          "models/checkpoints", "models/metadata", "logs", "notebooks",
          "frontend/assets/css", "frontend/assets/js", "frontend/assets/images",
          "frontend/components", "results/predictions"]:
    (ROOT / k).mkdir(parents=True, exist_ok=True)
for k in ["data/processed", "results", "results/metrics", "results/figures",
          "results/reports", "src", "src/models", "tests", "scripts",
          "data/raw", "config"]:
    g = ROOT / k / ".gitkeep"
    if g.exists():
        g.unlink()
        out.append(f"del stale {k}/.gitkeep")
LOG.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
