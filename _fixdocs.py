"""Fix 3 stale docstring refs (docs only), then compile-check all files."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
log = []
def R(rel, old, new):
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    if old in t:
        p.write_text(t.replace(old, new), encoding="utf-8")
        log.append(f"OK {rel}")
    else:
        log.append(f"MISS {rel}: {old[:60]!r}")
R("src/data/split.py", 'Run: python src/split.py --in data/processed/india_proc.csv --outdir data/processed',
  'Run: python src/data/split.py --in data/processed/india_proc.csv --outdir data/processed')
R("src/evaluation/diagnose_model.py",
  '  python src/diagnose.py --train data/processed/train.csv --val data/processed/val.csv',
  '  python src/evaluation/diagnose_model.py --train data/processed/train.csv --val data/processed/val.csv')
R("src/evaluation/diagnose_model.py",
  'print("Retrain: python src/train.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --config results/config_corrected.yaml")',
  'print("Retrain: python src/training/train.py --train data/processed/train.csv --folds data/processed/cv_folds.csv --config config/config_corrected.yaml")')
R("src/training/train.py", '"""All folds x 10 models -> results/cv_metrics.csv. REAL training, no stubs."""',
  '"""All folds x 10 models -> results/metrics/cv_metrics.csv. REAL training, no stubs."""')
import compileall
ok = compileall.compile_dir(str(ROOT / "src"), quiet=1)
ok2 = compileall.compile_dir(str(ROOT / "scripts"), quiet=1)
ok3 = compileall.compile_dir(str(ROOT / "tests"), quiet=1)
log.append(f"compile src={ok} scripts={ok2} tests={ok3}")
(ROOT / "logs" / "fix_docs.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
