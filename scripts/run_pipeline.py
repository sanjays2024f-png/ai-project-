"""Run-all (smoke): generate -> preprocess -> leakage -> split -> report.

SMOKE ONLY on synthetic data — never research evidence. Full 10-model CV +
final test eval need the venv deps + real data; see README commands.
Writes a run report to results/run_smoke.json with provenance + versions.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.require import need

def main() -> int:
    """Run smoke pipeline. Returns 0 ok / 1 FAILED."""
    root = Path(__file__).resolve().parents[1]
    rep: dict = {"provenance": "synthetic_generator_v1", "smoke_only": True,
                 "steps": {}, "versions": {}}
    try:
        need("numpy")
        need("pandas")
        need("yaml")
        need("sklearn")
        import numpy, pandas, sklearn, platform
        rep["versions"] = {"python": platform.python_version(),
                           "numpy": numpy.__version__, "pandas": pandas.__version__,
                           "sklearn": sklearn.__version__}
        from src.data.build_csv import generate
        df = generate(n_days=30, seed=42)
        assert (df["provenance"] == "synthetic_generator_v1").all()
        rep["steps"]["build_csv"] = {"rows": len(df), "provenance": "synthetic_generator_v1"}
        from src.preprocessing.preprocess import parse_dates, qa_fill, add_wind_season, add_lags
        p = qa_fill(add_lags(add_wind_season(parse_dates(df))))
        assert int(p.isna().sum().sum()) == 0
        rep["steps"]["preprocess"] = {"rows": len(p), "nan_left": 0}
        from src.data.split import time_split
        parts = time_split(p, 0.70, 0.15, 0.15, "train_val_test")
        assert parts["train"]["date"].max() <= parts["val"]["date"].min()
        rep["steps"]["split"] = {k: len(v) for k, v in parts.items()}
        from src.preprocessing.leakage_checks import write_report
        write_report(p, str(root / "results" / "reports" / "leakage_report.txt"),
                     parts["train"], parts["test"])
        rep["steps"]["leakage"] = {"report": "results/reports/leakage_report.txt"}
        o = root / "results" / "reports" / "run_smoke.json"
        o.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        print("SMOKE OK (synthetic only — not research evidence)")
        return 0
    except Exception as ex:
        print(f"FAILED: {ex}")
        (root / "results" / "reports" / "run_smoke.json").write_text(json.dumps({**rep, "FAILED": str(ex)}, indent=1))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
