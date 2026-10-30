"""Grid/random hyperparameter search (sklearn baselines). Fails loudly without sklearn."""
from __future__ import annotations
import itertools
import pandas as pd
def grid_search(build_fn, Xtr, ytr, Xva, yva, grid: dict):
    """Exhaustive grid over build_fn(**params). Returns (best, best_f1, rows)."""
    from sklearn.metrics import f1_score
    best, best_f1, rows = None, -1.0, []
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        m = build_fn(**params)
        m.fit(Xtr, ytr)
        f1 = float(f1_score(yva, m.predict(Xva), average="macro", zero_division=0))
        rows.append({**params, "f1_macro": round(f1, 4)})
        if f1 > best_f1:
            best, best_f1 = m, f1
    return best, round(best_f1, 4), pd.DataFrame(rows)
def main() -> None:
    print("hyperparameter_tuning: import grid_search() from scripts or notebooks")
if __name__ == "__main__":
    main()
