"""3-day PM2.5 forecast heuristic (mirrors src/api/main.py logic)."""
from __future__ import annotations
def forecast_3day(pm_today: float) -> list[float]:
    """Naive drift forecast: +3/-1/+2 pattern clamped >=8. Smoke only."""
    return [round(max(8.0, pm_today + ((i + 1) * 3 - 4)), 1) for i in range(3)]
