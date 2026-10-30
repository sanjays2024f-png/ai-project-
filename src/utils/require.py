"""Hard dependency guard. Import-first, fail loudly — no silent fallbacks.

Usage:
    from require import need
    need("sklearn", "scikit-learn")
    need("torch", "torch", extra="pip install torch --index-url https://download.pytorch.org/whl/cpu")
"""
from __future__ import annotations
import importlib

_HINTS: dict[str, str] = {
    "sklearn": "pip install scikit-learn==1.5.1",
    "imblearn": "pip install imbalanced-learn==0.12.4",
    "torch": "pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu",
    "yaml": "pip install pyyaml==6.0.2",
    "matplotlib": "pip install matplotlib==3.9.2",
    "shap": "pip install shap==0.46.0",
    "pandas": "pip install pandas==2.2.2",
    "numpy": "pip install numpy==1.26.4",
    "scipy": "pip install scipy==1.13.1",
    "xgboost": "pip install xgboost==2.1.1",
    "pytest": "pip install pytest==8.3.2",
}


def need(module: str, dist: str | None = None, extra: str = "") -> object:
    """Import module or raise actionable RuntimeError. No fallback."""
    try:
        return importlib.import_module(module)
    except ImportError as ex:
        hint = extra or _HINTS.get(module, f"pip install {dist or module}")
        raise RuntimeError(
            f"FAILED: missing required dependency '{module}'. "
            f"Activate venv first ('.\\.venv\\Scripts\\Activate.ps1'), then run: {hint}"
        ) from ex
