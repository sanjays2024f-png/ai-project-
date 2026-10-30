"""Env check: python/OS + required imports + versions. Fails loudly with fix hints."""
from __future__ import annotations
import platform
import sys

REQS = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "joblib",
        "torch", "yaml", "matplotlib", "shap", "pytest", "xgboost",
        "fastapi", "rapidfuzz"]
OPTIONAL = ["earthaccess", "xarray", "netCDF4", "h5py", "zarr"]

def main() -> int:
    """Print env report; return 0 if all present else 1."""
    print(f"python={platform.python_version()} os={platform.platform()}")
    print(f"exe={sys.executable}")
    bad: list[str] = []
    for m in REQS:
        try:
            mod = __import__(m)
            print(f"OK {m} {getattr(mod, '__version__', '?')}")
        except Exception as ex:
            bad.append(m)
            print(f"MISSING {m}: {ex}")
    if bad:
        print("FAILED: install missing with:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install -r requirements.txt")
        return 1
    import torch
    print(f"torch_cuda={torch.cuda.is_available()} torch_ver={torch.__version__} (CPU build expected on this box)")
    for m in OPTIONAL:
        try:
            mod = __import__(m)
            print(f"OPTIONAL OK {m} {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"OPTIONAL MISSING {m} (only needed for real NASA downloads)")
    print("ENV OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
