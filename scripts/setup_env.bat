@echo off
REM Setup project venv (Windows) + CPU torch + requirements. Run from envmeshnet\.
python --version
python -m venv .venv
call .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts\check_env.py
REM SHAP/Numba can be slow on first import (~1-2 min) — be patient.
set PYTHONUTF8=1
python scripts\run_pipeline.py
python -m pytest -v
