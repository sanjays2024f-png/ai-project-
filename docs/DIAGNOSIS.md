# Env-MeshNet — DIAGNOSIS + REPAIR REPORT (Task 1-8)

## Task 1 — root causes found
1. `src/train.py:run_cv` ran only `folds[:2]` (demo) but labeled output as full CV; `envmeshnet`
   was `run_sk("rf")` with `note="mesh-stub(rf)"` — NOT the PyTorch geo-mesh. FIXED: all folds,
   `run_mesh()` trains real `EnvMeshNet` (Delaunay faces + mesh-conv + global pool + 3 heads).
2. Silent fallbacks masked missing deps as success:
   - `f1_acc` returned `acc==f1` without sklearn; `run_sk/run_torch` returned majority class;
   - `smote_fold` fell back to random oversampling; `cv_folds` fell back to hash folds;
   - `evaluate.cls_scores` faked metrics; fabricated `pd3` Brier probs from AQI class;
   - `explain` wrote RF importances to `shap_importance.csv` (mislabeled).
   FIXED: `src/require.py::need()` — every critical module calls `need()` first; failures raise
   `RuntimeError("FAILED: ...")` with the exact pip fix. No `except ImportError: fallback`.
3. `acc=1.0/f1=1.0` cause: probe `run_sk("rf", Xtr, Xtr)` evaluated on TRAIN slice (diagnose
   `quick_scores` predicts `Xtr[:n]` with a model fit on same rows) — memorisation, not research.
   FIXED: honest `diagnosis.json` now records `model_identity="RF probe (diagnosis only)"`,
   real train/val gap, versions, sample counts; final metrics come only from untouched test.
4. Graph model gap: `EnvMeshNet` existed but `train.py` never called it. FIXED: `run_mesh()`
   builds Delaunay faces (`src/mesh.py`), trains PyTorch mesh-conv with class-weighted CE
   (SMOTE intentionally NOT applied to graph — would break spatial structure), MC-dropout kept
   in `predict_mc` for uncertainty. Docstring records adaptation-vs-canonical-MeshNet discrepancy.
5. Disease leakage: `disease_risk` synthetic labels derived from `pm25` (circular). FIXED:
   `evaluate.py` sets `brier_disease=NaN + brier_note="needs real IDSP/HMIS/NFHS labels"`;
   `EnvMeshNet.dis_head` documented valid-only-with-real-labels; provenance column added.
6. Data provenance: `build_csv.py` synthetic had no tag. FIXED: `provenance="synthetic_generator_v1"`
   column; schema test asserts it; README + reports state smoke-only.
7. Preprocess norm: fit on earliest-70% pseudo-train slice inside `preprocess.py` — kept, but
   documented that production must fit on `train.csv` only (split first, then normalise).

## Task 2 — environment (Python 3.11.9, Windows 10 SP0, AMD64)
- `python -m venv .venv` was started (see `_diag_env.txt`: numpy 2.4.6/pandas 3.1.0/pip present,
  sklearn/torch/matplotlib/shap/yaml MISSING).
- Fix: `scripts\setup_env.bat` (venv + CPU torch + requirements + `check_env.py`).
- `requirements.txt` repaired: added `joblib==1.4.2`, install order note for CPU torch,
  NASA/geo stack moved to optional (earthaccess/xarray/netCDF4/h5py/zarr need network+HDF5).
- `scripts/check_env.py` verifies all 14 imports; `tests/test_env.py` guards config.

## Task 3 — fallback removal (files changed)
- `src/require.py` NEW (single fail-loud gate).
- `src/balance.py`: strict SMOTE (needs imblearn+sklearn, raises on single-class fold).
- `src/train.py`: `f1_acc/run_sk/run_torch` no fallbacks; `run_mesh` real PyTorch; all folds;
  `cv_summary.json` now has folds_completed/models/split/smote/seed/epochs/versions.
- `src/evaluate.py`: real sklearn scores; Brier=NaN with reason; model renamed `envmeshnet(rf-probe)`.
- `src/diagnose.py`: strict deps; diagnosis.json has model_identity/data/versions/warnings.
- `src/split.py`: strict StratifiedGroupKFold (no hash fallback).
- `src/leakage.py`: strict sklearn AUC (no skip).
- `src/explain.py`: strict shap (no mislabeled CSV).
- `src/models/*.py`: strict torch imports + adaptation discrepancy documented.
- `src/preprocess.py/load_cfg`: fail loudly.

## Task 4 — graph model
- Nodes = grid cells (`cell_id`), edges = Delaunay face adjacency (neighbour faces, -1=border).
- Node/face features = mean AOD/NO2/met/wind/fire/rain per face (`mesh.env_face`).
- Temporal: inner time split 80/20 per fold; lag/rolling features from preprocess.
- Heads: AQI-CE + PM2.5-MSE + disease-CE (masked/NA unless real labels).
- Actual PyTorch training: YES after repair (`run_mesh`); RF probe retained ONLY for
  sklearn baselines/diagnosis, never presented as Env-MeshNet.

## Task 5 — pipeline order enforced
synthetic/raw -> validate schema -> preprocess -> leakage -> split (time + holdout) ->
SMOTE train-only per fold (graph uses class weights) -> train/tune -> untouched test ->
metrics + LaTeX. Imputation/scaling fit on train only (documented limitation in
preprocess pseudo-slice; production: split-then-fit).

## Task 6-9 — outputs/tests/runs: see final chat summary (counts depend on env install).
