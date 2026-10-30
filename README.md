# Env-MeshNet — airborne pollution warning + respiratory risk for whole India

Research prototype: novel **Env-MeshNet** (ADAPTATION of MeshNet, Feng et al. AAAI 2019,
originally for 3D shapes — here adapted to geo-environmental triangular meshes over India)
+ 9 baselines, plus a simple web app (type any location, see air quality).

> REPAIR STATUS (2026-10): the pipeline was hardened to fail loudly instead of faking
> metrics. `src/require.py` gates every critical module; `train.py` runs REAL PyTorch
> geo-mesh training on ALL folds; disease Brier = NA (synthetic labels are circular);
> synthetic data carries `provenance=synthetic_generator_v1` and is smoke-only.
> See `DIAGNOSIS.md` for the full root-cause report.

## Step 1 — what exists now
- `configs/config.yaml` — all settings (bbox 68/6/97.5/37.5, CPCB breakpoints, zones, split, SMOTE, model, webapp)
- `requirements.txt` — pinned deps
- `.env.example` — copy to `.env` for EARTHDATA_USERNAME/PASSWORD, FIRMS_MAP_KEY, OPENAQ_API_KEY
- `src/build_csv.py` — synthetic `data/india_dataset.csv` generator (same schema as real NASA pipeline)
- Folder skeleton: `data/raw, data/processed, src/models, app/static, scripts, notebooks, tests, results`

## Environment setup (Python 3.11, Windows, CPU torch)
```powershell
cd envmeshnet
# automated:
.\scripts\setup_env.bat
# manual:
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
$env:PYTHONUTF8=1
python scripts\check_env.py   # expect ENV OK + torch_ver=2.4.1+cpu
```

## Quickstart (synthetic SMOKE ONLY — never research evidence)

## Schema
`date, cell_id, lat, lon, district, state, zone, aod, no2, so2, co, bc, dust, temp, rh,
wind_u, wind_v, blh, rain, fire_count, ndvi, lst, elevation, population, pm25, aqi_class, disease_risk`

## Important caveats (research honesty)
- Synthetic `disease_risk` is derived from `pm25` → CIRCULAR. Replace with real
  IDSP/HMIS/NFHS labels before any paper claim. Stated here and in code.
- NASA TEMPO covers North America only — NOT used for India.
- Known satellite gaps simulated as NaN: Himalaya snow/cloud, Thar bright surface,
  NE/Ghats monsoon cloud, islands (satellite-only), OMI row anomaly (QA flags in real pipeline).
- Env-MeshNet is an adaptation of MeshNet to geo-meshes (see future `src/models/envmeshnet.py`).

## Step 2 — NASA download + station registry (new)
- `src/download.py` — earthaccess search/download for MCD19A2 (061), M2T1NXAER (5.12.4),
  MOD04_L2 DeepBlue, OMNO2d (003, row-anomaly QA mask), MOPITT CO, M2T1NXSLV (5.12.4),
  GPM_3IMERGDF (07), MOD13A2 (061), MOD11A1 (061), NASADEM_HGT (001) over bbox 68/6/97.5/37.5.
  No TEMPO (North-America-only). `firms_fires()` pulls VIIRS_SNPP_NRT via FIRMS_MAP_KEY.
- `scripts/update_stations.py` — daily refresh: OpenAQ v3 (bbox, OPENAQ_API_KEY) + seed
  `data/stations.csv` + any `data/raw/cpcb*.csv` dumps; dedup = rapidfuzz>=85 AND dist<300m.
- Tests: `tests/test_download.py`, `tests/test_update_stations.py` (no login/network needed).

## Next steps (do NOT run yet — wait for `next`)
Step 3: `src/preprocess.py` (QA masks, gap-fill, lags, wind, Diwali flags, train-only normalisation).
## Step 3 — preprocess (new)
- `src/preprocess.py`: parse dates -> QA gap-fill (time interp per cell, then zone median as
  MERRA-2 fallback) -> quantile clip (0.5-99.5%) -> wind speed/dir + winter/monsoon/Diwali flags
  -> lags 1/3/7d + rolling 3/7d per cell -> z-score with stats FIT ON TRAIN slice only (earliest
  70% dates) saved to `data/processed/norm_stats.json`.
- Test: `tests/test_preprocess.py` (NaN gone, wind Dir 0-360, lag cols, train mean ~0).

## Step 4 — leakage + split (new)
- `src/leakage.py`: [1] target (banned PM10/AQI cols, |corr|>0.95 vs pm25), [2] duplicates +
  (date,cell_id) overlap train/test, [3] suspicious (near-constant/ID-like/single-feat AUC>0.98),
  [4] temporal+spatial note. Writes `results/leakage_report.txt` (re-run with --train/--test).
- `src/split.py`: time-based 70/15/15 (latest=test, never random) or train_test mode;
  unseen-district hold-out (`Leh,Kavaratti,Itanagar`) -> `holdout.csv`; StratifiedGroupKFold(10,
  district) on TRAIN only -> `cv_folds.csv`.
- Test: `tests/test_leakage_split.py` (no banned miss, 0 dups, train<=val<=test, overlap 0).

## Step 5 — SMOTE train-only (new)
- `src/balance.py`: per-CV-fold SMOTE (`k=5`, `sampling_strategy=auto`, `random_state=seed+fold`)
  on TRAINING fold only; minority classes upsampled, val/test never touched. `k` auto-shrinks if a
  fold minority has <6 samples. Falls back to random oversampling if imblearn missing.
  Writes `results/smote_log.json` (before/after counts per fold) + `train_balanced_demo_fold0.csv`.
- Test: `tests/test_balance.py` (balanced fold grows, val/test untouched).

## Step 6 — mesh + models + training (new)
- `src/mesh.py`: Delaunay over cell points; faces Fx3 + neighbours Fx3 (-1=border);
  `face_centres` (lat/lon), `struct_feats` (6 offsets + 3 neighbour ids), `env_face`
  (mean AOD/NO2/met/wind/fire/rain over 3 corners per date).
- `src/models/envmeshnet.py`: ADAPTATION of MeshNet (Feng AAAI19, 3D shapes -> geo-meshes):
  spatial MLP + 2 mesh-conv (face+neighbour mean) + global pool -> AQI softmax + PM2.5 +
  disease (pollution emb + pop); `predict_mc` for uncertainty. Loss weights in config.
- `src/models/baselines.py`: 10 total = envmeshnet + logreg/rf/xgb/svm/mlp + cnn1d/lstm/gru/gat.
- `src/train.py`: same splits + same SMOTE rule for all; inner time split 80/20 per fold;
  2-fold demo x 10 models -> `results/cv_metrics.csv` + `cv_summary.json` (full 10 folds for paper).
- Test: `tests/test_mesh_models.py` (F faces, (F,9) struct, forward shapes).

## Step 7 — diagnose + correct + retrain (new)
- `src/diagnose.py`: RF probe -> train/val macro-F1; `overfit` if gap>10%, `underfit` if both<0.5
  else `good`. Saves `results/learning_curve.png` + `diagnosis.json` + `config_corrected.yaml`.
  Overfit fix: dropout 0.5, wd 1e-3, patience 3, hidden 32, 1 block, noise+label-smooth.
  Underfit fix: hidden 128, 3 blocks, 80 epochs, dropout 0.1, wd 1e-5.
  Retrain: `python src/train.py --config results/config_corrected.yaml`.
- Test: `tests/test_diagnose.py` (thresholds + recipes).

## Step 8 — FINAL evaluation + explain (new)
- `src/evaluate.py` (run ONCE on untouched test): acc/macro-F1/prec/rec/AUC-ovr/kappa +
  confusion + RMSE/MAE (class-median pm25 map) + Brier/PICP/CRPS-width; per-zone (8) +
  winter-smog/monsoon splits; ablations no_fire/no_lags/no_pop/no_mesh.
  Outputs: `results/test_metrics.csv` + `test_table.tex` + `confusion_matrix.png` + `india_aqi_map.png`.
- `src/explain.py`: SHAP (TreeExplainer RF probe, RF-importance fallback) ->
  `shap_importance.csv/.png`.
- Test: `tests/test_evaluate.py` (helper ranges).

## Step 9 — web app (new, final)
- `app/main.py`: `answer(q/lat/lon)` + FastAPI `GET /api/air`. Search: (1) fuzzy>=80 (CPCB strip),
  (2) Photon->Nominatim (in, cache, 1/s, UA), (3) did-you-mean top-3 (never bare no-match).
  PIN 560068 mapped. Value: <=10km Measured, <=50km IDW-x3 Interpolated, else Satellite (0.1deg grid).
  Card: place, pm25, AQI badge, source, nearest+km, confidence, updated, disease risk, advice, 3-day forecast.
- `app/static/index.html`: single dark page, one search + button, 300ms autocomplete, no tabs/metrics/thresholds.
- Test: `tests/test_app.py` (all 12 required queries + 9 India points, no network for station hits).

Geographic search uses `data/external/vellore_places.csv` as a verified
reference seed (OpenStreetMap Nominatim coordinates, aliases, and PIN codes),
then uses cached Photon/Nominatim lookup across India for other resolvable
places. This is not a complete India administrative index: district-boundary
GeoJSON, an audited all-India place database, and PIN coverage are not present
in this workspace. The available project pollution datasets also contain no
Vellore district observations, so unresolved or geocoded places use the
existing 0.1-degree satellite-grid resolution rather than being presented as
station measurements or Env-MeshNet model predictions.

## All steps complete. Run the full pipeline top-to-bottom, then `uvicorn app.main:app`.
