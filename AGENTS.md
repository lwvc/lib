# AGENTS.md

## Run (from repo root, `conda activate lib`)

```sh
python -m scripts.check_data          # pre-flight: Excel + .nii existence/shape
python -m scripts.run_predict         # main: all models × Maclaren/MMRR/OAS1 → output/*.xlsx
python -m scripts.run_resnet_val_test # separate val/test split only, → output/ResNet_val_test.xlsx
```

No requirements.txt / pyproject / tests / lint / CI. No `pytest`, no formatter.

## Hardcoded `C:\Lab` paths — do not relativize

- Data: `src/dataset.py:DATA_ROOT` = `C:\Lab\test_retest_dataset` (truth).
- Checkpoints: `src/model_registry.py:MODEL_ROOT` = `C:\Lab\model_val_test_train`.
- Outputs: `src/output.py:OUTPUT_DIR` = `C:\Lab\brain_age_test_retest\output`.
- Root `config.py` is **stale/duplicated** (`MODEL_ROOT=model_val_test`, `df/mwp1` dirs disagree with `src/dataset.py: ndf/rmi|rp1|rp2`). Ignore it; trust `src/dataset.py` + `src/model_registry.py`.

## Architecture (`src/` → `scripts/` → `analysis/`)

- `src/model_registry.py` — single source for checkpoint path, `crop`, `avg_shape`, `use_sex/use_b0`, `cat_dim`. Add models here, then list them in `scripts/run_predict.py:SFCN_*_MODELS / RESNET_MODELS`.
- `src/dataset.py:BrainAgeDataset` — Excel-driven (`.xlsx` per dataset), `return_metadata=True` is required for inference (sex/scanner). `crop_coords` inclusive on max bound.
- `src/predict.py:predict_batch` — dispatcher; keep each family's math untouched (softmax = `exp` + expected age over 95 bins `linspace(5,100,96)`; regression = direct output; ResNet feat order = sex then scanner/B0).
- `src/output.py` — 3 workbooks (`ResNet.xlsx`, `SFCN_reg.xlsx`, `SFCN_sm.xlsx`), sheets `{Dataset}_{Modality|T1[_SEX][_B0]}` truncated to 31 chars. `run_model_group` deletes the workbook at start — full reruns are destructive.
- `analysis/*.m` — MATLAB post-processing, reads `output/*.xlsx`, writes `output/plots_*/`. Run in MATLAB, not Python.

## Gotchas

- DGM is disabled everywhere (`run_predict.py`, `check_data.py`, registry `sfcn_*_dgm` commented out). Needs extra image processing — do not enable.
- Modalities actually wired: `T1WI (rmi)`, `GMP (rp1)`, `WMP (rp2)`. Required Excel cols: `Subject_ID`, `Age`, `Sex (1=Male, 2=Female)`, `Scanner (1.5 or 3 Tesla)`, plus filename col.
- SFCN: `crop=GLOBAL_CROP` + `weights_only=True` state_dict load, `DataLoader(batch_size=4, shuffle=False)`. ResNet: `crop=None` (full volume), full-model pickle load with `_alias_resnet_for_unpickle()` + `weights_only=False`, `TestBalancedAgeBatchSampler(batch_size=1)` — keep both as-is, do not unify.
- `output/*.xlsx` is gitignored. `__pycache__/` dirs are committed in tree; leave them alone.
