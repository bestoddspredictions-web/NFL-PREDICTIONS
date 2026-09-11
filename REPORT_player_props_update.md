# Player Props Update Report

**Date:** 2026-09-11

## Status
- In-progress (code and workflow updated locally). End-to-end verification possible after pushing changes and running the workflow.

## Summary
I verified `player_props/predict.py`, diagnosed failures, applied minimal fixes to make it run reliably, and added a step to the nightly workflow to generate player prop predictions after model retraining.

## What I ran
- Initial command (unmodified):

```
python player_props/predict.py
```

- Observed failures (examples):
  - XGBoost / SciPy import hang/interrupt during initial import.
  - Model prediction failures due to feature-name mismatches (models expected std/target_share fields; input had different/missing names and extra situational fields like `opponent_def_rank`, `is_home`, `days_rest`).

## Fixes applied (files changed)
- `player_props/predict.py`
  - Auto-select the most appropriate schedule CSV (prefer current-year non-empty file, otherwise newest non-empty).
  - Include `*_std_L5` and `target_share_*` fields into assembled features when present in source CSVs.
  - Align the prediction input DataFrame to each model's expected feature names by reading `booster.feature_names`, adding missing columns with default 0, and ordering columns accordingly.

(These are minimal, focused changes to address the immediate runtime and mismatch issues.)

## Workflow change (exact addition)
I added a new step to `.github/workflows/nightly-update.yml` immediately after the `Retrain player prop models` step and before artifact upload/commit. The new step runs `python player_props/predict.py` and uses `continue-on-error: true`.

Inserted block (added):

```yaml
- name: Generate fresh player prop predictions
  if: steps.season_check.outputs.in_season == 'true'
  run: |
    echo "Generating player prop predictions..."
    python player_props/predict.py
  timeout-minutes: 5
  continue-on-error: true
```

Notes:
- Placement is after `player_props/train_models.py --skip-aggregation` and before the `Upload updated predictions` and `Commit and push changes` steps (so predictions are generated before commit/push).
- The existing `git add data_files/*.csv data_files/*.json` already covers `data_files/player_props_predictions.csv`.

## Verification (local)
- After fixes, I ran:

```
python player_props/predict.py
```

- Results:
  - Script completed successfully and wrote `data_files/player_props_predictions.csv`.
  - Generated 47 prop predictions (sample rows present).
  - Example prediction `game_date`: `2026-02-08 00:00:00+00:00` (Week 22).
  - Local runtime observed: ~10.7 seconds.

**Important note about dates:**
- The produced predictions are for `2026-02-08` (Feb 8, 2026). That is stale relative to today's date (2026-09-11). The cause is local schedule files under `data_files/`: `nfl_schedule_2026.csv` is empty and the script fell back to `nfl_schedule_2025.csv` and used the most recent week available in that file.
- On CI (GitHub Actions), the workflow's earlier `build_and_train_pipeline.py` step should refresh schedule files before the prediction step. When run on GitHub in a normal nightly run, predictions should be for upcoming/current games provided the pipeline successfully updates schedule artifacts that run.

## How to test this change immediately (recommended)
Option A — CI end-to-end test (preferred):
1. Commit & push the local changes (code + workflow) to `main` (or a testing branch) and open a PR or push directly.
2. In GitHub Actions UI, manually trigger the workflow (Nightly NFL Data Update) using `workflow_dispatch`.
   - Actions → Nightly NFL Data Update → Run workflow → select branch → Run.
3. Watch the run: it will execute `build_and_train_pipeline.py`, retrain models, then run `player_props/predict.py` using the refreshed schedule and models.

Option B — Local quick test:
1. Run the pipeline locally to refresh schedule and models:

```bash
python build_and_train_pipeline.py
```

2. Re-run the predictions locally:

```bash
python player_props/predict.py
```

This will generate `data_files/player_props_predictions.csv` with whatever schedule/data the pipeline produced locally.

## Next steps / recommendations
- Push the changes to the remote repository and trigger the workflow in Actions to verify predictions are generated with fresh schedule data.
- If you want me to push and open a PR and/or trigger the workflow from here, I can create a branch, commit the changes, and open a PR; tell me to proceed.
- Optional: extend the workflow timeout for the new step if your production run takes longer than the local test (I used 5 minutes; local run was ~11s so 5 minutes should be a safe buffer).

## Artifacts
- Predictions file (local): `data_files/player_props_predictions.csv` (sample rows present).
- Code changes: `player_props/predict.py` (feature alignment and schedule selection adjustments).
- Workflow change: `.github/workflows/nightly-update.yml` (new prediction step).

---

If you want, I can now:
- Push these changes and open a PR, then trigger the workflow (Option 1), or
- Run the full local pipeline (`build_and_train_pipeline.py`) and re-run predictions locally to produce up-to-date, non-stale predictions (Option 2).

Tell me which to do next and I will proceed.
