# Urban Water Consumption Prediction and Analytics System

A web dashboard around the project's **existing, unchanged** CNN-LSTM-Attention forecasting models:
analytics, dataset upload and validation, predictions, model comparison, validation checks, a derived
confusion matrix, prediction history, CSV reports, the original rule-based chatbot, and a protected admin area.

> **Read this first - what the project can and cannot do** (details in `docs/MODEL_PRESERVATION_REPORT.md`)
> - The data is **32 two-month Kharghar billing totals** (APR-MAY2021 to JUN-JLY2026). The unit is shown as **Units**; the project itself marks it as unconfirmed.
> - There are **four neural variants** (LSTM, CNN-LSTM, LSTM-Attention, CNN-LSTM-Attention) and **two baselines**, not three models.
> - Only **`cnn_lstm_attention` has saved weights**, so it is the only model that can predict new data. The other three have walk-forward evaluation results only.
> - **No training/validation loss curves were ever saved**, so none are shown.
> - The models are **regressors**. R-squared is negative for every model on the 8 test bills. The confusion matrix is a *derived* low/normal/high evaluation and is labelled that way.
> - No model is trained, retrained or modified by the dashboard.

## Dashboard features

| Page | What it does (all values come from saved artifacts or the real model) |
|---|---|
| Overview | 13 summary cards (records, total/avg/max/min/latest, next-period forecast, best model + RMSE/MAE/R2, last upload, validation status) and a history + forecast chart |
| Upload Dataset | Browse/drag a .csv or .xlsx, progress, validation report, column types, missing/duplicate/invalid counts, preview, confirmation step, upload history, delete |
| Water Consumption Analytics | Trend with moving average, % change, season-by-year comparison, yearly totals, date-range filter |
| Make Prediction | Pick dataset and horizon (1-6); genuine future forecast with measured error ranges; separate one-step backtest vs actual with an in-sample/held-out label; CSV export |
| Model Comparison | Sortable table of all 6 models (marks which have saved weights), RMSE / MAE / R2 / accuracy, charts, error-by-horizon, actual vs predicted (any model), residuals |
| Training and Validation | Training facts and accuracy, learning curves (training vs validation loss), test results, train vs validation vs test accuracy with an overfitting check, 5-fold time-series cross-validation (per fold and mean ± std), residual analysis, tuning trials, and a validation panel with Passed / Warning / Failed / Not evaluated checks |
| Confusion Matrix | Derived low/normal/high matrix, accuracy, precision/recall/F1, majority-class baseline, configurable quantiles |
| Prediction History, Reports | Saved predictions, details, CSV downloads (model metrics, walk-forward predictions, validation summary) |
| Chatbot | The original chatbot UI, unchanged, at `/chatbot` |
| Profile and Settings | Account, password change, metric guide |
| Admin Dashboard (admins only) | Overview counts, dataset management, model info and metric re-evaluation, prediction history/export, user roles/activation, audit log |

## Project layout (new items marked +)

```
api/main.py                  original FastAPI app (unchanged)
api/dashboard_app.py       + entry point: wraps the original app and adds the dashboard API
dashboard/                 + auth.py, config.py, db.py, routes.py, service.py, validation.py, create_admin.py
src/water_forecast/          original model, data, training, inference code (unchanged)
chatbot.py, predict.py, run_*.py   original scripts (unchanged)
outputs/kharghar_*/          original trained runs (model.pt, scaler, metrics, predictions) (unchanged)
data/                        original datasets (unchanged)
frontend/src/app/app.*       original chatbot component (unchanged)
frontend/src/app/dash/     + dashboard pages, services, styles
tests/                       original tests + test_dashboard_api.py (new) + conftest.py (new)
docs/                      + MODEL_PRESERVATION_REPORT.md, TESTING_REPORT.md, screenshots/, original_sha256_baseline.txt
requirements-dashboard.txt + .env.example +
```

## Requirements

Python 3.10+ (tested on 3.12), Node.js 22.22.3+ or 24.15+ (Angular 22 CLI requirement), npm.
Python packages: see `requirements.txt` (original) and `requirements-dashboard.txt` (adds openpyxl, bcrypt, PyJWT, python-multipart, httpx).
Tested with torch 2.14.1, fastapi 0.146, pandas 3.0.2, scikit-learn 1.8.0, numpy 2.4.4, bcrypt 5.0, PyJWT 2.7, Angular 22.2.

## Install

Windows PowerShell (the project's original environment):

```powershell
cd path\to\urban-water-cnn-lstm-am
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dashboard.txt
cd frontend
npm ci
cd ..
```

macOS / Linux:

```bash
cd urban-water-cnn-lstm-am
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dashboard.txt
(cd frontend && npm ci)
```

## Configure

Everything has a working default; see `.env.example` for the optional variables (`WATER_JWT_SECRET`, `WATER_DATA_STORE`,
`WATER_RUN_ID`, `WATER_MAX_UPLOAD_MB`, ...). Set them in your shell, e.g. PowerShell `$env:WATER_JWT_SECRET = "a-long-random-string"`.
If you do not set `WATER_JWT_SECRET`, a random key is generated once into `instance/secret.key`. The chatbot needs no API key.

## Create the first administrator

Public sign-up only ever creates normal users. Create an admin from the command line (the password is prompted for, or read
from `WATER_ADMIN_PASSWORD`; it must be 10+ characters with letters and digits and is stored only as a bcrypt hash):

```powershell
python -m dashboard.create_admin --username admin
```

Later, an admin can promote or deactivate other users on the Admin Dashboard > Users tab. Forgotten admin password:
`python -m dashboard.create_admin --username admin --reset-password`.

## Run

**Option A - one server (recommended).** Build the frontend once, then start the backend; it serves the dashboard too.

```powershell
cd frontend; npx ng build; cd ..
python -m uvicorn api.dashboard_app:app --port 8000
```

Open <http://localhost:8000> (use `localhost`, not `127.0.0.1`, because the original chatbot component calls `http://localhost:8000`).
The first start takes a few seconds while PyTorch loads.

**Option B - development.** Two terminals:

```powershell
python -m uvicorn api.dashboard_app:app --port 8000      # backend
cd frontend; npx ng serve                                   # frontend on http://localhost:4200
```

To run only the original chatbot API as before: `python -m uvicorn api.main:app --port 8000`.

## Use it

1. **Sign in** (or Register) at `/login`. Administrators use `/admin/login`.
2. **Upload a dataset** (Upload Dataset): a CSV/XLSX with columns `period_start, period_end, billing_period, total_qty`
   (see `data/kharghar_period_totals.csv`): one row per two-month period, consecutive, at least 6 periods. Review the validation
   report, then press *Use this dataset* to confirm. Nothing is retrained and no value is altered, renamed or filled in.
3. **Predict** (Make Prediction): choose the dataset and horizon. The page shows the future forecast and, separately, the
   one-step backtest against actual values. Export the result as CSV.
4. **Compare models** on Model Comparison and check the reliability panel on Training and Validation.
5. **Ask the chatbot** via the sidebar (opens the original chat UI; a "Dashboard" button returns you).

## How to read the metrics

- **RMSE / MAE** are in the data's unit (Units); lower is better. RMSE punishes big misses more.
- **MSE** is RMSE squared. **MAPE** is the average error as a percentage of the actual value.
- **Accuracy %** = 100 − MAPE: how close, on average, a forecast is to the actual bill (98% = off by 2% on average). It is a regression accuracy, not a classification hit rate.
- **Training vs testing accuracy**: training accuracy is measured on bills the model learned from, testing accuracy on bills it never saw. A gap above 5 points is flagged as possible overfitting.
- **5-fold time-series cross-validation**: the series is split into 5 consecutive test blocks; each fold trains on every bill before its block and tests on the block. Results are reported per fold and as mean ± std. The data is never shuffled (a shuffled k-fold would let the model learn from future bills).
- **R2**: 1 is perfect; **below 0 means worse than predicting the average of the test bills**. All models are currently below 0.
- **Walk-forward**: each test bill is predicted by a model trained only on earlier bills; this is the honest accuracy.
- **In-sample**: errors on bills the model was trained on; optimistic, shown only with a label.
- **Category accuracy** (Confusion Matrix) is derived from regression outputs and is *not* a native classifier accuracy.

## Run the tests

```powershell
python -m pytest tests -q
```

`tests/test_dashboard_api.py` exercises the API against the real saved model; the original tests are unchanged.
Verify the protected originals are unmodified: `python docs/verify_originals.py` (compares against `docs/original_sha256_baseline.txt`).

## Security notes

Passwords: bcrypt. Sessions: signed JWT kept in `sessionStorage` (cleared when the tab closes); the account is re-checked in the
database on every request so deactivation is immediate. Admin routes enforce the role on the server. Sign-in is rate-limited
(5 failures / 15 min per user+IP, in memory). Uploads: extension and size limits, parsed only as CSV/XLSX (no pickle),
stored under random names, per-user access. CSV exports neutralise spreadsheet formulas. The original `/api/chat` endpoints are
unchanged and remain unauthenticated; do not expose the server to the internet without a reverse proxy, HTTPS and review.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach the backend" in the UI | Start `uvicorn api.dashboard_app:app --port 8000`; open the UI at `localhost` |
| 503 "Model unavailable" | `outputs/` has no run with `model_bundle.json`/`model.pt`; restore it or set `WATER_RUN_ID` / `WATER_OUTPUTS_DIR` |
| Chatbot page shows an error | The chatbot calls `http://localhost:8000/api`; use `localhost` and make sure the backend is running |
| Upload rejected | Read the reasons shown; columns are never renamed or auto-fixed |
| `ng` says Node is too old | Install Node 22.22.3+ or 24.15+ |
| `ng build` fails offline | The production build inlines Google Fonts and needs internet access; use `ng build --configuration development` offline |
| Locked out after 5 wrong passwords | Wait 15 minutes or restart the backend |

---

# Original project documentation

# Urban Water Supply Forecasting with CNN-LSTM-Attention

An end-to-end, reproducible implementation of the Stage-I project: daily urban water-supply forecasting with a 1-D CNN, LSTM, and additive attention mechanism. It includes LSTM, CNN-LSTM, and LSTM-Attention ablations, naive baselines, leak-free walk-forward evaluation, and equal Bayesian hyperparameter search for every model.

## What is included

- Causal data cleaning: daily reindexing with last-value gap filling, outlier limits fitted on training days only, and backward-looking (causal) Savitzky-Golay smoothing of model inputs.
- Walk-forward (rolling-origin) evaluation with train-only scaling and early stopping on training windows.
- CNN-LSTM-Attention, LSTM, CNN-LSTM, and LSTM-Attention models, each tuned with the same Optuna TPE budget.
- Naive baselines (last value, seasonal naive) scored on the same test points.
- MAE, MSE, RMSE, R-squared, relMAE and MASE reports; actual-vs-forecast, scatter, model-comparison and (Kharghar) horizon plots.
- A deterministic synthetic daily-supply generator so the project runs immediately. Replace it with a real CSV when available.

## Evaluation protocol

These rules keep reported accuracy honest; `tests/test_evaluation.py` checks them.

- **No test data in training decisions.** Early stopping uses the most recent *training* windows; the model is then refit on all training windows for the chosen number of epochs. Test points are only predicted, once.
- **Equal tuning.** All four variants get the same number of Optuna trials, on data that ends before the first test point, and each searches only the settings it uses. Each trial is scored by `--cv-folds` (default 5) time-series cross-validation.
- **Time-series k-fold cross-validation.** Kharghar runs also evaluate every variant with `--cv-folds` expanding-window folds (train on all earlier bills, test on the next block), recording train, validation (early-stopping windows) and test scores per fold. The hyperparameters were tuned on bills that overlap the later CV test blocks, so the walk-forward result remains the primary unbiased test; the CV shows how stable the models are across periods.
- **Walk-forward testing.** Kharghar: each of the last `--test-periods` bills is forecast by models trained only on earlier bills. Daily: the last `--test-fraction` of days is split into `--folds` consecutive blocks, each forecast by models trained only on earlier days. Every neural model is repeated over `--runs` seeds; `mae_run_std` shows the seed-to-seed spread.
- **No future information in preprocessing.** Gaps take the previous value, clip limits come from training data, smoothing is causal and only applied to model inputs. Forecasts are scored against raw observed values.
- **Baselines.** `rel_mae` is MAE divided by the MAE of repeating the last value on the same test points (below 1 = better). MASE uses the in-sample naive error as its scale; on Kharghar that scale is inflated by the `DEC-JAN2024` spike, so prefer `rel_mae` there.
- **Deployment.** The chatbot and `predict.py` use the neural variant with the lowest walk-forward MAE, retrained on all data with the same procedure. The run's `model_bundle.json` records which variant and how it compared with the best baseline.

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\atul\Documents\Development\urban-water-cnn-lstm-am
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_experiment.py --fast
```

`--fast` verifies the full workflow quickly. For the planned Stage-II experiment, remove `--fast`; the default allows up to 100 epochs, runs 12 Bayesian trials per model variant, and repeats each model over three seeds in three walk-forward test blocks.

## Using a real dataset

Provide a CSV with a `date` column and a daily target column (default: `supply_m3`):

```powershell
python run_experiment.py --data data\your_daily_supply.csv --date-column date --target-column supply_m3
```

The data must represent daily total water supply in cubic metres. Keep the CSV outside `outputs/`; all generated artifacts are written below `outputs/<timestamp>/`.

## Kharghar billing-period data

The Kharghar Excel files contain customer billing quantities for two-month periods, not daily supply readings. Prepare one combined total per complete billing period, then train the same CNN/LSTM/attention model family with one time step per billing period:

```powershell
python prepare_kharghar_totals.py
python run_kharghar_experiment.py --fast
python run_kharghar_experiment.py
python chatbot.py
```

The default Kharghar run tunes each variant with 10 trials, tests the last 8 bills walk-forward with 3 seeds, and also scores forecasts 2 and 3 bills ahead (`--horizon`). After training, `python chatbot.py` automatically loads the latest run. You can also pass `--run outputs\\kharghar_<timestamp>` to choose a specific run. Ask `next billing cycle`, `forecast 2 periods`, `average last 6 periods`, `highest period`, `accuracy`, or `help`.

The run saves a forecast for the next billing period in `forecast_next_period.csv`, alongside walk-forward metrics and plots under `outputs/kharghar_<timestamp>/`. The default six-step window uses about one year of history. There are only 32 complete periods, so treat the forecast as exploratory. The physical unit behind `Units` has not been confirmed; one period (`DEC-JAN2024`) also has a notably larger total than the others and should be checked before relying on its forecast.

## Expected output

Each run creates `metrics.csv` (neural models and baselines, next step), `walk_forward_predictions.csv`, `trial_history.csv` (all variants), `best_hyperparameters.json` (per variant), `config.json`, `model_bundle.json`, and plots in an isolated timestamped folder; Kharghar runs add `horizon_metrics.csv` and `horizon_comparison.png`, plus the validation outputs: `cv_folds.csv` / `cv_summary.csv` / `cv_folds.png` (k-fold cross-validation), `train_test_accuracy.csv` / `.png` (training vs validation vs testing accuracy), `learning_curves.csv` / `.png` (loss per epoch) and `residual_diagnostics.csv` / `residuals.png`. The console ends with a verdict comparing the best neural model with the best baseline. Results on the generated synthetic dataset validate execution only; they are not evidence of real-world forecasting accuracy.

## Design notes

The implementation follows the supplied project brief: CNN extracts local patterns from a sliding window, LSTM models the sequence, and attention weights informative time steps before a one-day-ahead forecast. The Savitzky-Golay step required by the project proposal is implemented as a causal (trailing-window) filter so that it is usable in live forecasting; set `--disable-smoothing` to evaluate raw operational readings. Runs created before the walk-forward protocol used early stopping on the test split and should not be quoted.

## Validation

```powershell
python -m pytest -q
```

## Sources

- Zhao, Y. et al. (2023). *Urban Water Supply Forecasting Based on CNN-LSTM-AM Spatiotemporal Deep Learning Model*, IEEE Access, 11, 144204-144212.
- Supplied `Project_Presentation_Stage-I_CNN-LSTM-AM (2).pptx` and `CNN-LSTM-AM_Presentation_Script (1).pdf` (requirements source only).
