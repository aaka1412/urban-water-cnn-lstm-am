# Model-Preservation and Integration Report

## 1. What was protected
77 original files were hashed (SHA-256) before any change: every `.pt`, `.json`, `.csv`, `.py` under `outputs/`, `data/`, `src/`, `api/`, `tests/`,
plus `chatbot.py`, `predict.py`, `run_kharghar_experiment.py`, `run_experiment.py`, `prepare_kharghar_totals.py`, `validate_data.py`, `requirements.txt`.
The list is `docs/original_sha256_baseline.txt`; `python docs/verify_originals.py` re-checks it.

**Result after the full upgrade, all tests and the browser run: 77 identical, 0 changed, 0 missing.**

Not touched: model architecture (`src/water_forecast/models.py`), training and tuning code, preprocessing, scaler, `model.pt` of all five runs,
datasets, `chatbot.py`, `api/main.py`, the original Angular chat component (`app.ts/html/scss`, `chart.component.ts`, `api.service.ts`).

## 2. How the dashboard calls the models
| Need | Original code reused |
|---|---|
| Load weights + scaler | `water_forecast.inference.load_bundle` (`torch.load(..., weights_only=True)`) |
| Forecast / inverse scaling | `water_forecast.inference.forecast_future` (the same function the chatbot and `predict.py` use) |
| Latest run | `water_forecast.inference.find_latest_run` |
| Metrics and test predictions | Read from the run's saved `metrics.csv`, `walk_forward_predictions.csv`, `horizon_metrics.csv`, `trial_history.csv` |

New code lives only in `dashboard/`, `api/dashboard_app.py` (wraps the original `app`), `frontend/src/app/dash/`, `frontend/src/dash.scss`, plus `app.config.ts`/`main.ts`/`angular.json` (bootstrapping and routing; the original chat component is now the `/chatbot` route).

Not done by the dashboard: no training, no tuning, no weight writes, no scaler refits. An upload only stores a file and runs validation; prediction only runs inference.
Tests assert that `model.pt`, `scaler.json` and `model_bundle.json` hash identically before and after uploads and predictions.

## 3. Models found (exact names from the source)
`cnn_lstm_attention`, `cnn_lstm`, `lstm_attention`, `lstm` (class `Forecaster`), baselines `naive_last_value`, `seasonal_naive_6`.
This is four neural variants plus two baselines, not three models; all six are shown and none removed.

| Model | Saved weights | Can predict new data |
|---|---|---|
| cnn_lstm_attention (deployed; 51,842 parameters) | yes (`model.pt`, identical in all five run folders) | yes |
| cnn_lstm, lstm_attention, lstm | no (trained inside walk-forward folds, not saved) | no, evaluation results only |
| naive_last_value, seasonal_naive_6 | n/a (rule-based) | not exposed |

## 4. Verified behaviour of the preserved model
- Reloaded model forecast for AUG-SEP2026 = 4,318,689.71, identical to the saved `forecast_next_period.csv` (difference 0.0).
- Metrics recomputed from the saved walk-forward predictions match `metrics.csv` to about 1e-10 for all six models.

## 5. Actual results (walk-forward, last 8 bills, unit QTY, unit not confirmed)
| Model | RMSE | MAE | MSE | R2 | MAPE |
|---|---|---|---|---|---|
| CNN-LSTM + Attention (deployed) | 74,822.48 | 59,970.74 | 5.598e9 | -1.446 | 1.41% |
| Repeat last value | 76,508.85 | 66,814.88 | 5.854e9 | -1.558 | 1.56% |
| CNN-LSTM | 80,069.71 | 73,666.03 | 6.411e9 | -1.802 | 1.73% |
| Same bill last year | 83,333.64 | 71,934.38 | 6.944e9 | -2.035 | 1.69% |
| LSTM + Attention | 91,621.05 | 78,004.33 | 8.394e9 | -2.668 | 1.84% |
| LSTM | 98,355.33 | 81,099.34 | 9.674e9 | -3.227 | 1.91% |

Deployed model vs best baseline: MAE 10% lower, RMSE 2.2% lower (small margin). The 8 test bills are all in the high-consumption regime after the DEC-JAN2024 spike, so R2 (relative to the test mean) is harshly negative for every model. These are exploratory numbers.

Further findings the dashboard surfaces rather than hides:
- Fit on the training bills (in-sample, saved weights): RMSE 515,973, MAE 314,558, R2 0.004. The model largely predicts a near-constant level; its one-step forecasts for the next three bills are 4,318,690 / 4,318,164 / 4,319,336 (almost flat).
- No per-epoch loss history was saved, so loss curves are not drawn; no validation RMSE/MAE were saved either.
- The five run folders contain byte-identical weights (same seed/configuration re-run).

## 6. Requested items that could not be implemented as asked, and what was done instead
| Request | Status |
|---|---|
| Three-model comparison | Six shown (four neural, two baselines) |
| Predict with every model | Only the deployed model; others labelled "Evaluation only" |
| Training/validation loss curves | Not available; explicit notice; Optuna tuning scores shown instead |
| Training/testing accuracy percentage | Not applicable to regression; RMSE/MAE/R2 shown, with explanation |
| Native confusion matrix | No classifier exists; derived low/normal/high matrix labelled as such. With the current data all test bills are "high", so accuracy equals the majority-class baseline and the page says it is not informative |
| Region filter | Data has a single combined total per period |
| Retraining workflow | Deliberately not exposed (would create new weights) |
| Chatbot reading dashboard state/uploaded data | The original chatbot is unchanged and answers from the run's saved artifacts; it does not see uploaded datasets. Its numbers match the dashboard (MAE 59,971, RMSE 74,822). It does not mention the negative R2 |
