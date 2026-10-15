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
- **Equal tuning.** All four variants get the same number of Optuna trials, on data that ends before the first test point, and each searches only the settings it uses.
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

The run saves a forecast for the next billing period in `forecast_next_period.csv`, alongside walk-forward metrics and plots under `outputs/kharghar_<timestamp>/`. The default six-step window uses about one year of history. There are only 32 complete periods, so treat the forecast as exploratory. `QTY` units have not been confirmed; one period (`DEC-JAN2024`) also has a notably larger total than the others and should be checked before relying on its forecast.

## Expected output

Each run creates `metrics.csv` (neural models and baselines, next step), `walk_forward_predictions.csv`, `trial_history.csv` (all variants), `best_hyperparameters.json` (per variant), `config.json`, `model_bundle.json`, and plots in an isolated timestamped folder; Kharghar runs add `horizon_metrics.csv` and `horizon_comparison.png`. The console ends with a verdict comparing the best neural model with the best baseline. Results on the generated synthetic dataset validate execution only; they are not evidence of real-world forecasting accuracy.

## Design notes

The implementation follows the supplied project brief: CNN extracts local patterns from a sliding window, LSTM models the sequence, and attention weights informative time steps before a one-day-ahead forecast. The Savitzky-Golay step required by the project proposal is implemented as a causal (trailing-window) filter so that it is usable in live forecasting; set `--disable-smoothing` to evaluate raw operational readings. Runs created before the walk-forward protocol used early stopping on the test split and should not be quoted.

## Validation

```powershell
python -m pytest -q
```

## Sources

- Zhao, Y. et al. (2023). *Urban Water Supply Forecasting Based on CNN-LSTM-AM Spatiotemporal Deep Learning Model*, IEEE Access, 11, 144204-144212.
- Supplied `Project_Presentation_Stage-I_CNN-LSTM-AM (2).pptx` and `CNN-LSTM-AM_Presentation_Script (1).pdf` (requirements source only).
