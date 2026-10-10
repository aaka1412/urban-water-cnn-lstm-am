# Urban Water Supply Forecasting with CNN-LSTM-Attention

An end-to-end, reproducible implementation of the Stage-I project: daily urban water-supply forecasting with a 1-D CNN, LSTM, and additive attention mechanism. It includes LSTM, CNN-LSTM, and LSTM-Attention baselines, chronological evaluation, and Bayesian hyperparameter search.

## What is included

- Data cleaning, daily reindexing, interpolation, Savitzky-Golay smoothing, and train-only min-max scaling.
- Sliding-window forecasting with an 80:20 chronological holdout split.
- CNN-LSTM-Attention, LSTM, CNN-LSTM, and LSTM-Attention models.
- Optuna TPE (Bayesian) tuning with time-series cross-validation.
- MAE, MSE, RMSE, and R-squared reports; actual-vs-forecast, scatter, and model-comparison plots.
- A deterministic synthetic daily-supply generator so the project runs immediately. Replace it with a real CSV when available.

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\atul\Documents\Development\urban-water-cnn-lstm-am
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_experiment.py --fast
```

`--fast` verifies the full workflow quickly. For the planned Stage-II experiment, remove `--fast`; the default trains for 100 epochs, runs 12 Bayesian trials, and repeats each model three times.

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
python run_kharghar_experiment.py --trials 3 --epochs 50 --runs 1
python chatbot.py
```

After training, `python chatbot.py` automatically loads the latest run. You can also pass `--run outputs\\kharghar_<timestamp>` to choose a specific run. Ask `next billing cycle`, `forecast 2 periods`, `average last 6 periods`, `highest period`, `accuracy`, or `help`.

The full run saves a forecast for the next billing period in `forecast_next_period.csv`, alongside holdout metrics and plots under `outputs/kharghar_<timestamp>/`. The default six-step window uses about one year of history. There are only 32 complete periods, so treat the forecast as exploratory. `QTY` units have not been confirmed; one period (`DEC-JAN2024`) also has a notably larger total than the others and should be checked before relying on its forecast.

## Expected output

Each run creates `metrics.csv`, `trial_history.csv`, `config.json`, `best_hyperparameters.json`, and plots in an isolated timestamped folder. Results on the generated synthetic dataset validate execution only; they are not evidence of real-world forecasting accuracy.

## Design notes

The implementation follows the supplied project brief: CNN extracts local patterns from a sliding window, LSTM models the sequence, and attention weights informative time steps before a one-day-ahead forecast. The scaler is fitted solely on the training portion to prevent leakage. The optional Savitzky-Golay step is applied before modeling because it is required by the project proposal; set `--disable-smoothing` to evaluate raw operational readings.

## Validation

```powershell
python -m pytest -q
```

## Sources

- Zhao, Y. et al. (2023). *Urban Water Supply Forecasting Based on CNN-LSTM-AM Spatiotemporal Deep Learning Model*, IEEE Access, 11, 144204-144212.
- Supplied `Project_Presentation_Stage-I_CNN-LSTM-AM (2).pptx` and `CNN-LSTM-AM_Presentation_Script (1).pdf` (requirements source only).
