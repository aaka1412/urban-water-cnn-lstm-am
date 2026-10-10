# Project plan

## Completed: foundation and reproducible prototype

1. Convert the Stage-I design into a testable CNN-LSTM-Attention codebase.
2. Implement cleaning, daily regularization, Savitzky-Golay smoothing, train-only min-max scaling, and sliding windows.
3. Build the proposed model and its LSTM, CNN-LSTM, and LSTM-Attention ablations.
4. Add Bayesian (TPE) hyperparameter tuning with chronological cross-validation.
5. Produce metrics and visual reports from a held-out chronological test set.
6. Verify the full path on a synthetic daily water-supply series.

## Next: real-data experiment

1. Obtain a documented daily urban-supply CSV with `date` and `supply_m3` columns.
2. Run data-quality review: missing days, units, duplicate readings, unusual operational events, and date coverage.
3. Run the default experiment (100 epochs, 12 trials, 3 repeated seeds) on the real CSV.
4. Record environment, dataset provenance, time span, all hyperparameters, and random seeds.
5. Compare the four models by averaged MAE, MSE, RMSE, and R-squared; inspect peak-day errors separately.

## Final: research reporting and deployment readiness

1. Test a recent out-of-time period to mimic future forecasting.
2. Decide whether smoothing belongs in the operational pipeline; avoid using future observations in a live setting.
3. Add optional external features only if their forecast-time availability is documented (weather, holiday, population proxies).
4. Prepare the final results table, forecast/scatter figures, limitations, and reproducibility appendix.
