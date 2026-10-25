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

## Done: evaluation methodology review

1. Early stopping moved from the test split to the last training windows (refit on all training windows afterwards).
2. All four variants tuned with the same Optuna budget before the first test point.
3. Causal preprocessing: last-value gap filling, train-only clip limits, trailing Savitzky-Golay on inputs, scoring on raw values.
4. Walk-forward evaluation with repeated seeds instead of one 80:20 split.
5. Naive baselines with relMAE and MASE; Kharghar also reports 2- and 3-bill-ahead errors.

## Final: research reporting and deployment readiness

1. Test a recent out-of-time period to mimic future forecasting.
2. Decide whether causal smoothing helps on real data (compare with `--disable-smoothing`).
3. Add optional external features only if their forecast-time availability is documented (weather, holiday, population proxies).
4. Prepare the final results table, forecast/scatter figures, limitations, and reproducibility appendix.
