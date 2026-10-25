# Testing Report

Environment: Linux sandbox, Python 3.12, CPU-only PyTorch 2.14.1, Node 24.22 (Angular CLI needs >= 22.22.3), headless Chromium via puppeteer-core.
Date of run: 2026-10-19.

## Summary
| Suite | Result |
|---|---|
| Original tests (`tests/test_data.py`, `tests/test_evaluation.py`) | 11 passed |
| New API tests (`tests/test_dashboard_api.py`) | 37 passed |
| Full run `pytest tests` | **48 passed**, 0 failed |
| Browser end-to-end walkthrough (headless Chromium, real server, real model) | **38 of 38 checks passed** (`docs/ui_e2e_results.json`) |
| Protected-file hashes | 77 identical, 0 changed, 0 missing |
| Angular development build | OK |
| Angular production build | Compiles; one budget warning (initial bundle 549 kB vs 500 kB, about 150 kB gzipped). Run with Google-Fonts inlining switched off because the sandbox is offline; your config was restored afterwards |

## Requirement-by-requirement
| # | Test (brief section 18) | Evidence | Result |
|---|---|---|---|
| 1 | Application startup | uvicorn starts, `/api/health` 200, SPA served | pass |
| 2 | User dashboard loading | browser: 13 cards, charts render, all 12 sidebar pages | pass |
| 3 | Admin login and logout | admin login: API + browser; logout: browser (user session) and API audit entry | pass |
| 4 | Unauthorized access to admin routes | API: 401 without token, 403 for user on every `/api/admin/*`; browser: user bounced from `/admin`; user cannot use admin login | pass |
| 5 | Dataset upload | CSV and XLSX | pass |
| 6 | Invalid file rejection | 7 invalid-content cases, wrong extension, empty, oversize, fake xlsx; rejected files not stored | pass |
| 7 | Dataset preview | API + browser | pass |
| 8 | Missing-value and schema validation | missing column, short history, gap, negative, non-numeric, bad date, duplicate | pass |
| 9 | Prediction with the original model | forecast equals saved `forecast_next_period.csv` (abs diff < 0.01) | pass |
| 10 | RMSE/MAE/MSE/R2 | catalog values equal `metrics.csv` (MSE = RMSE squared) | pass |
| 11 | Training/testing/validation display | loss history reported as not recorded; validation panel statuses checked | pass |
| 12 | Actual-versus-predicted graphs | browser: charts present on Compare and Predict | pass (visual check of screenshots) |
| 13 | Confusion matrix | accuracy = diagonal / total; thresholds use pre-test bills only; invalid quantiles rejected; non-informative accuracy flagged | pass |
| 14 | Prediction history | user and admin views, filters | pass |
| 15 | CSV export | prediction, metrics, validation, audit; formula-injection neutralised | pass |
| 16 | Chatbot integration | original `/api/chat` unchanged; UI under `/chatbot` answers a forecast question; accuracy answer matches the dashboard | pass |
| 17 | Admin dataset management | list, preview, validation details, delete with confirmation, original protected (403) | pass |
| 18 | Missing-model handling | outputs folder empty -> HTTP 503 "Model unavailable"; no fabricated results | pass |
| - | Model preservation | model/scaler/bundle hashes unchanged after uploads and predictions | pass |
| - | Uploads never retrain | no training code is imported or called by `dashboard/` | pass (by inspection and hash test) |
| - | Audit log | login success/failure, uploads, rejections, deletions, predictions, evaluations, user changes; no passwords or tokens | pass |
| - | Responsive layout | 390 px wide: sidebar collapses, no page-level horizontal overflow on Overview and Model Comparison | pass |

## Bugs found and fixed during testing
1. Charts that load data after the page renders (overview, actual vs predicted, residuals) showed axes but no data in screenshots. Chart animation is now off (also matching the "no unnecessary animation" requirement); verified afterwards.
2. The confusion matrix headlined 100% accuracy while all test bills belong to one class. Added a majority-class baseline and a "not informative" warning.
3. Cramped header on phones; the long title is hidden below 600 px.

## Not tested / limitations
- No Angular unit tests (`ng test`) exist or were written; UI testing was the scripted browser run above, in headless Chromium only (not Firefox/Safari, no real phone, no accessibility audit).
- The commands for Windows PowerShell in the README were not executed on Windows (the sandbox is Linux); the Python and npm steps they wrap were.
- Production Angular build with font inlining needs internet access; not possible offline here.
- No load, concurrency or penetration testing. The login rate limit is in memory (reset on restart). Tokens live in `sessionStorage` (readable by scripts on the page); there is no HTTPS in the dev setup.
- `.xls` (legacy Excel) uploads are intentionally refused.
- The original `/api/chat` endpoints stay unauthenticated, as before.
- Cross-validation, retraining and regional analysis are not available (see the preservation report).
- Windows line endings: original files keep their CRLF endings; new files use LF.
