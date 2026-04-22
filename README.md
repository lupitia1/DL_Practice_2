# DL Practice 2 — Electricity Load Forecasting (RNNs)

Predict New York State total electricity load **3 hours ahead** using NYISO hourly data (2021–2025).

## Notebook structure

| Section | What it does |
|---|---|
| 1–5 | **Data pipeline** — load CSV, temporal split (train 21-23 / val 24 / test 25), z-score normalization, sliding-window sequences (48h input → 3h ahead target) |
| 6–7 | **Simple baselines** — last-value and daily (same hour previous day) |
| 8 | **Neural baseline** — Conv1D + Dense |
| 9 | **SimpleRNN** — two experiments (1 layer vs 2 stacked + dropout) |
| 10 | **LSTM** — two experiments (same structure) |
| 11 | **GRU** — two experiments (same structure) |
| 12 | **Final comparison** — summary table, bar chart, discussion |

All neural models share: Adam (lr=1e-3), batch 64, 50 epochs, early stopping (patience 5). Metric: **denormalized MAE (MW)**.

## Files

- `P2_Electricity_Load_Forecasting.ipynb` — main notebook
- `nyiso_hourly_load.csv` — dataset (11 zones + total load, hourly)
- `download_data_nyiso_practice_rnn.py` — data download script (reference only)
