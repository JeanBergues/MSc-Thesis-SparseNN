
This ReadMe file will describe all the steps necessary to obtain the exact results used in the Thesis.
Before recreating any experiments, you first have to prepare the data.

1) Put the btcusd_full.csv file in the working directory
2) Run impute_btc_data.py, which will produce a new file named imputed_btc_data.csv where all missing 5-minute intervals have been filled in
3) Run aggregate_btc_data.py, which will produce two new files: level_btc_day.csv and level_btc_hour.csv. These files contain the data aggregated to their respective time frequencies.
4) Run pct_diff_btc_data.py, which produces pct_btc_day.csv and pct_btc_hour.csv, where the data has been transformed into simple returns

Now you can reproduce the experiments.

Reproducing the competing models:
ARIMA and ARIMAX: Run ARIMA.R
GARCH: Run GARCH.R
MIDAS: Run MIDAS.R
MIDASX: Run MIDASX.R

These models will write their forecasts to the folder final_R_forecasts