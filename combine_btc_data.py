import pandas as pd
import numpy as np

minute_interval = 5
minutes_per_row = 60

# Read in data
data = pd.read_csv('btcusd_full.csv')
value_col_names = data.columns[1:]

newcols = ["date"]
for minute in range(0, minutes_per_row, minute_interval):
    for feature in value_col_names:
        newcols.append(f"{feature}_{minute}")

current_minute = 0
new_data_list = []
current_row = []

for row in data.itertuples():
    # Add date to start of new row
    if current_minute == 0:
        if int(row.date.split(':')[1]) == 0:
            current_row = [row.date] 
        else:
            continue

    # Add values to each column
    for feature in value_col_names:
        current_row.append(getattr(row, feature))

    # Update to new minute interval
    if current_minute == minutes_per_row - minute_interval:
        current_minute = 0
        new_data_list.append(current_row)
        if len(new_data_list) % 100 == 0: print(f"Finished row {len(new_data_list)}")
    else:
        current_minute += 5

new_data = pd.DataFrame(data=new_data_list, columns=newcols)
new_data.to_csv("btc_data_hourly.csv")