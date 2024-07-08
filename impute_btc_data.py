import pandas as pd

# Read in data
data = pd.read_csv('btcusd_full.csv', parse_dates=['date'], index_col=['date'])

# Grab datetimes for first and last row
start_time = data.index[0]
end_time = data.index[-1]

# Generate a date range that includes all 5 minute intervals that should be in the data
date_range = pd.date_range(start_time, end_time, freq='5min')

dict_list = []
last_valid_t = 0

for t in date_range:
    # Try to access the 5 minute interval. If it does not exist in the data, Pandas will throw a KeyError
    try:
        t_df = data.loc[t]
        last_valid_t = t
    except KeyError:
        # If the 5 minute interval is missing, copy over the values of the last known valid interval
        # print(f"{t}")
        floored_date = t.floor('d')
        new_dict = {}
        new_dict['ddate'] = floored_date
        new_dict['date'] = t
        dict_list.append(new_dict)
        data.loc[t] = data.loc[last_valid_t].copy()

# Store to dataframe and print out table of missing values per month
df = pd.DataFrame(dict_list)
df['yearmonth'] = df.apply(lambda x: f"{x.ddate.year}-{x.ddate.month:02}", axis=1)
print(df.yearmonth.value_counts().sort_index().to_latex())

# Save the imputed data
new_df = data.sort_index()
new_df.to_csv('imputed_btc_data.csv')