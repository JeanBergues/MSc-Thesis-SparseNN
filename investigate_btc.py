import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('btcusd_full.csv')

date_col = pd.to_datetime(data.date)

for col in data.columns[1:]:
    sns.lineplot(x=date_col, y=data[col])
    plt.title(col)
    plt.show()