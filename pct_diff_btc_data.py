import pandas as pd
import numpy as np

def main(name):
    full_df = pd.read_csv(f'level_btc_{name}.csv', parse_dates=['date', 'daydate'], index_col=0)
    pct_df = full_df.drop(['date', 'daydate'], axis=1).pct_change(periods=1).drop(0, axis=0) * 100
    log_df = full_df.drop(['date', 'daydate'], axis=1).drop(0, axis=0).apply(np.log)
    log_df.columns = list(map(lambda c: 'log_' + c, log_df.columns))

    #pct_df['date'] = full_df.date[1:]
    pct_df.to_csv(f'pct_btc_{name}.csv', index=False)
    pd.concat([pct_df, log_df], axis=1).to_csv(f'com_btc_{name}.csv', index=False)

if __name__ == '__main__':
    main('day')
    print("Finished differencing day level")
    main('hour')
    print("Finished differencing hour level")