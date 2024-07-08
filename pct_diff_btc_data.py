import pandas as pd

def main(name):
    full_df = pd.read_csv(f'level_btc_{name}.csv', parse_dates=['date', 'daydate'], index_col=0)
    pct_df = full_df.drop(['date', 'daydate'], axis=1).pct_change(periods=1) * 100
    pct_df.drop(0, axis=0).to_csv(f'pct_btc_{name}.csv')

if __name__ == '__main__':
    main('day')
    print("Finished differencing day level")
    main('hour')
    print("Finished differencing hour level")