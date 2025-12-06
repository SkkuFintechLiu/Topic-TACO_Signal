import pandas as pd
import numpy as np

def get_yf_returns(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=['date', 'asset_ret'])
    col = 'Close' if 'Close' in df.columns else 'Adj Close'
    df = df[[col]].rename(columns={col: 'px'})
    df['asset_ret'] = np.log(df['px']).diff()
    df = df.dropna()
    df['date'] = pd.to_datetime(df.index)
    return df[['date', 'asset_ret']]
