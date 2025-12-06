import os
import pandas as pd

def connect():
    import wrds
    u = os.environ.get('WRDS_USERNAME')
    p = os.environ.get('WRDS_PASSWORD')
    return wrds.Connection(wrds_username=u, wrds_password=p)

def get_market_returns(conn, start_date: str, end_date: str) -> pd.DataFrame:
    sql = f"select date, vwretd from crsp.dsi where date between '{start_date}' and '{end_date}'"
    df = conn.raw_sql(sql)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'vwretd': 'market_ret'})
    return df[['date', 'market_ret']]

TICKER_MAP = {
    'META': 'FB'
}

def resolve_permnos(conn, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    t = TICKER_MAP.get(ticker, ticker)
    sql = f"select permno, ticker, namedt, nameenddt from crsp.stocknames where ticker = '{t}'"
    sn = conn.raw_sql(sql)
    if sn.empty:
        return pd.DataFrame(columns=['permno'])
    sn['namedt'] = pd.to_datetime(sn['namedt'])
    sn['nameenddt'] = pd.to_datetime(sn['nameenddt'])
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    overlap = sn[(sn['nameenddt'] >= s) & (sn['namedt'] <= e)]
    overlap = overlap.sort_values('namedt')
    return overlap[['permno']].drop_duplicates()

def get_asset_returns_by_permno(conn, permno: int, start_date: str, end_date: str) -> pd.DataFrame:
    sql = f"select date, ret from crsp.dsf where permno = {int(permno)} and date between '{start_date}' and '{end_date}'"
    df = conn.raw_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=['date', 'asset_ret'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'ret': 'asset_ret'})
    return df[['date', 'asset_ret']]

def get_asset_returns(conn, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    perms = resolve_permnos(conn, ticker, start_date, end_date)
    if not perms.empty:
        dfs = []
        for p in perms['permno'].tolist():
            d = get_asset_returns_by_permno(conn, p, start_date, end_date)
            if not d.empty:
                dfs.append(d)
        if dfs:
            out = pd.concat(dfs, ignore_index=True).sort_values('date')
            out = out.drop_duplicates(subset=['date'])
            return out
    t = TICKER_MAP.get(ticker, ticker)
    sql = f"select date, ret from crsp.dsf where ticker = '{t}' and date between '{start_date}' and '{end_date}'"
    df = conn.raw_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=['date', 'asset_ret'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'ret': 'asset_ret'})
    return df[['date', 'asset_ret']]
