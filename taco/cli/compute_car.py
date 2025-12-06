import argparse
import os
import pandas as pd
from taco.finance.wrds_data import connect, get_market_returns, get_asset_returns
from taco.finance.yf_data import get_yf_returns
from taco.finance.capm import compute_car_for_events

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--assets', default='SPY')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ev = pd.read_csv(args.events, parse_dates=['peak_date'])
    if ev.empty:
        pd.DataFrame([]).to_csv(args.out, index=False)
        return
    start = pd.Timestamp('2015-01-01')
    end = pd.Timestamp('2021-04-01')
    conn = None
    try:
        conn = connect()
        mrk = get_market_returns(conn, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    except Exception:
        yf_m = get_yf_returns('SPY', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if yf_m.empty:
            pd.DataFrame([]).to_csv(args.out, index=False)
            return
        mrk = yf_m.rename(columns={'asset_ret':'market_ret'})
    sector_map = {
        'NVDA': 'Technology',
        'AAPL': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'PFE': 'Health Care',
        'UNH': 'Health Care',
        'GOOG': 'Communication Services',
        'META': 'Communication Services',
        'BTC-USD': 'Crypto'
    }
    rows = []
    for t in [s.strip() for s in args.assets.split(',') if s.strip()]:
        if t.upper() == 'BTC-USD' or conn is None:
            ast = get_yf_returns('BTC-USD', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        else:
            ast = get_asset_returns(conn, t, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if ast.empty:
            continue
        car = compute_car_for_events(ev, ast, mrk)
        car['asset'] = t
        car['sector'] = sector_map.get(t, '')
        rows.append(car)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame([])
    out.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
