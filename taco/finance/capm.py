import pandas as pd
import numpy as np

def estimate_capm(asset: pd.Series, market: pd.Series):
    m = market.values
    a = asset.values
    v = np.var(m)
    c = np.cov(a, m)[0, 1]
    b = c / v if v > 0 else 0.0
    alpha = a.mean() - b * m.mean()
    return alpha, b

def compute_car_for_events(events: pd.DataFrame, asset_df: pd.DataFrame, market_df: pd.DataFrame, windows=None) -> pd.DataFrame:
    if windows is None:
        windows = [[-5, -2], [-1, 1], [2, 5]]
    asset = asset_df.set_index('date')['asset_ret']
    market = market_df.set_index('date')['market_ret']
    asset.index = pd.to_datetime(asset.index).tz_localize(None)
    market.index = pd.to_datetime(market.index).tz_localize(None)
    m_idx = market.index.sort_values()
    rows = []
    for _, r in events.iterrows():
        d = pd.to_datetime(r['peak_date'])
        try:
            d = d.tz_localize(None)
        except Exception:
            pass
        d = d.normalize()
        pos = m_idx.searchsorted(d)
        if pos >= len(m_idx):
            continue
        est_start_pos = max(0, pos - 250)
        est_end_pos = max(0, min(len(m_idx) - 1, pos - 60))
        if est_end_pos <= est_start_pos:
            continue
        est_idx = m_idx[est_start_pos:est_end_pos + 1]
        x = market.loc[est_idx]
        y = asset.loc[est_idx].dropna()
        x = x.loc[y.index]
        if len(y) < 30 or len(x) < 30:
            continue
        alpha, beta = estimate_capm(y, x)
        for w in windows:
            s_pos = max(0, min(len(m_idx) - 1, pos + w[0]))
            e_pos = max(0, min(len(m_idx) - 1, pos + w[1]))
            if e_pos < s_pos:
                s_pos, e_pos = e_pos, s_pos
            idx2 = m_idx[s_pos:e_pos + 1]
            xm = market.loc[idx2]
            ya = asset.loc[idx2].dropna()
            xm = xm.loc[ya.index]
            if len(idx2) == 0:
                continue
            ar = ya - (alpha + beta * xm)
            car = float(np.nansum(ar.values))
            rows.append({
                'event_date': d,
                'window': f'{w[0]}:{w[1]}',
                'CAR': float(car)
            })
    return pd.DataFrame(rows)
