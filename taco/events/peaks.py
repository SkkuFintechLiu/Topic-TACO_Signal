import pandas as pd
import numpy as np

def sliding_window_peak_detection(df: pd.DataFrame, date_col: str = 'date', score_col: str = 'taco_signal', window_days: int = 15) -> pd.DataFrame:
    u = sorted(pd.to_datetime(df[date_col]).dt.normalize().unique())
    a = df[score_col].abs().values
    amin = float(np.nanmin(a)) if len(a) else 0.0
    amax = float(np.nanmax(a)) if len(a) else 1.0
    rng = amax - amin if amax > amin else 1.0
    records = []
    for d in u:
        ws = d - pd.Timedelta(days=window_days)
        we = d + pd.Timedelta(days=window_days)
        w = df[(pd.to_datetime(df[date_col]).dt.normalize() >= ws) & (pd.to_datetime(df[date_col]).dt.normalize() <= we)]
        wp = w[w[score_col] > 0]
        wn = w[w[score_col] < 0]
        pos1 = wp[score_col].nlargest(1)
        pos2 = wp[score_col].nlargest(2)
        neg1 = wn[score_col].nsmallest(1)
        neg2 = wn[score_col].nsmallest(2)
        pos_idx = pos1.index[0] if not pos1.empty else None
        pos2_idx = pos2.index[1] if len(pos2) == 2 else None
        neg_idx = neg1.index[0] if not neg1.empty else None
        neg2_idx = neg2.index[1] if len(neg2) == 2 else None
        conflict = 0
        if pos_idx is not None and neg_idx is not None:
            pd_pos = pd.to_datetime(df.loc[pos_idx, date_col]).normalize()
            pd_neg = pd.to_datetime(df.loc[neg_idx, date_col]).normalize()
            if pd_pos == pd_neg:
                conflict = 1
                if abs(df.loc[pos_idx, score_col]) >= abs(df.loc[neg_idx, score_col]):
                    if neg2_idx is not None:
                        neg_idx = neg2_idx
                else:
                    if pos2_idx is not None:
                        pos_idx = pos2_idx
        if pos_idx is not None:
            raw = float(df.loc[pos_idx, score_col])
            norm = (abs(raw) - amin) / rng
            records.append({
                'window_center': d,
                'window_start': ws,
                'window_end': we,
                'peak_date': pd.to_datetime(df.loc[pos_idx, date_col]).normalize(),
                'peak_score': float(norm),
                'peak_idx': int(pos_idx),
                'direction': 'pos',
                'peak_rank': 1,
                'conflict_flag': conflict
            })
        if neg_idx is not None:
            rawn = float(df.loc[neg_idx, score_col])
            normn = (abs(rawn) - amin) / rng
            records.append({
                'window_center': d,
                'window_start': ws,
                'window_end': we,
                'peak_date': pd.to_datetime(df.loc[neg_idx, date_col]).normalize(),
                'peak_score': float(normn),
                'peak_idx': int(neg_idx),
                'direction': 'neg',
                'peak_rank': 1,
                'conflict_flag': conflict
            })
    return pd.DataFrame(records)
