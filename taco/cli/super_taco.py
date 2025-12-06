import argparse
import os
import pandas as pd
import numpy as np
from subprocess import run, CalledProcessError

def filter_super_taco(features_path: str, events_path: str) -> pd.DataFrame:
    feats = pd.read_csv(features_path)
    events = pd.read_csv(events_path, parse_dates=['peak_date'])
    if feats.empty or events.empty:
        return pd.DataFrame([])
    feats['created_at'] = pd.to_datetime(feats['created_at']).dt.tz_localize(None)
    start = pd.Timestamp('2017-01-20')
    end = pd.Timestamp('2021-01-08')
    fsub = feats[(feats['created_at'] >= start) & (feats['created_at'] <= end)].copy()
    # 联合：以peak_idx连接到推文子集，若不可用则按peak_date范围
    if 'peak_idx' in events.columns:
        fsub = fsub.copy()
        fsub['row_idx'] = fsub.index.astype(int)
        joined = events.merge(fsub[['row_idx']], left_on='peak_idx', right_on='row_idx', how='inner')
    else:
        events['peak_date'] = pd.to_datetime(events['peak_date']).dt.normalize()
        joined = events[(events['peak_date'] >= start) & (events['peak_date'] <= end)].copy()
    # 峰值得分字段统一
    if 'peak_score' not in joined.columns:
        if 'epeak_scor' in joined.columns:
            joined = joined.rename(columns={'epeak_scor': 'peak_score'})
        else:
            joined['peak_score'] = np.nan
    joined['direction'] = joined['direction'].astype(str)
    # 按方向分位阈值
    pos = joined[joined['direction'] == 'pos']
    neg = joined[joined['direction'] == 'neg']
    q95 = np.nanquantile(pos['peak_score'], 0.95) if len(pos) else np.inf
    q05 = np.nanquantile(neg['peak_score'], 0.05) if len(neg) else -np.inf
    keep_pos = pos[pos['peak_score'] >= q95]
    keep_neg = neg[neg['peak_score'] <= q05]
    out = pd.concat([keep_pos, keep_neg], ignore_index=True)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--events', required=True)
    p.add_argument('--out_dir', default='result/super_taco')
    p.add_argument('--assets', default='NVDA,AAPL,AMZN,TSLA,PFE,UNH,GOOG,META,BTC-USD')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    filt = filter_super_taco(args.features, args.events)
    filt_path = os.path.join(args.out_dir, 'events_filtered.csv')
    filt.to_csv(filt_path, index=False)
    if filt.empty:
        return
    car_path = os.path.join(args.out_dir, 'car_results.csv')
    stats_path = os.path.join(args.out_dir, 'stats_summary.csv')
    sector_path = os.path.join(args.out_dir, 'sector_stats.csv')
    plots_dir = os.path.join(args.out_dir, 'plots')
    reg_path = os.path.join(args.out_dir, 'regression_summary.txt')
    os.makedirs(plots_dir, exist_ok=True)
    # 逐步调用现有CLI
    env = os.environ.copy()
    assets_use = args.assets
    if not env.get('WRDS_USERNAME') or not env.get('WRDS_PASSWORD'):
        assets_use = 'BTC-USD'
    try:
        run(f"python -m taco.cli.compute_car --events {filt_path} --assets {assets_use} --out {car_path}", shell=True, check=True, env=env)
    except CalledProcessError:
        return
    try:
        run(f"python -m taco.cli.stats --car {car_path} --out {stats_path}", shell=True, check=True, env=env)
    except CalledProcessError:
        pass
    try:
        run(f"python -m taco.cli.sector_stats --car {car_path} --out {sector_path} --plots_dir {plots_dir}", shell=True, check=True, env=env)
    except CalledProcessError:
        pass
    try:
        run(f"python -m taco.cli.regression --car {car_path} --events {filt_path} --features {args.features} --out {reg_path}", shell=True, check=True, env=env)
    except CalledProcessError:
        pass

if __name__ == '__main__':
    main()
