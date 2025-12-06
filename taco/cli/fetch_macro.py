import argparse
import os
import pandas as pd
import pandas_datareader.data as web

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2021-03-01')
    p.add_argument('--out_dir', default='result')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    codes = {
        'VIX': 'VIXCLS',
        'Y10': 'DGS10',
        'DXY': 'DTWEXBGS',
        'WTI': 'DCOILWTICO',
        'TED': 'TEDRATE'
    }
    df = web.DataReader(list(codes.values()), 'fred', args.start, args.end)
    df.columns = list(codes.keys())
    df = df.sort_index().ffill()
    macro = df.copy()
    macro = macro.reset_index().rename(columns={'DATE':'date'})
    macro['date'] = pd.to_datetime(macro['date']).dt.tz_localize(None).dt.normalize()
    macro.to_csv(os.path.join(args.out_dir, 'macro.csv'), index=False)
    md = df.copy()
    md['Y10'] = pd.to_numeric(md['Y10'], errors='coerce') / 100.0
    md = md.diff().dropna()
    md = md.reset_index().rename(columns={'DATE':'date'})
    md['date'] = pd.to_datetime(md['date']).dt.tz_localize(None).dt.normalize()
    md.to_csv(os.path.join(args.out_dir, 'macro_diff.csv'), index=False)

if __name__ == '__main__':
    main()

