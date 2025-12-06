import argparse
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='out', required=True)
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--severity_only', action='store_true')
    p.add_argument('--topic_whitelist', default='')
    p.add_argument('--direction', default='')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.inp, parse_dates=['peak_date'])
    if df.empty:
        pd.DataFrame([]).to_csv(args.out, index=False)
        return
    df['peak_date'] = pd.to_datetime(df['peak_date']).dt.tz_localize(None).dt.normalize()
    if args.start:
        df = df[df['peak_date'] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df['peak_date'] <= pd.to_datetime(args.end)]
    if args.severity_only and 'severity_flag' in df.columns:
        df = df[df['severity_flag'] == True]
    wl = [t.strip() for t in args.topic_whitelist.split(',') if t.strip()]
    if wl and 'topic' in df.columns:
        df = df[df['topic'].isin(wl)]
    if args.direction in ('pos','neg') and 'direction' in df.columns:
        df = df[df['direction'] == args.direction]
    df.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()

