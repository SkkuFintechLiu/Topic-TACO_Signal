import argparse
import os
import pandas as pd
from taco.events.peaks import sliding_window_peak_detection
from taco.events.dedup import deduplicate_events

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--out_raw', default='data/processed/events_raw.csv')
    p.add_argument('--out', required=True)
    p.add_argument('--window_days', type=int, default=15)
    p.add_argument('--min_gap_days', type=int, default=5)
    p.add_argument('--severity_only', action='store_true')
    p.add_argument('--severity_threshold', type=float, default=4.0)
    p.add_argument('--verbs_only', action='store_true')
    p.add_argument('--topic_whitelist', default='')
    p.add_argument('--limit_per_day', type=int, default=0)
    p.add_argument('--limit_per_week', type=int, default=0)
    p.add_argument('--trading_days_only', action='store_true')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.features, parse_dates=['created_at', 'date'])
    events = sliding_window_peak_detection(df, date_col='date', score_col='taco_signal', window_days=args.window_days)
    if not df.empty and not events.empty:
        m = df['taco_signal'].mean()
        s = df['taco_signal'].std() if df['taco_signal'].std() > 0 else 1.0
        df['signal_z'] = (df['taco_signal'] - m) / s
        verbs = ['impose','sanction','order','sign','approve','restrict','ban','tariff','hike','cut','regulate']
        df['verb_hit'] = df['text_clean'].astype(str).str.lower().apply(lambda t: any(v in t for v in verbs))
        def sev_row(idx, pk):
            try:
                r = df.iloc[int(pk)]
                return (float(idx['peak_score']) >= 0.8) or (float(r.get('signal_z', 0.0)) >= args.severity_threshold) or bool(r.get('verb_hit', False))
            except Exception:
                return float(idx['peak_score']) >= 0.8
        events['severity_flag'] = events.apply(lambda x: sev_row(x, x['peak_idx']), axis=1)
        # enrich topic if available
        if 'topic' in df.columns:
            def topic_of(pk):
                try:
                    return df.iloc[int(pk)]['topic']
                except Exception:
                    return ''
            events['topic'] = events['peak_idx'].apply(topic_of)
        else:
            events['topic'] = ''
        if args.severity_only:
            events = events[events['severity_flag'] == True]
        if args.verbs_only:
            def verb_of(pk):
                try:
                    return bool(df.iloc[int(pk)].get('verb_hit', False))
                except Exception:
                    return False
            events = events[events['peak_idx'].apply(verb_of)]
        wl = [t.strip() for t in args.topic_whitelist.split(',') if t.strip()]
        if wl:
            events = events[events['topic'].isin(wl)]
        # trading day only
        if args.trading_days_only:
            events = events[pd.to_datetime(events['peak_date']).dt.weekday < 5]
        # per-day limit
        if args.limit_per_day and args.limit_per_day > 0:
            events = (events.sort_values(['peak_date','peak_score'], ascending=[True, False])
                            .groupby('peak_date', as_index=False)
                            .head(args.limit_per_day))
        # per-week limit
        if args.limit_per_week and args.limit_per_week > 0:
            wk = pd.to_datetime(events['peak_date']).dt.to_period('W-MON').astype(str)
            events['week'] = wk
            events = (events.sort_values(['week','peak_score'], ascending=[True, False])
                            .groupby('week', as_index=False)
                            .head(args.limit_per_week))
            events = events.drop(columns=['week'])
    os.makedirs(os.path.dirname(args.out_raw), exist_ok=True)
    events.to_csv(args.out_raw, index=False)
    dedup = deduplicate_events(events, min_gap_days=args.min_gap_days)
    dedup.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
