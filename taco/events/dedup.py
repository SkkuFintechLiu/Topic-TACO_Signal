import pandas as pd

def deduplicate_events(events: pd.DataFrame, min_gap_days: int = 5) -> pd.DataFrame:
    e = events.sort_values('peak_date').reset_index(drop=True)
    keep = []
    skip = set()
    for i in range(len(e)):
        if i in skip:
            continue
        curr = e.iloc[i]
        j = i + 1
        while j < len(e):
            nxt = e.iloc[j]
            gap = (pd.to_datetime(nxt['peak_date']) - pd.to_datetime(curr['peak_date'])).days
            # only dedup within same direction AND same topic (if topic exists)
            same_topic = True
            if 'topic' in e.columns:
                same_topic = str(nxt.get('topic','')) == str(curr.get('topic',''))
            if gap >= min_gap_days or nxt['direction'] != curr['direction'] or not same_topic:
                break
            if abs(nxt['peak_score']) > abs(curr['peak_score']):
                skip.add(i)
                curr = nxt
            else:
                skip.add(j)
            j += 1
        if i not in skip:
            keep.append(curr)
    return pd.DataFrame(keep)
