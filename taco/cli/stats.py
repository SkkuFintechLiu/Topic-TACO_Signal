import argparse
import os
import pandas as pd
import numpy as np
from taco.stats.tests import t_test_zero, bootstrap_ci, fdr_bh

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--car', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.car)
    if df.empty:
        pd.DataFrame([]).to_csv(args.out, index=False)
        return
    g = df.groupby(['asset', 'window'])['CAR']
    rows = []
    pvals = []
    idx = []
    for (a, w), vals in g:
        v = vals.dropna().values
        t, p = t_test_zero(v)
        lo, hi = bootstrap_ci(v)
        rows.append({'asset': a, 'window': w, 'mean_car': float(np.mean(v)), 'std_car': float(np.std(v, ddof=1)), 't_stat': float(t), 'p_value': float(p), 'ci_lower': float(lo), 'ci_upper': float(hi)})
        pvals.append(p)
        idx.append((a, w))
    adj = fdr_bh(pvals) if pvals else []
    for i in range(len(rows)):
        rows[i]['p_fdr'] = float(adj[i]) if i < len(adj) else float('nan')
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
