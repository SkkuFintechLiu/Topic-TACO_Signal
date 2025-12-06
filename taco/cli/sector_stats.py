import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from taco.stats.tests import t_test_zero, bootstrap_ci

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--car', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--plots_dir', default='data/results/plots')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    df = pd.read_csv(args.car)
    if df.empty or 'sector' not in df.columns:
        pd.DataFrame([]).to_csv(args.out, index=False)
        return
    g = df.groupby(['sector', 'window'])['CAR']
    rows = []
    for (s, w), vals in g:
        v = vals.dropna().values
        if len(v) < 3:
            continue
        t, pval = t_test_zero(v)
        lo, hi = bootstrap_ci(v)
        rows.append({'sector': s, 'window': w, 'mean_car': float(np.mean(v)), 'std_car': float(np.std(v, ddof=1)), 't_stat': float(t), 'p_value': float(pval), 'ci_lower': float(lo), 'ci_upper': float(hi), 'n': int(len(v))})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    if not out.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=out, x='sector', y='mean_car', hue='window', errorbar=None)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Mean CAR')
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, 'sector_car_bar.png'))
        plt.close()
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[df['sector'].isin(out['sector'])], x='sector', y='CAR', hue='window')
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('CAR')
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, 'sector_car_box.png'))
        plt.close()

if __name__ == '__main__':
    main()
