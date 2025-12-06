import numpy as np

def t_test_zero(values):
    n = len(values)
    m = np.mean(values)
    s = np.std(values, ddof=1)
    se = s / np.sqrt(n) if n > 0 else 1.0
    t = m / (se + 1e-8)
    from scipy.stats import t as tdist
    p = tdist.sf(abs(t), n - 1) * 2
    return t, p

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    np.random.seed(0)
    means = []
    for _ in range(n_boot):
        b = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(b))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi

def fdr_bh(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adj = np.zeros(m)
    for i in range(m):
        adj[order[i]] = min(ranked[i] * m / (i + 1), 1.0)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    return adj
