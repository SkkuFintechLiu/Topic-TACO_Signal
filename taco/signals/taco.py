import numpy as np
import pandas as pd

def engagement_weight(fav: pd.Series, rt: pd.Series) -> pd.Series:
    e = (fav.fillna(0) + rt.fillna(0)).astype(float)
    return np.log1p(e)

def taco_signal(core: pd.Series, pc1: pd.Series, alpha: float = 1.0, beta: float = 0.2) -> pd.Series:
    return alpha * core.astype(float) + beta * pc1.astype(float)

def quality_score(pos: pd.Series, neg: pd.Series, neu: pd.Series, engagement: pd.Series, text_len: pd.Series, kw_weight: pd.Series) -> pd.Series:
    conf = pd.concat([pos, neg, neu], axis=1).max(axis=1)
    pol = (pos - neg).abs() / (pos + neg + neu).replace(0, np.nan)
    pol = pol.fillna(0)
    rank = engagement.rank(pct=True)
    ls = 1 / (1 + np.exp(-(text_len - 15) / 10))
    return 0.30 * conf + 0.25 * pol + 0.20 * rank + 0.15 * ls + 0.10 * kw_weight
