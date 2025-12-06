import argparse
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from taco.finance.sensitivity import get_sensitivity, load_sensitivity
from taco.finance.macro import get_macro_fred
from taco.stats.tests import t_test_zero, bootstrap_ci, fdr_bh

def build_dataset(car_path: str, events_path: str, features_path: str) -> pd.DataFrame:
    car = pd.read_csv(car_path)
    ev = pd.read_csv(events_path, parse_dates=['peak_date'])
    feats = pd.read_csv(features_path)
    if car.empty or ev.empty:
        return pd.DataFrame([])
    ev['peak_date'] = pd.to_datetime(ev['peak_date']).dt.tz_localize(None).dt.normalize()
    car['event_date'] = pd.to_datetime(car['event_date']).dt.tz_localize(None).dt.normalize()
    df = car.merge(ev[['peak_date','peak_idx','direction','peak_score']], left_on='event_date', right_on='peak_date', how='left')
    df = df.dropna(subset=['peak_idx'])
    feats['row_idx'] = np.arange(len(feats))
    feats['date'] = pd.to_datetime(feats['date']).dt.normalize()
    df = df.merge(feats[['row_idx','quality_score','topic','policy_topic','weight']], left_on='peak_idx', right_on='row_idx', how='left')
    df['direction_pos'] = (df['direction']=='pos').astype(int)
    df['direction_neg'] = (df['direction']=='neg').astype(int)
    df['window'] = df['window'].astype(str)
    df['sector'] = df['sector'].fillna('')
    df['CAR'] = pd.to_numeric(df['CAR'], errors='coerce')
    df['peak_score'] = pd.to_numeric(df['peak_score'], errors='coerce')
    df['quality_score'] = pd.to_numeric(df['quality_score'], errors='coerce')
    df = df.dropna(subset=['CAR','peak_score'])
    sens = load_sensitivity()
    df['sensitivity'] = df.apply(lambda r: get_sensitivity(r.get('asset',''), r.get('sector',''), r.get('policy_topic',''), sens), axis=1)
    df['interaction'] = df['peak_score'] * df['sensitivity']
    return df

def ols_reg(df: pd.DataFrame):
    X = pd.get_dummies(df[['peak_score','quality_score','interaction','direction','sector','window','topic']], columns=['direction','sector','window','topic'], drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(df['CAR'], errors='coerce').astype(float)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def fe_reg(df: pd.DataFrame):
    # 资产与窗口固定效应通过哑变量实现
    X = pd.get_dummies(df[['peak_score','quality_score','interaction','asset','window']], columns=['asset','window'], drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(df['CAR'], errors='coerce').astype(float)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def fm_reg(df: pd.DataFrame):
    # 按事件日期分组做横截面回归，汇总系数均值与t值
    coefs = []
    dates = []
    for d, g in df.groupby('event_date'):
        if len(g) < 3:
            continue
        X = sm.add_constant(g[['peak_score','quality_score','interaction']].apply(pd.to_numeric, errors='coerce'))
        y = pd.to_numeric(g['CAR'], errors='coerce')
        try:
            res = sm.OLS(y, X, missing='drop').fit()
            params = res.params.reindex(['const','peak_score','quality_score','interaction'], fill_value=np.nan)
            coefs.append(params.values)
            dates.append(d)
        except Exception:
            continue
    if not coefs:
        return None
    A = np.array(coefs, dtype=float)
    mean = np.nanmean(A, axis=0)
    std = np.nanstd(A, axis=0, ddof=1)
    count = np.sum(~np.isnan(A), axis=0)
    se = std / np.sqrt(np.maximum(count, 1))
    t = mean / (se + 1e-8)
    idx = ['const','peak_score','quality_score','interaction']
    return pd.DataFrame({'coef': mean, 't': t}, index=idx)

def stats_block(df: pd.DataFrame):
    rows = []
    for (w), g in df.groupby(['window']):
        v = g['CAR'].dropna().values
        if len(v) < 3:
            continue
        t, p = t_test_zero(v)
        lo, hi = bootstrap_ci(v)
        rows.append({'window': w, 'mean_car': float(np.mean(v)), 't_stat': float(t), 'p_value': float(p), 'ci_lower': float(lo), 'ci_upper': float(hi), 'n': int(len(v))})
    R = pd.DataFrame(rows)
    if not R.empty:
        R['p_fdr'] = fdr_bh(R['p_value'].values)
    return R

def write_report(path: str, df: pd.DataFrame, ols, fe, fm, stat_df: pd.DataFrame):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('回归分析统一报告\n')
        f.write('================\n\n')
        f.write(f'样本大小: {len(df)}\n')
        f.write(f'资产数: {df["asset"].nunique() if "asset" in df.columns else 0}\n')
        f.write(f'窗口: {sorted(df["window"].unique().tolist())}\n\n')
        f.write('OLS结果\n--------\n')
        f.write(str(ols.summary()) + '\n\n')
        f.write('固定效应(资产+窗口) OLS结果\n-------------------------\n')
        f.write(str(fe.summary()) + '\n\n')
        f.write('Fama-MacBeth结果\n-----------------\n')
        if fm is not None:
            f.write(fm.to_string() + '\n\n')
        else:
            f.write('样本不足，FM未计算\n\n')
        f.write('t检验/Bootstrap/FDR 汇总\n--------------------------\n')
        if stat_df is not None and not stat_df.empty:
            f.write(stat_df.to_string(index=False) + '\n')
        else:
            f.write('无可汇总统计\n')

def macro_regression(df: pd.DataFrame, start: str = '2016-09-01', end: str = '2021-03-01', macro_source: str = 'fred', macro_csv: str = 'result/macro_diff.csv'):
    dates = sorted(pd.to_datetime(df['event_date']).dt.tz_localize(None).dt.normalize().unique())
    if macro_source == 'csv':
        try:
            m = pd.read_csv(macro_csv, parse_dates=['date'])
            m['date'] = pd.to_datetime(m['date']).dt.tz_localize(None).dt.normalize()
            cols = {}
            if 'VIX' in m.columns:
                cols['VIX'] = 'dVIX'
            if 'Y10' in m.columns:
                cols['Y10'] = 'dY10'
            if 'DXY' in m.columns:
                cols['DXY'] = 'dDXY'
            if 'WTI' in m.columns:
                cols['WTI'] = 'dWTI'
            if 'TED' in m.columns:
                cols['TED'] = 'dTED'
            m = m.rename(columns=cols)
            for c in ['dVIX','dY10','dDXY','dWTI','dTED','dGold']:
                if c not in m.columns:
                    m[c] = 0.0
            mac = m[['date','dVIX','dY10','dDXY','dWTI','dTED','dGold']].set_index('date')
            idx = pd.to_datetime(dates)
            mac = mac.reindex(idx).fillna(0.0)
            macro_source = 'csv'
        except Exception:
            mac = pd.DataFrame(index=pd.to_datetime(dates), columns=['dVIX','dY10','dDXY','dGold','dWTI','dTED']).fillna(0.0)
            macro_source = 'none'
    else:
        try:
            mac = get_macro_fred(start, end, priority=macro_source, align_dates=dates)
        except Exception:
            mac = pd.DataFrame(index=pd.to_datetime(dates), columns=['dVIX','dY10','dDXY','dGold','dWTI','dTED']).fillna(0.0)
            macro_source = 'none'
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['event_date']).dt.tz_localize(None).dt.normalize()
    dfm = df2.merge(mac, left_on='date', right_index=True, how='left')
    X = dfm[['peak_score','interaction','dVIX','dY10','dDXY','dGold','dWTI','dTED']]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(dfm['CAR'], errors='coerce').astype(float)
    model = sm.OLS(y, X, missing='drop').fit()
    fill_rates = {}
    for c in ['dVIX','dY10','dDXY','dGold','dWTI','dTED']:
        s = dfm[c]
        fill_rates[c] = float(s.isna().mean())
    return model, {'source': macro_source, 'dates': len(dates), 'fill_na_rate': fill_rates}

def taco_ols_with_macro(df: pd.DataFrame, start: str = '2016-09-01', end: str = '2021-03-01', macro_source: str = 'csv', macro_csv: str = 'result/macro_diff.csv'):
    dates = sorted(pd.to_datetime(df['event_date']).dt.tz_localize(None).dt.normalize().unique())
    if macro_source == 'csv':
        try:
            m = pd.read_csv(macro_csv, parse_dates=['date'])
            m['date'] = pd.to_datetime(m['date']).dt.tz_localize(None).dt.normalize()
            cols = {}
            if 'VIX' in m.columns:
                cols['VIX'] = 'dVIX'
            if 'Y10' in m.columns:
                cols['Y10'] = 'dY10'
            if 'DXY' in m.columns:
                cols['DXY'] = 'dDXY'
            if 'WTI' in m.columns:
                cols['WTI'] = 'dWTI'
            if 'TED' in m.columns:
                cols['TED'] = 'dTED'
            m = m.rename(columns=cols)
            for c in ['dVIX','dY10','dDXY','dWTI','dTED','dGold']:
                if c not in m.columns:
                    m[c] = 0.0
            mac = m[['date','dVIX','dY10','dDXY','dWTI','dTED','dGold']].set_index('date')
            idx = pd.to_datetime(dates)
            mac = mac.reindex(idx).fillna(0.0)
        except Exception:
            mac = pd.DataFrame(index=pd.to_datetime(dates), columns=['dVIX','dY10','dDXY','dGold','dWTI','dTED']).fillna(0.0)
    else:
        try:
            mac = get_macro_fred(start, end, priority=macro_source, align_dates=dates)
        except Exception:
            mac = pd.DataFrame(index=pd.to_datetime(dates), columns=['dVIX','dY10','dDXY','dGold','dWTI','dTED']).fillna(0.0)
    df2 = df.copy()
    df2['event_date'] = pd.to_datetime(df2['event_date']).dt.tz_localize(None).dt.normalize()
    mac_slice = mac.reindex(df2['event_date']).reset_index(drop=True)
    X = pd.DataFrame({'peak_score': df2['peak_score'], 'interaction': df2['interaction']})
    X = pd.concat([X, mac_slice[['dVIX','dY10','dDXY','dWTI','dTED','dGold']]], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(df2['CAR'], errors='coerce').astype(float)
    model = sm.OLS(y, X, missing='drop').fit()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--car', required=True)
    p.add_argument('--events', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--macro_source', default='csv')
    p.add_argument('--macro_csv', default='result/macro_diff.csv')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = build_dataset(args.car, args.events, args.features)
    if df.empty:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('数据为空，无法进行回归分析\n')
        return
    ols = ols_reg(df)
    fe = fe_reg(df)
    fm = fm_reg(df)
    stat_df = stats_block(df)
    write_report(args.out, df, ols, fe, fm, stat_df)
    mac_model, mac_info = macro_regression(df, macro_source=args.macro_source, macro_csv=args.macro_csv)
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write('\n宏观控制的事件期 CAR 回归\n-----------------------\n')
        f.write(f"来源: {mac_info['source']}, 交集日期数: {mac_info['dates']}\n")
        fr = mac_info['fill_na_rate']
        f.write(f"缺失率: dVIX={fr['dVIX']:.3f}, dY10={fr['dY10']:.3f}, dDXY={fr['dDXY']:.3f}, dGold={fr['dGold']:.3f}, dWTI={fr['dWTI']:.3f}, dTED={fr['dTED']:.3f}\n\n")
        f.write(str(mac_model.summary()) + '\n')
    taco_macro_model = taco_ols_with_macro(df, macro_source=args.macro_source, macro_csv=args.macro_csv)
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write('\n宏观控制下的 TACO OLS 分析\n-------------------------\n')
        f.write(str(taco_macro_model.summary()) + '\n')

if __name__ == '__main__':
    main()
