import pandas as pd
import numpy as np

def _yf_download(ticker: str, start: str, end: str):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame([])
    df['date'] = pd.to_datetime(df.index)
    return df

def get_macro_fred(start_date: str, end_date: str, priority: str = 'fred', align_dates: list | None = None) -> pd.DataFrame:
    codes = ['VIXCLS','DGS10','DTWEXBGS','GOLDAMGBD228NLBM','DCOILWTICO','TEDRATE']
    macro = None
    if priority == 'wrds':
        try:
            import wrds
            db = wrds.Connection()
            acc = []
            for code in codes:
                df = db.get_table(library='frb', table='fred_dseries', columns=['date','value'], obs=None, coerce_float=True, sql=f"WHERE series_id = '{code}'")
                df = df[(pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))]
                df['series'] = code
                acc.append(df)
            if acc:
                long = pd.concat(acc, ignore_index=True)
                long['date'] = pd.to_datetime(long['date']).dt.tz_localize(None).dt.normalize()
                macro = long.pivot(index='date', columns='series', values='value').reset_index()
            else:
                macro = None
        except Exception:
            macro = None
    if macro is None:
        try:
            import pandas_datareader.data as web
            acc = []
            for code in codes:
                df = web.DataReader(code, 'fred', start_date, end_date)
                df = df.reset_index().rename(columns={'DATE':'date', code: code})
                acc.append(df)
            macro = acc[0]
            for i in range(1, len(acc)):
                macro = macro.merge(acc[i], on='date', how='outer')
        except Exception:
            macro = None
    if macro is None:
        # last resort: Yahoo for DXY/Gold/WTI, yfinance for VIX
        vix = _yf_download('^VIX', start_date, end_date)[['date','Close']].rename(columns={'Close':'VIXCLS'})
        dxy = _yf_download('DX-Y.NYB', start_date, end_date)[['date','Close']].rename(columns={'Close':'DTWEXBGS'})
        if dxy.empty:
            dxy = _yf_download('UUP', start_date, end_date)[['date','Close']].rename(columns={'Close':'DTWEXBGS'})
        gold = _yf_download('GC=F', start_date, end_date)[['date','Close']].rename(columns={'Close':'GOLDAMGBD228NLBM'})
        wti = _yf_download('CL=F', start_date, end_date)[['date','Close']].rename(columns={'Close':'DCOILWTICO'})
        dates_union = sorted(pd.to_datetime(pd.concat([vix[['date']], dxy[['date']], gold[['date']], wti[['date']]], ignore_index=True)['date']).dt.tz_localize(None).dt.normalize().unique())
        macro = pd.DataFrame({'date': dates_union})
        for df, col in [(vix,'VIXCLS'), (dxy,'DTWEXBGS'), (gold,'GOLDAMGBD228NLBM'), (wti,'DCOILWTICO')]:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
            macro = macro.merge(df.set_index('date')[col].rename(col).reset_index(), on='date', how='left')
        macro['TEDRATE'] = np.nan
        macro['DGS10'] = np.nan
    macro['date'] = pd.to_datetime(macro['date']).dt.tz_localize(None).dt.normalize()
    macro = macro.sort_values('date').set_index('date').ffill()
    for col in ['VIXCLS','DGS10','DTWEXBGS','GOLDAMGBD228NLBM','DCOILWTICO','TEDRATE']:
        if col not in macro.columns:
            macro[col] = np.nan
    if 'DGS10' in macro.columns:
        macro['DGS10'] = pd.to_numeric(macro['DGS10'], errors='coerce') / 100.0
    diff = macro[['VIXCLS','DGS10','DTWEXBGS','GOLDAMGBD228NLBM','DCOILWTICO','TEDRATE']].apply(pd.to_numeric, errors='coerce').diff()
    diff = diff.rename(columns={'VIXCLS':'dVIX','DGS10':'dY10','DTWEXBGS':'dDXY','GOLDAMGBD228NLBM':'dGold','DCOILWTICO':'dWTI','TEDRATE':'dTED'})
    if align_dates is not None and len(align_dates) > 0:
        idx = pd.to_datetime(align_dates).tz_localize(None).normalize()
        diff = diff.reindex(idx)
    diff = diff.fillna(0.0)
    return diff
