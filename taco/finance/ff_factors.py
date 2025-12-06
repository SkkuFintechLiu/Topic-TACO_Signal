import os
import io
import zipfile
import pandas as pd
import numpy as np
import requests
from io import StringIO

def _parse_ken_csv_from_zip(url_candidates):
    for url in url_candidates:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            # pick the first CSV/TXT file
            name = None
            for n in z.namelist():
                if n.lower().endswith('.csv') or n.lower().endswith('.txt'):
                    name = n
                    break
            if name is None:
                continue
            content = z.read(name).decode('latin-1')
            lines = [ln.strip() for ln in content.splitlines()]
            data_rows = []
            for ln in lines:
                # lines that start with a date (YYYYMMDD or YYYYMM)
                if ln and ln[0].isdigit():
                    parts = [p for p in ln.replace('\t', ',').replace('  ', ' ').replace(' ', ',').split(',') if p!='']
                    data_rows.append(parts)
            if not data_rows:
                continue
            # detect daily vs monthly by length of date
            date_len = len(data_rows[0][0])
            is_daily = date_len == 8
            cols = None
            # Try common column sets
            # Expected order: Date, Mkt-RF, SMB, HML, RMW, CMA, RF
            df = pd.DataFrame(data_rows)
            if df.shape[1] >= 7:
                df = df.iloc[:, :7]
                cols = ['date','mktrf','smb','hml','rmw','cma','rf']
                df.columns = cols
            else:
                continue
            if is_daily:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            else:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')
            for c in ['mktrf','smb','hml','rmw','cma','rf']:
                df[c] = pd.to_numeric(df[c], errors='coerce') / 100.0
            df = df.dropna(subset=['date'])
            return df[['date','mktrf','smb','hml','rmw','cma','rf']]
        except Exception:
            continue
    return pd.DataFrame([])

def get_ff5_factors_ken(start_date: str, end_date: str) -> pd.DataFrame:
    base = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
    urls = [
        base + 'F-F_Research_Data_5_Factors_2x3_Daily_CSV.zip',
        base + 'F-F_Research_Data_5_Factors_2x3_Daily.zip',
    ]
    df = _parse_ken_csv_from_zip(urls)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    return df[['date','mktrf','smb','hml','rmw','cma','rf']]

def get_momentum_umd_ken(start_date: str, end_date: str) -> pd.DataFrame:
    base = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
    urls = [
        base + 'F-F_Momentum_Factor_Daily_CSV.zip',
        base + 'F-F_Momentum_Factor_CSV.zip',
        base + 'F-F_Momentum_Factor_Daily.zip',
        base + 'F-F_Momentum_Factor.zip',
    ]
    df = _parse_ken_csv_from_zip(urls)
    if df.empty:
        return pd.DataFrame([])
    # momentum files output as single column sometimes; reuse parsing but map to umd if present
    if 'rf' in df.columns and 'mktrf' in df.columns:
        # Not a pure momentum file; attempt to extract UMD if present
        pass
    # Build from generic parse: first non-date column as UMD
    # Fallback: try pandas_datareader
    try:
        import pandas_datareader.data as web
        data = web.DataReader('F-F_Momentum_Factor', 'famafrench')
        umd = data[0].copy()
        umd.index = pd.to_datetime(umd.index)
        umd = umd.loc[(umd.index >= pd.to_datetime(start_date)) & (umd.index <= pd.to_datetime(end_date))]
        umd = umd.reset_index().rename(columns={'index':'date','Mom':'umd'})
        umd['umd'] = umd['umd'] / 100.0
        umd['date'] = pd.to_datetime(umd['date']).dt.tz_localize(None).dt.normalize()
        return umd[['date','umd']]
    except Exception:
        return pd.DataFrame([])

def get_ff5_factors(start_date: str, end_date: str) -> pd.DataFrame:
    df = get_ff5_factors_ken(start_date, end_date)
    if not df.empty:
        return df
    # Fallback to WRDS then famafrench
    try:
        import wrds
        conn = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME'), wrds_password=os.environ.get('WRDS_PASSWORD'))
        sql = f"select date, mktrf, smb, hml, rmw, cma, rf from ff.five_factors_daily where date between '{start_date}' and '{end_date}'"
        df = conn.raw_sql(sql)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        return df[['date','mktrf','smb','hml','rmw','cma','rf']]
    except Exception:
        try:
            import pandas_datareader.data as web
            data = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench')
            ff = data[0].copy()
            ff.index = pd.to_datetime(ff.index)
            ff = ff.loc[(ff.index >= pd.to_datetime(start_date)) & (ff.index <= pd.to_datetime(end_date))]
            ff = ff.rename(columns={'Mkt-RF':'mktrf','SMB':'smb','HML':'hml','RMW':'rmw','CMA':'cma','RF':'rf'})
            for c in ['mktrf','smb','hml','rmw','cma','rf']:
                ff[c] = ff[c] / 100.0
            ff = ff.reset_index().rename(columns={'index':'date'})
            ff['date'] = pd.to_datetime(ff['date']).dt.tz_localize(None).dt.normalize()
            return ff[['date','mktrf','smb','hml','rmw','cma','rf']]
        except Exception:
            return pd.DataFrame(columns=['date','mktrf','smb','hml','rmw','cma','rf'])

def get_ff5_factors_local(csv_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        with open(csv_path, 'r', encoding='latin-1') as f:
            lines = [ln.strip() for ln in f.readlines()]
        header_idx = None
        for i, ln in enumerate(lines):
            low = ln.lower()
            if ('date' in low) and ('mkt-rf' in low or 'mktrf' in low):
                header_idx = i
                break
        if header_idx is None:
            for i, ln in enumerate(lines):
                if ln and ln[0].isdigit():
                    header_idx = i
                    break
        if header_idx is None:
            return pd.DataFrame([])
        rows = []
        for ln in lines[header_idx+1:]:
            if not ln:
                continue
            t = ln.split(',') if (',' in ln) else ln.split()
            if not t or not t[0].isdigit():
                continue
            if len(t) < 7:
                continue
            rows.append(t[:7])
        if not rows:
            return pd.DataFrame([])
        df = pd.DataFrame(rows, columns=['date','mktrf','smb','hml','rmw','cma','rf'])
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        for c in ['mktrf','smb','hml','rmw','cma','rf']:
            df[c] = pd.to_numeric(df[c], errors='coerce') / 100.0
        df = df.dropna(subset=['date'])
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        start = max(pd.to_datetime('2015-01-01'), pd.to_datetime(start_date))
        end = pd.to_datetime(end_date)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        return df[['date','mktrf','smb','hml','rmw','cma','rf']]
    except Exception:
        return pd.DataFrame([])
