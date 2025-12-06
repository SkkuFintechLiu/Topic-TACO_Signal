import json
import os

DEFAULT = {
    ("Technology","Trade_Tariff"): 0.9,
    ("Technology","Rate_Fed"): 0.6,
    ("Technology","China_Conflict"): 0.7,
    ("Technology","Tech_Regulation"): 0.9,
    ("Technology","Defense_NationalSecurity"): 0.3,
    ("Crypto","Rate_Fed"): 0.9,
    ("Crypto","Trade_Tariff"): 0.3,
    ("Crypto","China_Conflict"): 0.5,
    ("Crypto","Tech_Regulation"): 0.4,
    ("Crypto","Defense_NationalSecurity"): 0.3,
    ("Health Care","Defense_NationalSecurity"): 0.2,
    ("Health Care","Rate_Fed"): 0.5,
    ("Health Care","Trade_Tariff"): 0.3,
    ("Health Care","Tech_Regulation"): 0.4,
    ("Health Care","China_Conflict"): 0.5,
    ("Consumer Discretionary","Trade_Tariff"): 0.8,
    ("Consumer Discretionary","Rate_Fed"): 0.6,
    ("Consumer Discretionary","China_Conflict"): 0.6,
    ("Consumer Discretionary","Tech_Regulation"): 0.5,
    ("Consumer Discretionary","Defense_NationalSecurity"): 0.4,
    ("Communication Services","Tech_Regulation"): 0.8,
    ("Communication Services","Rate_Fed"): 0.6,
    ("Communication Services","Trade_Tariff"): 0.4,
    ("Communication Services","China_Conflict"): 0.5,
    ("Communication Services","Defense_NationalSecurity"): 0.4,
}

def load_sensitivity(path: str = 'data/config/sensitivity.json'):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return {tuple(k.split('||')): float(v) for k, v in raw.items()}
        except Exception:
            return DEFAULT
    return DEFAULT

def save_sensitivity(mapping, path: str = 'data/config/sensitivity.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"||".join(k): float(v) for k, v in mapping.items()}, f)

def get_sensitivity(asset: str, sector: str, topic: str, mapping=None) -> float:
    m = mapping or DEFAULT
    key = (sector or "", topic or "")
    return float(m.get(key, 0.5))

