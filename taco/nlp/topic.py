TOPIC_LEXICON = {
    'Trade_Tariff': {
        'keywords': ['tariff', 'trade', 'duties', 'customs', 'sanction', 'export', 'import'],
        'weight': 1.0
    },
    'Rate_Fed': {
        'keywords': ['fed', 'fomc', 'rate', 'interest', 'hike', 'cut', 'policy'],
        'weight': 1.0
    },
    'War_Conflict': {
        'keywords': ['war', 'conflict', 'attack', 'missile', 'military'],
        'weight': 1.2
    },
    'Macro_Risk': {
        'keywords': ['recession', 'inflation', 'gdp', 'crisis', 'volatility', 'vix'],
        'weight': 1.0
    },
    'Market_Index': {
        'keywords': ['sp500', 's&p', 'dow', 'nasdaq', 'stocks', 'market'],
        'weight': 0.8
    }
}

def classify_topic(text: str) -> str:
    t = text.lower()
    best = None
    best_count = 0
    for k, spec in TOPIC_LEXICON.items():
        c = sum(1 for kw in spec['keywords'] if kw in t)
        if c > best_count:
            best = k
            best_count = c
    return best or 'Other'
