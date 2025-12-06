import re

def clean_text(x: str) -> str:
    x = re.sub(r'https?://\S+', ' ', x)
    x = re.sub(r'@\w+', ' ', x)
    x = re.sub(r'#\w+', ' ', x)
    x = re.sub(r'^RT\s+', '', x)
    x = re.sub(r'[^\x00-\x7F]+', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip().lower()
    return x
