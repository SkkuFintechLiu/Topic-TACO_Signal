from typing import List
import numpy as np

_pipeline = None
_device = 'cpu'

def load_finbert():
    global _pipeline
    global _device
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                _device = 'mps'
            elif torch.cuda.is_available():
                _device = 'cuda'
        except Exception:
            _device = 'cpu'
        tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone', model_max_length=128)
        mdl = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        try:
            mdl.to(_device)
        except Exception:
            _device = 'cpu'
        _pipeline = pipeline(
            task='text-classification',
            model=mdl,
            tokenizer=tok,
            top_k=None,
            truncation=True
        )
        return _pipeline
    except Exception:
        return None

def predict_proba(texts: List[str]) -> np.ndarray:
    pipe = load_finbert()
    if pipe is None:
        return np.tile(np.array([0.33, 0.33, 0.34]), (len(texts), 1))
    out = pipe(texts)
    arr = np.zeros((len(texts), 3))
    for i, o in enumerate(out):
        d = {item['label'].lower(): item['score'] for item in o}
        arr[i] = [d.get('positive', 0.0), d.get('negative', 0.0), d.get('neutral', 0.0)]
    return arr
