from typing import List, Optional
import numpy as np
from sklearn.decomposition import PCA

def _select_device() -> str:
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'

def _encode_st(model_name: str, texts: List[str]) -> Optional[np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
        device = _select_device()
        m = SentenceTransformer(model_name, device=device)
        return np.array(m.encode(texts, convert_to_numpy=True))
    except Exception:
        return None

def _encode_transformers_mean(model_name: str, texts: List[str]) -> Optional[np.ndarray]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = _select_device()
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        if device != 'cpu':
            mdl.to(device)
        embs = []
        mdl.eval()
        with torch.no_grad():
            for t in texts:
                enc = tok(t, return_tensors='pt', truncation=True, max_length=128)
                if device != 'cpu':
                    enc = {k: v.to(device) for k, v in enc.items()}
                out = mdl(**enc)
                h = out.last_hidden_state
                m = h.mean(dim=1).detach().cpu().numpy()
                embs.append(m[0])
        return np.array(embs)
    except Exception:
        return None

def _normalize_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    proc = []
    for b in blocks:
        if b is None:
            continue
        x = b - np.nanmean(b, axis=0)
        std = np.nanstd(x, axis=0)
        std[std < 1e-8] = 1.0
        x = x / std
        scale = np.sqrt(x.shape[1]) if x.shape[1] > 0 else 1.0
        x = x / scale
        proc.append(x)
    if not proc:
        return np.zeros((0, 1))
    return np.concatenate(proc, axis=1)

def fused_pc1(texts: List[str], core: np.ndarray) -> np.ndarray:
    e_mpnet = _encode_st('all-mpnet-base-v2', texts)
    e_fin = _encode_transformers_mean('yiyanghkust/finbert-pretrain', texts)
    e_bt = _encode_transformers_mean('vinai/bertweet-base', texts)
    emb = _normalize_blocks([e_mpnet, e_fin, e_bt])
    if emb.size == 0:
        return np.zeros(len(texts))
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(emb).ravel()
    if len(core) == len(pc1):
        c = np.corrcoef(pc1, core)[0, 1]
        if np.isnan(c) or c < 0:
            pc1 = -pc1
    m = np.nanmean(pc1)
    s = np.nanstd(pc1)
    s = s if s > 0 else 1.0
    pc1 = (pc1 - m) / s
    pc1 = np.clip(pc1, -3.0, 3.0)
    pc1 = pc1 / 3.0
    return pc1
