import argparse
import os
import hashlib
import pandas as pd
import numpy as np
from taco.nlp.clean import clean_text
from taco.nlp.topic import classify_topic
from taco.nlp.policy_topic import PolicyTopicClassifier
from taco.nlp.sentiment import predict_proba
from taco.nlp.embeddings import fused_pc1
from taco.signals.taco import engagement_weight, taco_signal, quality_score

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tweets', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.2)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    src_hash = file_hash(args.tweets)
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out, nrows=1)
            if 'source_hash' in prev.columns and prev['source_hash'].iloc[0] == src_hash:
                return
        except Exception:
            pass
    df = pd.read_csv(args.tweets)
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    df['topic'] = df['text_clean'].apply(classify_topic)
    ptc = PolicyTopicClassifier(mode="gpt_zero_shot")
    df['policy_topic'] = df['text_clean'].apply(ptc.classify)
    df['created_at'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['date'] = df['created_at'].dt.normalize()
    df['hour'] = df['created_at'].dt.hour
    df['minute'] = df['created_at'].dt.minute
    df['is_retweet'] = df['is_retweet'].astype(str).str.upper().eq('TRUE')
    base = df[~df['is_retweet']].copy()
    start_bound = pd.Timestamp('2016-09-01', tz='UTC')
    end_bound = pd.Timestamp('2021-03-01', tz='UTC')
    base = base[(base['created_at'] >= start_bound) & (base['created_at'] <= end_bound)].copy()
    probs = predict_proba(base['text_clean'].tolist())
    base['pos'] = probs[:, 0]
    base['neg'] = probs[:, 1]
    base['neu'] = probs[:, 2]
    base['core'] = base['pos'] - base['neg']
    base['clarity'] = (base['pos'] + base['neg']) - base['neu']
    base['polarity'] = (base['pos'] - base['neg']).abs() / (base['pos'] + base['neg'] + base['neu']).replace(0, np.nan)
    base['polarity'] = base['polarity'].fillna(0)
    pc1 = fused_pc1(base['text_clean'].tolist(), base['core'].values)
    base['pc1'] = pc1
    base['engagement'] = (base['favorites'].fillna(0).astype(float) + base['retweets'].fillna(0).astype(float))
    base['weight'] = engagement_weight(base['favorites'], base['retweets'])
    base['taco_signal'] = taco_signal(base['core'], base['pc1'], alpha=args.alpha, beta=args.beta)
    m = base['taco_signal'].mean()
    s = base['taco_signal'].std() if base['taco_signal'].std() > 0 else 1.0
    base['signal_z'] = (base['taco_signal'] - m) / s
    kws = {'tariff': 2.0, 'war': 2.0, 'sanctions': 1.8, 'fed': 1.5, 'rate': 1.3, 'crisis': 1.8}
    kw_strength = base['text_clean'].apply(lambda t: sum(kws[w] for w in t.split() if w in kws))
    kw_weight = np.minimum(kw_strength / 2.0, 1.0)
    base['text_len'] = base['text_clean'].apply(lambda t: len([w for w in t.split() if w]))
    base['quality_score'] = quality_score(base['pos'], base['neg'], base['neu'], base['engagement'], base['text_len'], kw_weight)
    base['source_hash'] = src_hash
    cols = ['id', 'created_at', 'date', 'hour', 'minute', 'text_clean', 'topic', 'favorites', 'retweets', 'engagement', 'weight', 'pos', 'neg', 'neu', 'core', 'clarity', 'polarity', 'pc1', 'taco_signal', 'signal_z', 'quality_score', 'device', 'source_hash']
    base[cols].to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
