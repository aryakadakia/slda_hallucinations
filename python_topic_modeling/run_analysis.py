"""Run the full comparison across all outcomes.

Usage:  SLDA_DATA_DIR=/path/to/data  python3 python_topic_modeling/run_analysis.py

Every model is evaluated with 5-fold cross-validation grouped on participant,
so no participant appears on both sides of a split. Topic models are fitted on
chunks and predictions averaged within participant, so evaluation is always at
participant level.
"""
import os, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_corpus import build, participant_specific_tokens
from evaluate import slda_cv, tfidf_cv, lda_ridge_cv, report
from clean_text import repair

DATA = Path(os.environ.get('SLDA_DATA_DIR', '.'))

HPSVQ_MAPS = {
 'frequency':{'No voices':0,'Less than once a day':1,'Once or twice a day':2,'Several times a day':3,'All of the time / Constantly':4,'All of the time / constantly':4},
 'bad-voices':{'No voices saying bad things':0,'Not that bad':1,'Fairly bad':2,'Very bad':3,'Horrible':4},
 'volume-of-voices':{'Voices not present':0,'Very quiet (like whispering)':1,'Average (same as my own voice)':2,'Fairly loud':3,'Very loud (yelling or shouting)':4},
 'voices-length':{'Voices not present':0,'A few seconds to 1 minute':1,'A few minutes':2,'More than 10 minutes but less than an hour':3,'Longer than 1 hour / they just seem to persist':4},
 'interference-in-activities':{'No interference':0,'A little bit':1,'Moderately':2,'Quite a bit':3,'Extremely interfering':4},
 'distressing-voices':{'No voices are distressing me':0,'A little bit':1,'Moderately':2,'Quite a bit':3,'Extremely distressing':4},
 'worthless-useless-voices':{'No voices make me feel bad':0,'A little bit':1,'Fairly bad':2,'Very bad':3,'Extremely bad (as bad as I can feel)':4},
 'clarity-of-voices':{'Voices not present':0,'Very mumbled':1,'Fairly mumbled':2,'Fairly clear':3,'Very clear voices':4},
 'follow-voices-orders':{'No voices telling me what to do':0,'Rarely':1,'Sometimes':2,'Often':3,'Always':4}}

def compare(title, frame, idcol, textcol, ycol, binary, ks=(6, 10, 15)):
    """Run every model twice: once on the corpus as-is, once with
    participant-specific tokens removed inside each training fold.

    The second arm is the one to believe. Without it a model can score by
    latching onto vocabulary that belongs to one person - names, home towns,
    a specific medication - which is what made the topic models look
    interpretable. The drop set is refitted per fold, so nothing leaks.
    """
    frame = frame.dropna(subset=[textcol, ycol]).copy()
    outcome = dict(zip(frame[idcol], frame[ycol].astype(float)))
    texts = {p: repair(str(t)) for p, t in zip(frame[idcol], frame[textcol])}
    chunks = build(frame, idcol, textcol)
    print(f"\n{'='*74}\n{title}   participants={len(outcome)}  chunks={len(chunks)}")
    tot = chunks.groupby('pid').n_tokens.sum()
    print(f"  largest participant contributes {tot.max()/tot.sum()*100:.1f}% of tokens")

    for filt in (False, True):
        label = "participant-specific tokens REMOVED" if filt else "all tokens"
        print(f"\n  -- {label} --")
        # For the baseline the restriction is a fixed vocabulary ban rather than
        # a per-fold refit, because TfidfVectorizer is fitted on raw text, not
        # on the chunk table. It is computed from the chunks either way.
        drop = participant_specific_tokens(chunks) if filt else None
        p, t = tfidf_cv(texts, outcome, binary=binary, drop_tokens=drop)
        print("  " + report("TF-IDF + linear model", p, t, binary))
        for k in ks:
            p, t = slda_cv(chunks, outcome, binary=binary, k=k, iters=400,
                           drop_participant_specific=filt)
            print("  " + report(f"sLDA k={k}", p, t, binary))
        for k in ks[-2:]:
            p, t = lda_ridge_cv(chunks, outcome, binary=binary, k=k, iters=400,
                                drop_participant_specific=filt)
            print("  " + report(f"LDA topics + ridge k={k}", p, t, binary))

def main():
    d = pd.read_csv(DATA/'data_combined.csv', encoding='utf-8', encoding_errors='ignore')
    d = d[(d['Tracking'] != 'GAMER') & (d['Tracking'] != 'TEST')]
    h = d[['Prefix','Combined Text'] + list(HPSVQ_MAPS)].dropna().reset_index(drop=True)
    for c, mp in HPSVQ_MAPS.items():
        h[c] = h[c].replace(mp)
    h['hpsvq_total'] = h[list(HPSVQ_MAPS)].sum(axis=1).astype(float)
    compare("HPSVQ TOTAL (continuous)", h, 'Prefix', 'Combined Text', 'hpsvq_total', False)
    compare("IN TREATMENT (binary)", d[['Prefix','Combined Text','used-treatment']],
            'Prefix', 'Combined Text', 'used-treatment', True)

    coh = pd.read_csv(DATA/'coherencetext.csv', encoding='utf-8', encoding_errors='ignore')
    coh['uid'] = coh['file'].astype(str).str.split('@').str[0]
    compare("COHERENCE (continuous)", coh[['uid','text_manual','numscores']],
            'uid', 'text_manual', 'numscores', False)

    ema = pd.read_csv(DATA/'ematext.csv', encoding='utf-8', encoding_errors='ignore')
    for col, label in [('Experience-Volume','EMA LOUDNESS'), ('Appraisal-Power','EMA POWER')]:
        e = ema[['uid','Text',col]].copy()
        e = e[e[col] != 99]
        e = e.groupby('uid', as_index=False).agg({'Text': lambda s: ' '.join(map(str, s)), col: 'mean'})
        compare(f"{label} (participant mean)", e, 'uid', 'Text', col, False)

if __name__ == '__main__':
    main()
