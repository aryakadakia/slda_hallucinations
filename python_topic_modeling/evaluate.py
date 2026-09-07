"""Grouped cross-validation for chunked sLDA, against a bag-of-words baseline.

Design
  Topics are learned over chunks, each chunk inheriting its participant's
  outcome. Predictions are made per chunk and then averaged within participant,
  so evaluation happens at participant level. Folds are split on participant id,
  so no one appears on both sides.

The TF-IDF baseline exists to answer the question a reviewer asks first: does
the topic model beat a linear model on raw words? If it does not, the topics
may still be interpretable but they are not earning their place predictively.
"""
import sys, time, numpy as np, pandas as pd, tomotopy as tp
from pathlib import Path
from sklearn.model_selection import GroupKFold

# GroupKFold's unshuffled fold assignment is an implementation detail, and it
# changed between scikit-learn 1.6 and 1.9: on the same 325 participants the two
# versions put people in different folds, which moved the headline HPSVQ TF-IDF
# correlation from +0.39 to +0.48. Nothing about the data or the model changed.
# Pinning shuffle and random_state makes the split a stated choice rather than a
# property of whichever scikit-learn happens to be installed. It also means the
# single-split numbers below need reading with the spread in mind - see
# tfidf_split_spread() for how wide that is.
def _splitter(n_splits, seed):
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from build_corpus import build, participant_specific_tokens


def _fold_drop_set(train_chunks, enabled):
    """Tokens to remove, computed on the TRAINING fold only.

    participant_specific_tokens over the whole corpus would let the test fold
    influence the vocabulary, which is a transductive leak. It is harmless for
    the descriptive topic runs in run_topics.py, but not here, so the drop set
    is refitted inside every fold.
    """
    return participant_specific_tokens(train_chunks) if enabled else set()


def _apply_drop(chunks, drop):
    if not drop:
        return chunks
    out = chunks.copy()
    out['tokens'] = [[w for w in t if w not in drop] for t in out['tokens']]
    out['n_tokens'] = out['tokens'].str.len()
    return out[out.n_tokens > 0].reset_index(drop=True)

def _fit_slda(train, k, binary, iters, seed):
    vars_ = 'b' if binary else 'l'
    m = tp.SLDAModel(tw=tp.TermWeight.IDF, k=k, alpha=0.1, eta=0.01,
                     vars=vars_, seed=seed)
    for toks, y in zip(train['tokens'], train['y']):
        m.add_doc(words=toks, y=[float(y)])
    m.burn_in = 100
    m.train(0)
    m.train(iters)
    return m

def slda_cv(chunks, outcome, binary, k, n_splits=5, iters=400, seed=42,
            drop_participant_specific=False):
    """Return participant-level predictions and truths across all folds."""
    chunks = chunks.copy()
    chunks['y'] = chunks['pid'].map(outcome)
    chunks = chunks.dropna(subset=['y'])
    gkf = _splitter(n_splits, seed)
    preds, truths = [], []
    for tr, te in gkf.split(chunks, groups=chunks['pid']):
        train, test = chunks.iloc[tr], chunks.iloc[te]
        drop = _fold_drop_set(train, drop_participant_specific)
        train, test = _apply_drop(train, drop), _apply_drop(test, drop)
        if not len(test):
            continue
        m = _fit_slda(train, k, binary, iters, seed)
        docs = [m.make_doc(words=t, y=[0.0]) for t in test['tokens']]
        m.infer(docs)
        est = np.array([m.estimate(d)[0] for d in docs], float)
        out = pd.DataFrame({'pid': test['pid'].values, 'pred': est})
        agg = out.groupby('pid')['pred'].mean()
        preds.append(agg)
        truths.append(pd.Series({p: outcome[p] for p in agg.index}))
    p = pd.concat(preds); t = pd.concat(truths)
    return p.values.astype(float), t.values.astype(float)

def tfidf_cv(texts, outcome, binary, n_splits=5, seed=42, drop_tokens=None):
    """drop_tokens applies the same vocabulary restriction the topic models get,
    so the baseline is not quietly allowed to use terms they were denied."""
    pids = np.array(list(texts.keys()))
    X_raw = np.array([texts[p] for p in pids])
    y = np.array([outcome[p] for p in pids], float)
    keep = ~np.isnan(y); pids, X_raw, y = pids[keep], X_raw[keep], y[keep]
    gkf = _splitter(n_splits, seed)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X_raw, groups=pids):
        stop = 'english' if not drop_tokens else sorted(
            set(ENGLISH_STOP_WORDS) | set(drop_tokens))
        vec = TfidfVectorizer(max_features=5000, min_df=2, stop_words=stop,
                              sublinear_tf=True)
        Xtr = vec.fit_transform(X_raw[tr]); Xte = vec.transform(X_raw[te])
        if binary:
            mdl = LogisticRegression(max_iter=2000, C=1.0)
            mdl.fit(Xtr, y[tr]); preds[te] = mdl.predict_proba(Xte)[:, 1]
        else:
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12))
            mdl.fit(Xtr, y[tr]); preds[te] = mdl.predict(Xte)
    return preds, y

def report(name, pred, truth, binary):
    if binary:
        auc = roc_auc_score(truth, pred) if len(set(truth)) > 1 else float('nan')
        acc = ((pred > 0.5).astype(int) == truth.astype(int)).mean()
        base = max(truth.mean(), 1 - truth.mean())
        return f"{name:26s} AUC {auc:.3f}   acc {acc:.3f} (baseline {base:.3f})   n={len(truth)}"
    r = np.corrcoef(pred, truth)[0, 1] if np.std(pred) > 1e-9 else float('nan')
    mae = np.abs(pred - truth).mean()
    base = np.abs(truth - truth.mean()).mean()
    return f"{name:26s} r {r:+.3f}   MAE {mae:.2f} (mean-model {base:.2f})   n={len(truth)}"

def lda_ridge_cv(chunks, outcome, binary, k, n_splits=5, iters=500, seed=42,
                 drop_participant_specific=False):
    """Unsupervised LDA for the topics, then a regularised regression on the
    participant-averaged topic proportions.

    Often stronger than sLDA's built-in regression, because the topics are not
    pulled toward the outcome during fitting (less overfitting on a small n) and
    the downstream model is properly regularised and cross-validated.
    """
    chunks = chunks.copy()
    chunks['y'] = chunks['pid'].map(outcome)
    chunks = chunks.dropna(subset=['y'])
    gkf = _splitter(n_splits, seed)
    preds, truths = [], []
    for tr, te in gkf.split(chunks, groups=chunks['pid']):
        train, test = chunks.iloc[tr], chunks.iloc[te]
        drop = _fold_drop_set(train, drop_participant_specific)
        train, test = _apply_drop(train, drop), _apply_drop(test, drop)
        if not len(test):
            continue
        m = tp.LDAModel(tw=tp.TermWeight.IDF, k=k, alpha=0.1, eta=0.01, seed=seed)
        for toks in train['tokens']:
            m.add_doc(toks)
        m.burn_in = 100; m.train(0); m.train(iters)

        def feats(frame):
            docs = [m.make_doc(words=t) for t in frame['tokens']]
            m.infer(docs)
            T = np.vstack([d.get_topic_dist() for d in docs])
            f = pd.DataFrame(T, index=frame['pid'].values)
            return f.groupby(level=0).mean()

        Ftr, Fte = feats(train), feats(test)
        ytr = np.array([outcome[p] for p in Ftr.index], float)
        yte = np.array([outcome[p] for p in Fte.index], float)
        if binary:
            mdl = LogisticRegression(max_iter=2000)
            mdl.fit(Ftr.values, ytr); pr = mdl.predict_proba(Fte.values)[:, 1]
        else:
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12))
            mdl.fit(Ftr.values, ytr); pr = mdl.predict(Fte.values)
        preds.append(pd.Series(pr, index=Fte.index)); truths.append(pd.Series(yte, index=Fte.index))
    p = pd.concat(preds); t = pd.concat(truths)
    return p.values.astype(float), t.values.astype(float)


def tfidf_split_spread(texts, outcome, binary, n_repeats=25, n_splits=5):
    """How much of a reported correlation is the split rather than the signal.

    Every number in RESULTS.md comes from one 5-fold split. Re-running the same
    baseline over different random splits shows how far that single figure can
    travel on its own, which is the interval a headline r should be read
    against.
    """
    vals = [tfidf_cv(texts, outcome, binary=binary, n_splits=n_splits, seed=s)
            for s in range(n_repeats)]
    if binary:
        scores = [roc_auc_score(t, p) for p, t in vals]
    else:
        scores = [np.corrcoef(p, t)[0, 1] for p, t in vals]
    scores = np.array(scores)
    return {'mean': scores.mean(), 'sd': scores.std(),
            'min': scores.min(), 'max': scores.max(), 'n': n_repeats}
