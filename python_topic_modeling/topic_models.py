"""Topic models rebuilt on the shared preprocessing.

Every previous attempt in this project failed the same way, and it was not the
methods' fault. Disfluencies and contraction fragments were never removed, so
LDA returned topics that were all "im / know / dont / like / voic", and BERTopic
named its clusters "um know like uh" while assigning 61% of documents to the
outlier bucket. build_corpus.CUSTOM_STOP now strips those, so each method gets a
fair run and the comparison between them means something.

Everything here writes topic terms out as text. The earlier Top2Vec notebook
only ever rendered wordclouds, so its topics could not be read back from the
saved file at all.
"""
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_corpus import tokenise, CUSTOM_STOP
from clean_text import repair


def _docs_from_chunks(chunks):
    """BERTopic and Top2Vec embed raw text, so rejoin the filtered tokens
    rather than passing the original string. That way the disfluency removal
    applies to the embedding step too, not only to the topic representation."""
    return [' '.join(t) for t in chunks['tokens']]


def lda_topics(chunks, k_range=range(4, 21, 2), passes=15, iterations=400, seed=42):
    """LDA with c_v coherence over a range of k.

    The original sweeps used passes=2 and logged 'too few updates, training
    might not converge' on every fit, which is why one curve was flat to within
    0.005 and the other swung between -1.0 and -2.1. Raising passes and
    iterations is what makes the comparison between k values meaningful.
    """
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel
    texts = list(chunks['tokens'])
    d = Dictionary(texts)
    d.filter_extremes(no_below=5, no_above=0.5)
    corpus = [d.doc2bow(t) for t in texts]
    rows = []
    for k in k_range:
        m = LdaModel(corpus=corpus, id2word=d, num_topics=k, passes=passes,
                     iterations=iterations, random_state=seed, alpha='auto')
        # processes=1: gensim's default spawns workers, which breaks when this
        # module is driven from a non-file context and is not worth the overhead
        # at this corpus size.
        cv = CoherenceModel(model=m, texts=texts, dictionary=d,
                            coherence='c_v', processes=1).get_coherence()
        um = CoherenceModel(model=m, corpus=corpus, dictionary=d,
                            coherence='u_mass', processes=1).get_coherence()
        rows.append({'k': k, 'c_v': cv, 'u_mass': um, 'model': m})
        print(f"   k={k:3d}   c_v {cv:.4f}   u_mass {um:.3f}")
    best = max(rows, key=lambda r: r['c_v'])
    print(f"\n   best by c_v: k={best['k']}  ({best['c_v']:.4f})")
    for t in range(best['model'].num_topics):
        words = [w for w, _ in best['model'].show_topic(t, topn=12)]
        print(f"     topic {t:2d}: {', '.join(words)}")
    return pd.DataFrame([{c: r[c] for c in ('k', 'c_v', 'u_mass')} for r in rows]), best['model'], d


def bertopic_topics(chunks, min_topic_size=25, seed=42, quiet=False):
    """BERTopic with a cluster size suited to this corpus.

    The original used HDBSCAN(min_cluster_size=150) on ~2,700 documents, which
    forces a couple of huge clusters and sends everything else to the outlier
    bucket. 25 is a more reasonable floor here. The vectorizer is given the
    project stop list so topic labels are content words rather than filler.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    docs = _docs_from_chunks(chunks)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP)
    # min_df=1, not 3. BERTopic's c-TF-IDF vectoriser is fitted on ONE synthetic
    # document per topic, not on the corpus, so min_df is a floor on the number
    # of topics rather than on term frequency. With min_df=3 any run that finds
    # fewer than three topics dies with "max_df corresponds to < documents than
    # min_df", which is what happened on the filtered uncapped corpus.
    vec = CountVectorizer(stop_words=stop, min_df=1, ngram_range=(1, 2))
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                metric='cosine', random_state=seed)
    hdb = HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean',
                  cluster_selection_method='eom', prediction_data=True)
    model = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec,
                     ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(docs)
    info = model.get_topic_info()
    n_out = int((np.array(topics) == -1).sum())
    print(f"   {len(info) - 1} topics, {n_out} of {len(docs)} documents unassigned "
          f"({n_out / len(docs) * 100:.0f}%)")
    if not quiet:
        for t in sorted(set(topics)):
            if t == -1:
                continue
            words = [w for w, _ in model.get_topic(t)][:12]
            print(f"     topic {t:2d} (n={topics.count(t):4d}): {', '.join(words)}")
    return model, info


def top2vec_topics(chunks, speed='learn', seed=42, quiet=False):
    """Top2Vec, writing topic terms out as text.

    The original notebook only produced wordclouds, and one output cell was
    saved as 'Output hidden', so its topics were unreadable from the file.
    """
    from top2vec import Top2Vec
    docs = _docs_from_chunks(chunks)
    # embedding_model is explicit because Top2Vec 1.x defaults to
    # 'all-MiniLM-L6-v2' and then ignores `speed`, which is a doc2vec setting.
    # doc2vec keeps this arm faithful to the original notebook, so any change in
    # the result is attributable to the preprocessing rather than to a different
    # embedding. min_count is 5 rather than the 1.x default of 50 because the
    # corpus is ~2.6k short chunks, not a web-scale collection.
    # umap_args is not optional here. Top2Vec's default omits random_state,
    # which leaves UMAP on its parallel numba path; on this corpus that
    # segfaults the interpreter during "Creating lower dimension embedding of
    # documents" - no traceback, no exit code, the run just stops. Seeding it
    # takes the single-threaded path and makes the topics reproducible as well.
    model = Top2Vec(docs, embedding_model='doc2vec', speed=speed,
                    workers=4, min_count=5, verbose=False,
                    umap_args={'n_neighbors': 15, 'n_components': 5,
                               'metric': 'cosine', 'random_state': seed})
    n = model.get_num_topics()
    if not quiet:
        sizes, nums = model.get_topic_sizes()
        words, _, _ = model.get_topics(n)
        print(f"   {n} topics")
        for i, (tn, sz) in enumerate(zip(nums, sizes)):
            print(f"     topic {tn:2d} (n={sz:4d}): {', '.join(words[i][:12])}")
    return model
