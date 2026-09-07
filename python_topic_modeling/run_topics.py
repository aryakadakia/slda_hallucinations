"""Topic-model comparison on the shared preprocessing.

Usage:  SLDA_DATA_DIR=/path/to/datasets_final  python3 python_topic_modeling/run_topics.py

Every topic is reported with how many participants it draws on and what share
comes from its largest contributor. Without that column a topic list is not
interpretable: the largest topic in the first working run was 95% one person,
and all three method families reproduced it because all three were looking at
the same 29% of the corpus.

Two corpus variants are run:

  capped    8 chunks per participant, which is what the predictive pipeline
            uses. Capping equalises contribution (the largest participant falls
            from 29% of tokens to 1.3%) but halves the corpus, and HDBSCAN
            needs the density it removes.
  uncapped  every chunk. More to cluster, but dominated by one participant.

and each is run with and without participant-specific tokens removed.
"""
import os, sys, traceback, warnings
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_corpus import build, participant_specific_tokens
from topic_models import lda_topics

DATA = Path(os.environ.get('SLDA_DATA_DIR', '.'))


def concentration(chunks, assignments):
    """Per topic: size, participants, and the largest participant's share.

    `assignments` is one topic id per row of `chunks`, with -1 for unassigned.
    """
    frame = chunks.assign(topic=assignments)
    rows = []
    for t in sorted(set(assignments)):
        if t == -1:
            continue
        sub = frame[frame.topic == t]
        counts = sub.pid.value_counts()
        rows.append({'topic': t, 'n': len(sub), 'participants': sub.pid.nunique(),
                     'top_share': counts.iloc[0] / len(sub)})
    return pd.DataFrame(rows)


def _report(name, chunks, assignments, words_for):
    conc = concentration(chunks, assignments)
    if not len(conc):
        print("   no topics")
        return
    dominated = (conc.top_share > 0.5).sum()
    print(f"   {len(conc)} topics; {dominated} of them >50% one participant; "
          f"median largest-participant share {conc.top_share.median():.0%}")
    for _, r in conc.iterrows():
        flag = '  <-- one participant' if r.top_share > 0.5 else ''
        print(f"     topic {int(r.topic):2d} (n={int(r.n):4d}, "
              f"{int(r.participants):3d} participants, top {r.top_share:.0%}){flag}: "
              f"{', '.join(words_for(int(r.topic))[:10])}")


def run_bertopic(chunks, label, min_topic_size=15):
    from topic_models import bertopic_topics
    print(f"\nBERTopic  {label}  min_topic_size={min_topic_size}:")
    model, _ = bertopic_topics(chunks, min_topic_size=min_topic_size, quiet=True)
    topics = model.topics_
    _report('BERTopic', chunks, topics,
            lambda t: [w for w, _ in model.get_topic(t)])
    return model


def run_top2vec(chunks, label):
    from topic_models import top2vec_topics
    print(f"\nTop2Vec  {label}:")
    model = top2vec_topics(chunks, quiet=True)
    n = model.get_num_topics()
    words, _, _ = model.get_topics(n)
    doc_topics = model.get_documents_topics(list(range(len(chunks))))[0]
    _report('Top2Vec', chunks, list(doc_topics), lambda t: list(words[t]))
    return model


def main():
    d = pd.read_csv(DATA / 'data_combined.csv', encoding='utf-8', encoding_errors='ignore')
    d = d[(d['Tracking'] != 'GAMER') & (d['Tracking'] != 'TEST')].dropna(subset=['Combined Text'])

    corpora = {}
    for cap_label, cap in (('capped', 8), ('uncapped', None)):
        for filt in (False, True):
            c = build(d, 'Prefix', 'Combined Text', max_chunks_per_pid=cap,
                      drop_participant_specific=filt)
            name = f"{cap_label}{', filtered' if filt else ''}"
            corpora[name] = c
            share = c.groupby('pid').n_tokens.sum()
            print(f"corpus {name:22s} {len(c):5d} chunks, {c.pid.nunique():3d} "
                  f"participants, largest {share.max()/share.sum()*100:4.1f}% of tokens")

    drop = participant_specific_tokens(corpora['uncapped'])
    vocab = {w for t in corpora['uncapped'].tokens for w in t}
    print(f"\nparticipant-specific token types: {len(drop)} of {len(vocab)} "
          f"({len(drop)/len(vocab)*100:.0f}% of vocabulary)")

    print("\nLDA, c_v and u_mass over k (capped, filtered):")
    print("  c_v does not identify k on this corpus - across five seeds the")
    print("  variation between k values is less than half the variation between")
    print("  seeds at fixed k. Reported for completeness only.")
    lda_topics(corpora['capped, filtered'], k_range=range(4, 21, 2),
               passes=15, iterations=400)

    for name in ('uncapped', 'uncapped, filtered', 'capped', 'capped, filtered'):
        try:
            run_bertopic(corpora[name], name)
        except ImportError:
            print("\nBERTopic: not installed, skipping"); break
        except Exception:
            traceback.print_exc()

    for name in ('uncapped', 'uncapped, filtered'):
        try:
            run_top2vec(corpora[name], name)
        except ImportError:
            print("Top2Vec: not installed, skipping"); break
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
