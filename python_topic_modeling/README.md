# Python arm — topic models and the predictive comparison

Four topic-model families and a bag-of-words baseline, evaluated at participant
level under grouped cross-validation. This is the arm that answers *does topic
modelling earn its place predictively* — it does not.

Full topic listings, with the participant composition of every topic, are in
[`TOPICS.md`](TOPICS.md). Results below.

## Files

| | |
|---|---|
| `clean_text.py` | Encoding repair. Three separate corruptions, one unrepairable. |
| `build_corpus.py` | Tokenise, POS-filter, chunk, cap per participant, 298-term stop list, `participant_specific_tokens()`. |
| `topic_models.py` | LDA with a c_v/u_mass sweep, BERTopic, Top2Vec. |
| `evaluate.py` | Grouped CV for sLDA and LDA+ridge, the TF-IDF baseline, `tfidf_split_spread()`. |
| `run_analysis.py` | The predictive comparison, both filter arms. ~30 min. |
| `run_topics.py` | The topic comparison across four corpus variants. ~25 min. |

## Running

```bash
export SLDA_DATA_DIR=/path/to/datasets_final
.venv-topics/bin/python python_topic_modeling/run_analysis.py
.venv-topics/bin/python python_topic_modeling/run_topics.py
```

Check the repair works first — if this does not print a normal apostrophe,
nothing downstream is valid:

```bash
.venv-topics/bin/python -c "import sys; sys.path.insert(0,'python_topic_modeling'); from clean_text import repair; print(repair('I know itb\x19s been a couple of days'))"
```

---

## Prediction

Grouped 5-fold CV, participant level, split pinned with `shuffle=True,
random_state=seed`. Topic models are fitted on chunks; predictions averaged
within participant. Each model is run twice — once on all tokens, once with
participant-specific tokens removed inside every training fold, refitted per
fold so nothing leaks.

| Outcome | | TF-IDF + linear | best sLDA | best LDA + ridge | n |
|---|---|---|---|---|---|
| HPSVQ total | all tokens | **r +0.434** | +0.265 (k=6) | +0.108 (k=15) | 325 |
| | filtered | **r +0.434** | +0.306 (k=15) | +0.148 (k=15) | 325 |
| In treatment | all tokens | **AUC 0.646** | 0.543 (k=10) | 0.555 (k=10) | 325 |
| | filtered | **AUC 0.638** | 0.542 (k=6) | 0.501 (k=10) | 325 |
| Coherence | all tokens | **r +0.267** | +0.120 (k=10) | +0.153 (k=10) | 142 |
| | filtered | **r +0.267** | +0.122 (k=6) | +0.078 (k=15) | 142 |
| EMA loudness | all tokens | **r +0.249** | +0.030 (k=6) | -0.114 (k=15) | 268 |
| | filtered | **r +0.259** | +0.135 (k=6) | +0.058 (k=10) | 268 |
| EMA power | all tokens | r +0.035 | +0.028 (k=15) | -0.005 (k=10) | 265 |
| | filtered | r +0.039 | +0.162 (k=10) | +0.194 (k=10) | 265 |

TF-IDF wins every row except the last, and the last does not replicate. It is
also the only method beating a mean model: sLDA's MAE on HPSVQ is 7.25 against
5.68, rising to 10.53 at k=15. A correlation of +0.3 on a badly calibrated scale
is not a usable prediction.

**Filtering does not move the baseline** (0.434 → 0.434, 0.267 → 0.267), so the
negative result is not an artefact of the baseline exploiting participant-
identifying vocabulary.

### Split-to-split spread

Every row above is one split. The TF-IDF baseline over 25 random splits:

| Outcome | mean | sd | range |
|---|---|---|---|
| HPSVQ total | r +0.446 | 0.024 | +0.406 to +0.490 |
| In treatment | AUC 0.602 | 0.017 | 0.558 to 0.627 |
| Coherence | r +0.280 | 0.049 | +0.141 to +0.350 |
| EMA loudness | r +0.161 | 0.032 | +0.086 to +0.235 |
| EMA power | r -0.007 | 0.049 | -0.108 to +0.092 |

### The one apparent exception is noise

EMA power, filtered, showed sLDA at +0.162 and LDA+ridge at +0.194 against
TF-IDF's +0.039 — the only place a topic model beat the baseline. Across five
seeds:

| model | r per seed | mean | sd |
|---|---|---|---|
| sLDA k=10 | +0.162, -0.027, +0.014, 0.000, +0.083 | +0.046 | 0.068 |
| LDA+ridge k=10 | +0.194, -0.005, +0.104, +0.021, -0.028 | +0.057 | 0.082 |
| TF-IDF | +0.035, -0.057, +0.038, +0.060, -0.039 | +0.007 | — |

Best of five draws from a distribution on zero. EMA power has no signal in any
method.

---

## Topics

Four configurations, reported with participant composition because a word list
alone is not interpretable on this corpus:

| corpus | topics | unassigned | >50% one participant | median top share |
|---|---|---|---|---|
| uncapped | 13 | 32% | **7 of 13** | 53% |
| uncapped, filtered | 2 | 0% | 1 of 2 | 50% |
| capped | 4 | 51% | 0 of 4 | 6% |
| capped, filtered | 7 | 58% | **0 of 7** | 16% |

Capping is what makes topics generalise; vocabulary filtering alone does not.
All listings in [`TOPICS.md`](TOPICS.md).

### k is not identified by coherence

| k | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| mean c_v | .2876 | .2872 | .2874 | .2814 | .2821 | .2845 | .2831 | .2796 | .2807 |
| sd over 5 seeds | .0090 | .0064 | .0109 | .0063 | .0054 | .0070 | .0042 | .0063 | .0071 |

Variation across k (sd of means, 0.0029) is less than half the variation across
seeds at fixed k (0.0070); the five seeds pick k = 14, 20, 4, 8, 10.

---

## Per-outcome notebooks

Six notebooks, one per outcome, held with the data. They share this arm's
preprocessing — the same stop list, the participant filter fitted on training
rows only, and caps of 8 rows per participant and 1,200 tokens per document. All
run with zero errors.

| notebook | training rows | participants | vocab | largest participant's token share |
|---|---|---|---|---|
| hpsvq | 227 | 227 | 2,838 | 1.9% |
| in treatment | 227 | 227 | 2,926 | 1.8% |
| coherence | 149 | 48 | 401 | 4.9% |
| EMA loudness | 734 | 140 | 1,279 | 3.0% |
| EMA power | 731 | 138 | 1,324 | 3.7% |
| EMA negative affect | 725 | 140 | 1,263 | 3.1% |

Before capping the largest participant held 29–36% of training tokens and the
leading topic words were that person's vocabulary. After, the leading words are
*feel, hear, sound, sleep, kill, word, year* — shared vocabulary.

## Sentence extraction

Pulls example sentences containing theme keywords. 300 of 326 participants
match; **100%** of extracted passages contain a real keyword (substring and
case-sensitivity bugs had that at 75%).

Keywords are not hand-picked. Candidates come from the capped-filtered themes,
and each is dropped unless it appears in ≥20 participants with ≤50% from any
one. That audit currently keeps 21 of 30:

- **kept** — loud, louder, worse, quiet, head, sound, coming, inside, outside,
  hearing, schizophrenia, voice, spirit, demon, word, trigger, mind, thought,
  family, therapy, better
- **dropped** — volume, diagnosed (59% one participant), shadow (53%), entity,
  bible (57%), speech, coping, medication (54%), counselor (91%)

Re-run it after any change to the topics, and read the audit output.

---

## Bugs fixed

Inference and evaluation:

1. `make_doc(words=str(tokens))` passed the string repr of the token list;
   `tomotopy` iterated it character by character, so every test document arrived
   with zero in-vocabulary tokens and predictions collapsed to the training mean.
   This invalidated everything reported before it was found.
2. `stratify=uid` instead of grouping, in four notebooks → `GroupShuffleSplit`.
3. Inconsistent stop lists between train and test.
4. `GroupKFold` fold assignment left to the scikit-learn version → pinned.
5. Empty documents segfault `tomotopy` rather than raising → guarded.

Preprocessing:

6. 942 corrupted punctuation marks across 74 documents (`it's` → `itbs`).
7. Curly apostrophes in 390 transcripts, which `stop_words` misses because it
   holds the straight-quote forms — `i'm` survived at 1,022 occurrences.
8. Doubly cp1252-mangled UTF-8 in 139 rows, needing two round trips.
9. The apostrophe-to-`u` corruption family (`itus`, `ium`, `donut`, `theyure`)
   was handled by hand in notebooks and never reached the pipeline. Now
   generated by rule, 203 forms.
10. 95 disfluency and filler stop words added; substring, case-sensitive keyword
    matching replaced with word-boundary matching.

Topic models:

11. `min_cluster_size=150` on ~2,700 documents sent 61% to the outlier bucket.
12. Top2Vec 1.x ignores `speed` unless `embedding_model='doc2vec'` is explicit.
13. Top2Vec's default `umap_args` omits `random_state`, leaving UMAP on its
    parallel numba path, which **segfaults the interpreter** during
    "Creating lower dimension embedding of documents" — no traceback, no exit code.
14. BERTopic's c-TF-IDF vectoriser is fitted on one synthetic document *per
    topic*, so `min_df=3` is a floor on the number of topics; runs finding fewer
    than three died with "max_df corresponds to < documents than min_df".
15. LDA k-sweeps at `passes=2`, never converging.
