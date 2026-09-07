# Topic models of auditory verbal hallucination diaries

Supervised topic modelling applied to spoken diaries from people who hear
voices, asking whether the *way* someone describes the experience tracks with
symptom severity.

Standard LDA finds themes without reference to an outcome. Supervised LDA fits
the topics and a regression on a response variable together, so the themes that
emerge are the ones carrying signal about the outcome. The question is not what
people talk about, but which ways of describing an experience predict severity
and adverse outcomes.

**The headline result is negative, and that is the finding.** A TF-IDF
bag-of-words model with ridge regression beats supervised LDA on every outcome
tested. Topic models remain descriptively useful — seven interpretable,
reproducible themes come out of the corpus — but they do not earn their place
predictively on data of this size.

---

## Repository

| Path | |
|---|---|
| [`python_topic_modeling/`](python_topic_modeling/) | LDA, BERTopic, Top2Vec, sLDA via `tomotopy`, and the TF-IDF baseline, under grouped cross-validation. The predictive comparison. |
| [`python_topic_modeling/TOPICS.md`](python_topic_modeling/TOPICS.md) | Every topic from all four methods, with participant composition. |
| [`r_slda/`](r_slda/) | sLDA via `lda::slda.em`. Coefficients with standard errors and p-values, which `tomotopy` does not provide. The inferential arm. |
| [`DATA.md`](DATA.md) | What the inputs are, and the defects in the source text. |

The two arms are separate analyses on different units — participant-level in
Python, diary-level in R — and their numbers are not interchangeable. See
*Two arms* below.

## Data

Participant transcripts are not included and will not be released. They are
identifiable clinical research data collected under consent that does not permit
publication, and are held outside this repository. Both arms read their inputs
from a directory named by the `SLDA_DATA_DIR` environment variable.

Nothing here depends on that directory being present until you run something;
the results are written out in full.

## Outcomes

Six response variables from three sources: HPSVQ total score and treatment
status from baseline assessment; loudness, power and negative affect from
ecological momentary assessment; and a manually scored coherence rating.

---

## Method

Transcripts are repaired for encoding damage, lemmatised, POS-tagged to nouns,
verbs and adjectives, and stripped of a 298-term stop list covering disfluency,
contraction fragments and corruption artefacts. Documents are split into
150-word chunks and **capped at 8 chunks per participant**.

Topics are learned over chunks; each participant's chunk-level topic proportions
are averaged so the regression stays at participant level and no outcome is
split across the train/test boundary. Folds are grouped on participant, so
nobody appears on both sides.

## Key insights

**1. Topic modelling does not beat bag-of-words here.**

| Outcome | TF-IDF + ridge | best sLDA | best LDA + ridge | n |
|---|---|---|---|---|
| HPSVQ total | **r +0.434** | +0.306 | +0.148 | 325 |
| In treatment | **AUC 0.646** | 0.543 | 0.555 | 325 |
| Coherence | **r +0.267** | +0.122 | +0.153 | 142 |
| EMA loudness | **r +0.259** | +0.135 | +0.058 | 268 |
| EMA power | r +0.039 | — | — | 265 |

TF-IDF is also the only method that beats predicting the mean. sLDA's error is
*worse* than a mean model on HPSVQ and grows with the number of topics. Likely
reasons: 325 participants each contributing one outcome is a small sample for
fitting topics and a regression jointly, and compressing a 10,000-word
vocabulary into 6–15 topics discards detail ridge exploits directly.

**2. Corpus concentration decides what the topics are.** One participant
recorded 386 of the 3,090 diaries and holds 29% of all tokens. Uncapped, the
largest topic in BERTopic, Top2Vec and sLDA alike is that one person — 95% of
its documents in BERTopic's case — with near-identical words across all three.
That looked like strong cross-method convergence on a clinical theme. It was
three methods looking at the same 29% of the corpus. Capping contribution per
participant is what fixes it; filtering their vocabulary is not enough. Every
topic in this repository is reported with how many participants it draws on.

**3. Seven themes survive that test**, consistent across methods: describing the
voices (loudness, location, repetition); hostile and profane speech; religious
and spiritual imagery; threatening imagery; illness, treatment and coping;
word-finding and state of mind; surveillance and technology.

**4. Topic composition is associated with severity in sample, and does not
predict out of sample.** On HPSVQ, against a *voices, hear, hearing* reference
topic, the *mental illness* topic sits +16.5 and *god, church* +15.9 on an 8–35
scale, both overwhelmingly significant. The same models fail on held-out
participants. Both halves are worth reporting.

**5. Coherence is the one outcome worth following up.** It is the only model in
the project with genuine held-out signal (r +0.277, out-of-sample R² +0.076), it
appears only in the diary-level R arm, and only once participant-specific
vocabulary is removed.

## Three cautions that apply to every number here

- **Coherence does not identify the number of topics.** Across five seeds the
  variation in c_v between values of k is less than half the variation between
  seeds at fixed k, and each seed picks a different optimum. Choose k on
  interpretability.
- **Single-split results move.** The TF-IDF baseline varies by ±0.05 across
  random splits on coherence. The one place a topic model appeared to beat the
  baseline was the best of five seeds from a distribution centred on zero.
- **Environment changes results.** `GroupKFold`'s unshuffled fold assignment
  changed between scikit-learn 1.6 and 1.9 and moved a headline correlation from
  +0.39 to +0.48 on its own. The splitter is now pinned explicitly.

---

## Two arms

| | `python_topic_modeling/` | `r_slda/` |
|---|---|---|
| Unit | participant (325) | diary entry (2,097 train / 517 test) |
| Validation | grouped 5-fold CV, pinned split | fixed split, zero participant overlap |
| Implementation | `tomotopy`, collapsed Gibbs | `lda::slda.em`, variational EM |
| Gives | held-out prediction against a baseline | coefficients with standard errors |

Predicting one outcome per participant from all of their text is harder than
predicting a score attached to each diary entry, where the same person's score
repeats across rows. Diary-level correlations are the easier result, not the
better one, and the two tables must not be merged.

Both arms share preprocessing: `slda_stopwords.R` is the R port of
`clean_text.py` and the stop list in `build_corpus.py`. **Keep them in step** —
if a term is added on one side it belongs on the other, or the arms stop being
comparable.

## Reproducing

```bash
export SLDA_DATA_DIR=/path/to/datasets_final
python3 -m venv .venv-topics
.venv-topics/bin/pip install -r python_topic_modeling/requirements.txt
```

Then see the README in each folder. End to end is about an hour, most of it the
topic-model sweep.

## Corrections to the earlier analysis

This is a re-analysis. The original code ran, produced plausible numbers, and
was wrong in ways that were invisible from its output. The largest:

- Inference was run on a string, not a token list. `make_doc(words=str(tokens))`
  made `tomotopy` iterate character by character, so every test document arrived
  with zero in-vocabulary tokens and predictions collapsed to the training mean.
- The reported R² was `var(yhat)/var(y)`, which never compares a prediction to
  its own observation, is unchanged by permuting the predictions, and can exceed 1.
- R test documents were indexed against a fresh vocabulary, so word ids addressed
  different columns of the fitted topic matrix.
- p-values came from a model with no intercept, testing each topic against zero
  rather than against the other topics.
- A coherence "peak" that selected k was seed noise.
- Cross-method agreement on a theme was one participant.

Each produced a publishable-looking result that meant nothing, and each was
caught by a check that takes minutes. Full list in
[`python_topic_modeling/README.md`](python_topic_modeling/README.md).

## Stack

Python 3.13 — `tomotopy`, `gensim`, `bertopic`, `top2vec`, `scikit-learn`,
`nltk`, `pandas`. R 4.6 — `lda`, `tidytext`, `dplyr`, `rsample`.
