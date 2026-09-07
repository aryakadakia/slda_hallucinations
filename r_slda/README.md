# R arm — supervised LDA with inference

Four supervised LDA models via `lda::slda.em`, at **diary-entry level**. This
arm exists for one reason: `slda.em` returns coefficients with standard errors
and p-values, and `tomotopy` does not. The question the paper asks — which
themes are associated with severity — is inferential, and this is the only place
in the project that answers it.

These numbers are **not comparable** to the participant-level table in
`python_topic_modeling/`. Different unit, different task, easier target.

## Files

| | |
|---|---|
| `slda_stopwords.R` | R port of `clean_text.py` and the stop list in `build_corpus.py`, plus `participant_specific_tokens_r()` and `prepare_documents()`. |
| `sLDA_hpsvq_R.R` | HPSVQ total score |
| `sLDA_inpatient_R.R` | Inpatient treatment |
| `sLDA_used_treatment_R.R` | Used treatment |
| `sLDA_coherence_R.R` | Coherence rating |
| `results/` | Coefficient and contrast tables per outcome |
| `figures/` | Coefficient plot and prediction density per outcome |

`prepare_documents()` is used for **both** train and test. Writing those steps
out twice per script is how the two halves originally ended up with different
stop lists. **Keep `slda_stopwords.R` in step with
`python_topic_modeling/build_corpus.py`** — a term added on one side belongs on
the other.

## Running

```bash
export SLDA_DATA_DIR=/path/to/datasets_final
Rscript r_slda/sLDA_hpsvq_R.R    # and _inpatient_, _used_treatment_, _coherence_
```

---

## Setup

2,097 training documents from 117 participants; 517 test documents from 133;
**zero participant overlap**. The training corpus is 35.6% one participant
before filtering — worse concentration than the Python corpus — so the
participant-specific filter is fitted on training rows only and applied to both
sides.

`sLDA_coherence_R.R` builds its own participant-grouped split from
`coherencetext.csv`; it originally read two files that never existed and had
never run.

## Held-out performance

| Script | r | out-of-sample R² | AUC | in-sample adj. R² | omnibus F |
|---|---|---|---|---|---|
| `sLDA_coherence_R.R` | **+0.277** | **+0.076** | — | 0.183 | F(7,213)=8.1, p=1e-08 |
| `sLDA_hpsvq_R.R` | +0.049 | -0.408 | — | 0.238 | F(7,2089)=94.5, p=4e-120 |
| `sLDA_inpatient_R.R` | -0.027 | -0.118 | 0.502 (maj. 0.542) | 0.281 | F(7,2089)=117.9, p=5e-146 |
| `sLDA_used_treatment_R.R` | -0.039 | -0.171 | 0.522 (maj. 0.752) | 0.263 | F(7,2089)=108.1, p=2e-135 |

Topic composition explains real variance **in sample** everywhere — every
omnibus test is overwhelming, adjusted R² 0.18–0.28. Three of four do not
generalise: a negative out-of-sample R² means the model does worse than
predicting the training mean, and both AUCs are at chance.

**Coherence is the exception**, and the only model in the project with genuine
held-out signal. It appears only with the participant filter on — unfiltered it
was r -0.027, R² -0.081. Coherence is scored per diary entry, so this arm
matches how the outcome was actually measured, whereas aggregating to participant
level averages away the variation being predicted. Worth following up, and it
needs a second split before it is worth much.

## Topic contrasts — the testable claim

`slda.em` fits `lm(clinical ~ . - 1)`, without an intercept, so each coefficient
is the fitted outcome for a document made entirely of that topic and each
p-value tests it against **zero**. On a score bounded at 8 they are significant
by construction and say nothing about association.

Each script therefore also fits an intercept model in which the lowest-scoring
topic becomes the reference, so every coefficient is a *difference between
topics*. That is the contrast the question asks. Written to
`results/<outcome>_topic_contrasts.csv`; the raw no-intercept fit is kept in
`results/<outcome>_topic_coefficients.csv` for reference.

HPSVQ, reference = *voices, hear, hearing, talking*, scale 8–35:

| difference vs reference | p | topic |
|---|---|---|
| +16.47 ± 0.91 | 1e-67 | mental, illness, life, taking, people |
| +15.87 ± 0.98 | 3e-55 | god, church, child, life, devil |
| +14.29 ± 0.94 | 3e-49 | stuff, house, started, guy, leave |
| +13.45 ± 1.04 | 7e-37 | apartment, talking, money, food, basically |
| +7.64 ± 0.86 | 1e-18 | voices, hear, voice, hearing, head |
| +7.46 ± 0.94 | 3e-15 | morning, night, day, sleep, medication |
| +4.00 ± 0.95 | 3e-05 | people, talking, feel, fucking, control |

The ordering repeats for in-treatment and used-treatment: the *mental illness*
and *god/church* topics carry the largest contrasts, and topics that plainly
describe hearing voices carry the smallest.

**Describing the voices themselves goes with lower severity; describing illness,
treatment and religion goes with higher.** That association is real in sample
and does not predict held-out participants — both halves belong in the write-up.

## Bugs fixed

1. **Test documents indexed against the wrong vocabulary.** `lexicalize(df$text)`
   with no `vocab=` builds a fresh index in first-seen order, so the word ids
   addressed different columns of `slda_mod$topics` than the model was fitted on.
   Predictions were read off unrelated word–topic associations.
2. **The reported R² was not an R².** `var(yhat)/var(y)` never compares a
   prediction to its own observation, is unchanged by permuting `yhat` (0.132 on
   five permutations), and can exceed 1. The figures 0.181/0.092/0.086/0.062
   previously carried as R² were that ratio.
3. **p-values from a no-intercept model** tested each topic against zero → the
   contrast model above.
4. **`sLDA_coherence_R.R` had never run** — it read two nonexistent files.
5. **A six-word stop list**, and no encoding repair. `stop_words` contains `i'm`
   with a straight apostrophe; 390 transcripts carry the curly form, so `i'm`
   survived as the sixth commonest token in the corpus at 1,022 occurrences.
6. Made runnable non-interactively: `view()` → `print()`, `qplot` (deprecated in
   ggplot2 4.x) → `ggplot` + `ggsave`.
