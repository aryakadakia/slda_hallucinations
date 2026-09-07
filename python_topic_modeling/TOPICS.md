# Topics

The complete topic output from all four methods, with the participant
composition of every topic. This file exists because a topic word list on its
own is not interpretable on this corpus: one participant recorded 386 of the
3,090 diaries and holds 29% of the tokens, and uncapped, every method returns
that person as its largest topic.

Every topic below is drawn from the **capped** corpus (8 chunks per participant,
150 words each), which is the only configuration where topics generalise across
people. `n` is documents assigned, `participants` how many distinct people
contributed them, `top` the largest single contributor's share.

A note on what is *not* here: the uncapped topic lists are omitted deliberately.
They are dominated by individual participants and carry first names, home towns
and specific medications — quasi-identifying for a clinical cohort. Their
statistics are in the last section; their words are not reproduced.

---

## LDA (gensim / tomotopy)

### k = 6

| n | participants | top | words |
|---|---|---|---|
| 470 | 241 | 2% | little, today, hear, talking, fucking, bad, hearing, saying, thank, feel |
| 84 | 47 | 7% | word, part, positive, thought, question, control, experience, end, find, past |
| 56 | 34 | 12% | year, love, stuff, child, old, man, mental, guy, felt, church |
| 35 | 25 | 11% | leave, house, scared, want, coming, illness, running, tired, someone, help |
| 34 | 22 | 9% | message, angel, game, therapy, telephone, god, happened, woman, story, name |
| 34 | 20 | 18% | team, system, guy, using, cigarette, done, hallucination, government, money, drug |

### k = 8

| n | participants | top | words |
|---|---|---|---|
| 381 | 215 | 2% | feel, talking, hear, head, leave, today, bye, feeling, negative, loud |
| 77 | 46 | 9% | question, male, thought, answer, different, angel, research, friend, team, believe |
| 74 | 49 | 7% | little, morning, wanted, thanks, trying, woke, need, find, message, hope |
| 51 | 31 | 14% | fucking, kill, fuck, shit, house, med, hate, bitch, love, care |
| 45 | 36 | 11% | started, song, guy, year, memory, disorder, screaming, repeat, neighbor, psychosis |
| 39 | 18 | 21% | hallucination, word, telephone, point, order, world, government, mean, situation, become |
| 28 | 21 | 14% | music, hear, spirit, ear, sound, woman, sister, name, bus, real |
| 18 | 11 | 28% | game, therapy, cigarette, use, nice, file, smoke, soul, computer, phone |

LDA puts most documents in one large generic topic (470 of 713 at k=6, 381 at
k=8) and distributes the rest thinly. **Do not read the coherence sweep as
choosing k** — across five seeds, variation in c_v between k values (sd 0.0029)
is less than half the variation between seeds at fixed k (0.0070), and the five
seeds pick k = 14, 20, 4, 8, 10.

---

## BERTopic (MiniLM embeddings + UMAP + HDBSCAN)

`min_topic_size=15`. 51% of documents are unassigned on the capped corpus, 58%
capped and filtered — HDBSCAN needs density, and capping removes it.

### Capped — 4 topics, none participant-dominated

| n | participants | top | words |
|---|---|---|---|
| 173 | 105 | 5% | voice, feel, hearing, day, time, trying, medication, today, hear, make |
| 119 | 56 | 6% | fucking, stuff, hear, time, voice, want, make, guy, trying, people |
| 41 | 33 | 10% | voice, hear, head, sound, hear voice, think, coming, hearing, talking |
| 17 | 17 | 6% | voice, year, hearing, hear, dna, recording, apartment, hearing voice |

### Capped and filtered — 7 topics, none participant-dominated

| n | participants | top | words |
|---|---|---|---|
| 114 | 56 | 7% | fucking, stuff, mean, want, guy, trying, time, people, make, fuck |
| 51 | 43 | 8% | loud, voice, feel, worse, make, head, little, hearing, try, hear |
| 32 | 24 | 12% | hear, voice, head, think, sound, hearing, coming, talking, thinking |
| 32 | 23 | 16% | hear, voice, hearing, hearing voice, schizophrenia, loud, day, negative |
| 25 | 12 | 28% | shadow, hat, spirit, entity, people, burning, weird, started, eye, bible |
| 22 | 10 | 23% | word, finding, trying, trigger, trigger word, state mind, mind, figure |
| 20 | 16 | 20% | feel, today, good, day, family, feeling, medication, coping skill, better |

This is the cleanest topic set in the project: hostile speech; loudness and
worsening; voice localisation; hearing and diagnosis; shadow and spirit imagery;
word-finding and state of mind; coping, family and treatment.

---

## Top2Vec (doc2vec embeddings, seeded UMAP)

Assigns every document — no outlier bucket.

### Capped — 3 topics

| n | participants | top | words |
|---|---|---|---|
| 342 | 193 | 2% | feel, day, hear, better, feeling, hard, someone, uneasy, focus, head |
| 189 | 98 | 4% | snake, satan, dragon, hat, knew, shadow, saw, looked, promise, year |
| 182 | 95 | 4% | fail, business, communication, money, win, river, portion, relationship |

### Capped and filtered — 7 topics

| n | participants | top | words |
|---|---|---|---|
| 192 | 117 | 4% | pounding, louder, irritating, sitting, stay, sleep, stressed, watching, loud, night |
| 109 | 67 | 6% | fucking, prescribed, discussed, fucked, getting, task, eat, waking, breathe, happy |
| 101 | 53 | 8% | aspect, measure, become, addition, behavioral, history, prepare, environment |
| 95 | 57 | 8% | hat, saw, dragon, snake, shadow, burn, bible, skin, mark, looked |
| 90 | 55 | 9% | sold, individual, skull, drug, broadcast, bigger, fbi, scan, believe, technology |
| 78 | 54 | 10% | distinguish, quality, continuous, external, instance, challenging, pass, aware |
| 48 | 31 | 15% | angel, guardian, guide, spiritual, visiting, universe, sharing, earth, religious |

Top2Vec is the only method that recovers a distinct **religious and spiritual**
theme (angels, guardians, guides) separately from the **threatening imagery**
theme (snakes, dragons, shadows, Satan), and the only one to separate a
**surveillance and technology** theme. It is also the most even: 7 topics
covering all 713 documents, none above 15% one participant.

---

## What the four methods agree on

Read across the capped runs, four themes recur in every method that has the
resolution to find them:

| theme | LDA | BERTopic | Top2Vec | sLDA (R) |
|---|---|---|---|---|
| Hearing and describing the voices — loudness, location, repetition | ✓ | ✓ | ✓ | ✓ |
| Hostile and profane speech | ✓ | ✓ | ✓ | ✓ |
| Religious, spiritual and threatening imagery | ✓ | ✓ | ✓ (split in two) | ✓ |
| Illness, treatment and coping | ✓ | ✓ | ✓ | ✓ |

Two further themes appear in the embedding methods only, both plausibly real and
both small: **word-finding and state of mind** (BERTopic), and **surveillance,
government and technology** (Top2Vec, LDA at k=8).

The theme that does *not* survive is the one earlier passes led with. Uncapped,
BERTopic, Top2Vec and sLDA all returned a large "treatment and illness" topic
with near-identical words, which read as strong cross-method convergence. It is
one participant — see below.

---

## Why the uncapped runs are not reported

| corpus | topics | unassigned | >50% one participant | median top share |
|---|---|---|---|---|
| uncapped | 13 | 32% | **7 of 13** | 53% |
| uncapped, filtered | 2 | 0% | 1 of 2 | 50% |
| capped | 4 | 51% | 0 of 4 | 6% |
| capped, filtered | 7 | 58% | 0 of 7 | 16% |

Uncapped, BERTopic's largest topic is **95% one participant** (n=372 of 1,352)
and Top2Vec's equivalent is **99%**. At the word level the terms carrying that
topic are 71–99% a single person, against *hear* at 4% across 226 participants
and *voice* at 20% across 274.

Removing participant-specific vocabulary does not fix it — a participant whose
terms each sit just under the threshold still dominates, and on the uncapped
corpus removing the tail collapses the clustering to a single topic. Capping
contribution is what works, and it is the reason every table above uses the
capped corpus.
