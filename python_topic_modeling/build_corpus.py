"""Build the modelling corpus from the processed transcripts.

Two problems in the original setup are addressed here.

1. Corpus concentration. One participant contributed 386 of 3,090 diaries and
   26% of all words. Because LDA weights topic estimation by token count, the
   topics were substantially that person's vocabulary. Splitting long documents
   into fixed-size chunks equalises the contribution and, as a side effect,
   turns 227 training documents into several thousand, which matters because
   sLDA over a 10k vocabulary was badly under-determined at n=227.

2. Encoding damage in contractions (see clean_text.py).

Topics are learned over chunks; each participant's chunk-level topic
proportions are then averaged so the regression stays at participant level and
the outcome is never split across the train/test boundary.
"""
import re, sys, numpy as np, pandas as pd, nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
sys.path.insert(0, str(Path(__file__).parent))
from clean_text import repair

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()
VALID_POS = {"VB","VBD","VBG","VBN","VBP","VBZ","JJ","JJR","JJS","NN","NNS","NNP","NNPS"}

# Terms the original project added by hand. The encoding artefacts ("itbs",
# "donbt", "iãåâ") are dropped because clean_text.py now repairs them at source.
# Spoken diary transcripts are dominated by disfluencies and contraction
# fragments. Every topic model in this project previously failed the same way:
# LDA returned three topics that were all "im / know / dont / like / voic",
# BERTopic named its clusters "um know like uh" and dumped 61% of documents into
# the outlier bucket. NLTK's English stop list does not contain um, uh, ve, im,
# dont, or the stemmed forms, so they survived and dominated every model.
DISFLUENCY = {"um","uh","er","erm","hmm","mm","mhm","yeah","yep","nah","oh","ah",
              "okay","ok","alright","gonna","gotta","wanna","kinda","sorta"}

CONTRACTION_FRAGMENTS = {"im","ive","id","ill","dont","doesnt","didnt","cant","couldnt",
                         "wont","wouldnt","shouldnt","isnt","arent","wasnt","werent",
                         "thats","theres","theyre","youre","youve","weve","hes","shes",
                         "its","aint","ve","ll","re","don","didn","doesn","couldn","wouldn"}

FILLER = {"like","just","know","really","thing","things","kind","sort","lot","bit",
          "way","get","got","go","going","went","say","said","tell","told","come","came",
          "would","could","should","one","also","even","still","much","many","well"}

# Artefacts of lemmatising and of NLTK's tokeniser, which splits contractions:
#   was -> wa,  has -> ha,  gonna -> gon + na,  wanna -> wan + na
LEMMA_ARTEFACTS = {"wa","ha","doe","thi","gon","na","wan","ta","gotta","lemme"}

# An older corruption in the source CSVs replaced the apostrophe with a literal
# ASCII "u", so it's -> itus, I'm -> ium, don't -> donut. clean_text.py cannot
# repair these because nothing distinguishes them from ordinary words at the
# byte level, and "donut" is a real word, so repair would sometimes invent one.
# Three of them were previously handled by hand in the notebooks and never made
# it into this module, so ium (55), itus (48) and donut (45) were still reaching
# the topic models - "itus" and "thatus" turned up in a Top2Vec topic. The
# damage is regular, so generate the whole family rather than naming members:
# a contraction stem, a "u" where the apostrophe was, and the usual suffix.
_STEMS = ("it","that","i","don","doesn","didn","can","won","wouldn","couldn",
          "isn","aren","wasn","weren","there","they","you","we","he","she",
          "let","who","what","here","ain","shouldn","hasn","haven","hadn")
CORRUPTED_CONTRACTIONS = {st + "u" + sf for st in _STEMS
                          for sf in ("s","m","t","re","ve","ll","d")}

CUSTOM_STOP = (DISFLUENCY | CONTRACTION_FRAGMENTS | FILLER | LEMMA_ARTEFACTS
               | CORRUPTED_CONTRACTIONS)

def tokenise(text):
    text = repair(str(text))
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    lemmas = [LEM.lemmatize(w) for w in nltk.word_tokenize(text)]
    return [w for w, pos in nltk.pos_tag(lemmas)
            if pos in VALID_POS and w not in STOP and w not in CUSTOM_STOP and len(w) > 2]

def chunk(tokens, size=250):
    """Split a token list into chunks of `size`; merge a short tail into the
    previous chunk so we never emit a stub."""
    if len(tokens) <= size:
        return [tokens] if tokens else []
    out = [tokens[i:i+size] for i in range(0, len(tokens), size)]
    if len(out) > 1 and len(out[-1]) < size // 2:
        out[-2].extend(out.pop())
    return out

def participant_specific_tokens(chunks, min_participants=3, max_share=0.9):
    """Tokens that essentially belong to one person.

    Topic models on this corpus surface personal names, home towns and specific
    medications, and almost all of them trace to a single participant. Ten such
    terms are 100% one person, one first name 99%, a city 90%, two more 86%.
    Several BERTopic and Top2Vec topics are, in effect, one participant's
    vocabulary wearing a topic label. (The terms themselves are deliberately not
    listed here - this file is published and they are not.)

    That is a disclosure problem rather than only a quality one. A topic list
    naming a city, a first name and a specific medication, in a paper about an
    identifiable clinical cohort, is quasi-identifying: the combination narrows
    the field sharply for anyone who knows the person.

    Returns the tokens appearing in fewer than `min_participants` participants,
    or where one participant accounts for more than `max_share` of occurrences.
    """
    from collections import defaultdict, Counter
    per = defaultdict(Counter)
    for pid, toks in zip(chunks['pid'], chunks['tokens']):
        for w in set(toks):
            per[w][pid] += 1
    out = set()
    for w, c in per.items():
        tot = sum(c.values())
        if len(c) < min_participants or c.most_common(1)[0][1] / tot > max_share:
            out.add(w)
    return out


def build(df, id_col, text_col, chunk_size=150, min_tokens=20,
          max_chunks_per_pid=8, seed=42, drop_participant_specific=False):
    """Return a long dataframe: one row per chunk, carrying its participant id.

    max_chunks_per_pid caps how much any one participant can contribute. Chunking
    alone does not fix corpus concentration, because a prolific participant simply
    produces proportionally more chunks: in this data one person still accounted
    for 29% of tokens after chunking. Capping the number of chunks per participant
    is what actually equalises contribution. Chunks are sampled rather than taken
    from the start, so the retained text is not biased toward early recordings.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid, txt in zip(df[id_col], df[text_col]):
        toks = tokenise(txt)
        if len(toks) < min_tokens:
            continue
        chs = chunk(toks, chunk_size)
        if max_chunks_per_pid and len(chs) > max_chunks_per_pid:
            idx = sorted(rng.choice(len(chs), max_chunks_per_pid, replace=False))
            chs = [chs[i] for i in idx]
        for j, ch in enumerate(chs):
            rows.append({'pid': pid, 'chunk_ix': j, 'tokens': ch, 'n_tokens': len(ch)})
    out = pd.DataFrame(rows)

    # Off by default: turning it on changes every reported number, so it is a
    # decision to take deliberately rather than a silent default. Turn it on
    # before publishing topic word lists - see participant_specific_tokens.
    if drop_participant_specific and len(out):
        drop = participant_specific_tokens(out)
        out['tokens'] = [[w for w in t if w not in drop] for t in out['tokens']]
        out['n_tokens'] = out['tokens'].str.len()
        out = out[out.n_tokens >= min_tokens // 2].reset_index(drop=True)
    return out
