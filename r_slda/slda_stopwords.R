# Shared text cleaning and stop words for the R sLDA scripts.
# ---------------------------------------------------------------------------
# This is the R port of python_topic_modeling/clean_text.py and the CUSTOM_STOP list in
# python_topic_modeling/build_corpus.py. Keep the two in step: if a term is added on one
# side it belongs on the other, otherwise the R and Python results stop being
# comparable.
#
# Why this file exists
# --------------------
# Each script previously defined its own six-word list:
#
#     custom_stop_words <- c("i", "yeah", "um", "uh", "like", "you")
#
# tidytext's `stop_words` covers "like" and "you" already, and it does contain
# the straight-apostrophe contractions "i'm", "don't", "it's". What it cannot
# match is the *curly* apostrophe. 390 of the 2,184 training transcripts carry
# U+2019 instead of U+0027, so the tokeniser emits "i<U+2019>m", the anti_join
# misses it, and the term survives into the topics. In the unmodified pipeline
# the surviving vocabulary ranked, in the top 40:
#
#     i<U+2019>m 1022 | it<U+2019>s 517 | gonna 457 | i<U+2019>ve 381 | don<U+2019>t 378 | lot 1133 | basically 475
#
# which is what produced the topic label reading "i'm, it's, i've, don't,
# that's". Normalising the punctuation is therefore load-bearing, not cosmetic:
# without it the extra stop words below would still miss the damaged forms.

# Encoding repair -----------------------------------------------------------

# `Ã¢€œ` and friends are UTF-8 that was decoded as cp1252 and re-encoded, in
# some rows twice over (139 rows affected; one round trip leaves 7, two leaves
# none). Undo it by reversing that round trip until the markers are gone.
undo_mojibake <- function(x, max_passes = 3) {
  for (i in seq_len(max_passes)) {
    if (!any(grepl("\u00c3|\u00e2\u20ac|\u00c2|\u00c5", x, useBytes = FALSE))) break
    y <- iconv(x, "UTF-8", "WINDOWS-1252", sub = "")
    y[is.na(y)] <- x[is.na(y)]
    Encoding(y) <- "UTF-8"
    y <- iconv(y, "UTF-8", "UTF-8", sub = "")
    y[is.na(y)] <- x[is.na(y)]
    x <- y
  }
  x
}

# Anything non-ASCII still standing after the round trip is residue rather than
# language, and much of it is a mangled contraction. Drop the whole word, as
# clean_text.py does: stripping the offending character in place turns
# "don<junk>t" into "donut" and "it<junk>s" into "itus", real-looking words that
# then enter the vocabulary. Deleting the token loses one word; repairing it
# invents one.
drop_nonascii_words <- function(x) {
  vapply(strsplit(x, "[[:space:]]+"), function(w) {
    paste(w[!grepl("[^\u0001-\u007f]", w)], collapse = " ")
  }, character(1))
}

repair_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- undo_mojibake(x)
  # curly punctuation -> ASCII, so tidytext's stop_words can match the
  # contractions it already contains
  x <- gsub("\u2019|\u2018|\u02bc|\u00b4|`", "'", x)
  x <- gsub("\u201c|\u201d", '"', x)
  x <- gsub("\u2014|\u2013", " - ", x)
  x <- gsub("\u2026", "...", x)
  x <- drop_nonascii_words(x)
  trimws(gsub("[[:space:]]+", " ", x))
}

# Stop words ----------------------------------------------------------------
# Four groups, 95 terms, mirroring build_corpus.py. Spoken diary transcripts
# are dominated by disfluency and by the wreckage of contractions; NLTK's list
# has none of um, uh, ve, im, dont, and tidytext's has no um, uh, yeah or
# gonna. Leaving them in is what produced three LDA topics that were all
# "im / know / dont / like / voic" and BERTopic clusters named "um know like uh".

disfluency <- c(
  "ah", "alright", "er", "erm", "gonna", "gotta", "hmm", "kinda", "mhm",
  "mm", "nah", "oh", "ok", "okay", "sorta", "uh", "um", "wanna", "yeah",
  "yep")

contraction_fragments <- c(
  "aint", "arent", "cant", "couldn", "couldnt", "didn", "didnt", "doesn",
  "doesnt", "don", "dont", "hes", "id", "ill", "im", "isnt", "its", "ive",
  "ll", "re", "shes", "shouldnt", "thats", "theres", "theyre", "ve",
  "wasnt", "werent", "weve", "wont", "wouldn", "wouldnt", "youre", "youve")

filler <- c(
  "also", "bit", "came", "come", "could", "even", "get", "go", "going",
  "got", "just", "kind", "know", "like", "lot", "many", "much", "one",
  "really", "said", "say", "should", "sort", "still", "tell", "thing",
  "things", "told", "way", "well", "went", "would")

# Artefacts of the Python lemmatiser and of NLTK's tokeniser (was -> wa,
# gonna -> gon + na). tidytext does neither, so these cannot arise here; they
# are carried across only to keep the two lists identical, and are inert.
lemma_artefacts <- c(
  "doe", "gon", "gotta", "ha", "lemme", "na", "ta", "thi", "wa", "wan")

# The apostrophe forms. build_corpus.py never needs these because it strips
# punctuation before tokenising; tidytext keeps the apostrophe, so "don't"
# arrives intact and has to be matched as written.
apostrophe_forms <- c(
  "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't",
  "he's", "i'd", "i'll", "i'm", "i've", "isn't", "it's", "she's",
  "shouldn't", "that's", "there's", "they're", "wasn't", "we've",
  "weren't", "won't", "wouldn't", "you're", "you've")

# An older corruption in the source CSVs put a literal ASCII "u" where the
# apostrophe was: it's -> itus, I'm -> ium, don't -> donut, they're -> theyure.
# repair_text() cannot undo it, because at the byte level these look like
# ordinary words. 77 occurrences across 17 forms survive in
# baselinetext_training.csv. The damage is regular, so generate the family
# rather than naming members. Two collide with real words - "donut", and "iud"
# the device - and are dropped anyway: in a hearing-voices diary "don't" and
# "I'd" are the overwhelmingly likelier readings, and deleting a token loses one
# word where repairing it would invent one.
corrupted_contractions <- as.vector(outer(
  c("it","that","i","don","doesn","didn","can","won","wouldn","couldn",
    "isn","aren","wasn","weren","there","they","you","we","he","she",
    "let","who","what","here","ain","shouldn","hasn","haven","hadn"),
  c("s","m","t","re","ve","ll","d"),
  function(stem, suffix) paste0(stem, "u", suffix)))

custom_stop_words <- unique(c(disfluency, contraction_fragments, filler,
                              lemma_artefacts, apostrophe_forms,
                              corrupted_contractions))

# Original per-script additions, kept so nothing is silently lost.
custom_stop_words <- unique(c(custom_stop_words, "i", "you"))

build_stop_words <- function() {
  utils::data("stop_words", package = "tidytext", envir = environment())
  dplyr::bind_rows(
    get("stop_words", envir = environment()),
    tibble::tibble(word = custom_stop_words, lexicon = "slda_custom")
  )
}

# Corpus concentration ------------------------------------------------------

# One participant contributed 386 of the training diaries and 35.6% of all
# training words; the top five hold 48.5%. Topic models weight estimation by
# token count, so a topic can be one person's vocabulary wearing a topic label.
# In the Python arm the largest BERTopic topic turned out to be 95% one
# participant, and the words carrying it (a first name, a home town, a specific
# medication) were 90-100% that person.
#
# This returns the tokens to drop: those appearing in fewer than
# `min_participants` participants, or where one participant supplies more than
# `max_share` of the occurrences. Mirrors build_corpus.participant_specific_tokens,
# except that it counts per diary entry rather than per chunk.
#
# Compute it on the TRAINING documents only and apply the same set to test,
# otherwise the test fold influences the vocabulary.
participant_specific_tokens_r <- function(tokens, min_participants = 3,
                                          max_share = 0.9) {
  s <- tokens %>%
    dplyr::count(word, pid, name = "n") %>%
    dplyr::group_by(word) %>%
    dplyr::summarise(n_p = dplyr::n_distinct(pid),
                     share = max(n) / sum(n), .groups = "drop")
  s$word[s$n_p < min_participants | s$share > max_share]
}

# Document preparation ------------------------------------------------------

# One function for both train and test. These steps used to be written out twice
# per script with small differences between the copies, which is how the two
# halves ended up with different stop lists in the first place.
prepare_documents <- function(raw, outcome_col, text_col = "text_manual",
                              file_col = "file", drop_tokens = character(0)) {
  stop_df <- build_stop_words()
  d <- data.frame(clinical = raw[[outcome_col]],
                  text     = raw[[text_col]],
                  pid      = sub("@.*$", "", raw[[file_col]]),
                  stringsAsFactors = FALSE)
  d <- d[!is.na(d$clinical) & !is.na(d$text), ]
  d$diary_num <- seq_len(nrow(d))
  d$text <- repair_text(d$text)

  tk <- d %>%
    tidytext::unnest_tokens(output = word, input = text) %>%
    dplyr::anti_join(stop_df, by = "word")
  if (length(drop_tokens)) {
    tk <- dplyr::filter(tk, !word %in% drop_tokens)
  }

  # A diary whose tokens were all stop words leaves nothing; lexicalize() emits
  # a zero-length document for it and slda.em cannot use one.
  docs <- tk %>%
    dplyr::group_by(diary_num) %>%
    dplyr::summarise(clinical = dplyr::first(clinical),
                     pid      = dplyr::first(pid),
                     text     = paste(word, collapse = " "),
                     .groups  = "drop") %>%
    dplyr::filter(nchar(trimws(text)) > 0)

  list(tokens = tk, docs = docs)
}
