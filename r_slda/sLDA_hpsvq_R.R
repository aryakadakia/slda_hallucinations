# Set-up -----------------------------------------------

# Load packages
#install.packages("dplyr")
#install.packages("tidytext")
#install.packages("rsample")
#install.packages("tidyverse")
#install.packages("cld2")
#install.packages("lda")

library(dplyr)
library(rsample)
library(tidyverse)
library(cld2)
library(tidytext)
library(lda)

# Shared text cleaning and stop words. Resolved relative to this script so it
# works the same under Rscript and in RStudio.
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else getwd()
})
source(file.path(.script_dir, "slda_stopwords.R"))

# Load data
# ---- DATA LOCATION -------------------------------------------------
# >>> SWITCH THIS TO YOUR DATA DIRECTORY <<<
# Participant transcripts are not included in this repository.
data_dir <- Sys.getenv("SLDA_DATA_DIR", unset = NA)
if (is.na(data_dir) || !dir.exists(data_dir)) {
  stop("Set SLDA_DATA_DIR to the folder holding the training/test CSVs.")
}
# --------------------------------------------------------------------

df <- read.csv(file.path(data_dir, "baselinetext_training.csv"))
for_prediction <- read.csv(file.path(data_dir, "baselinetext_test.csv"))


OUTCOME <- "hpsvq.total.score"

# Set seed for reproducibility
set.seed(2023)


# Prepare dataset ------------------------------------

# Prepare the training documents. Repair, tokenise, drop stop words, collapse
# back to one string per diary -- all in prepare_documents() so train and test
# cannot drift apart, which is how the two halves ended up with different stop
# lists before.
train_prep <- prepare_documents(df, OUTCOME)

# Tokens that belong to one person rather than to the corpus. Computed on the
# TRAINING documents only, then applied to both sides. One participant supplies
# 35.6% of the training words here, and without this the top topic is reliably
# that person's vocabulary -- names, home town, a specific medication.
drop_tokens <- participant_specific_tokens_r(train_prep$tokens)
cat("dropping", length(drop_tokens), "participant-specific token types\n")

train_prep <- prepare_documents(df, OUTCOME, drop_tokens = drop_tokens)
df <- train_prep$docs
cat("training documents:", nrow(df), "from", length(unique(df$pid)), "participants\n")

# Perform sLDA ------------------------------------

# Create input that is in the format the model is expecting
train_input <- lexicalize(df$text)

slda_mod <- slda.em(documents = train_input$documents,
                    K = 8,#number of topics
                    vocab = train_input$vocab,
                    num.e.iterations = 100,
                    num.m.iterations = 2,
                    alpha = 1, 
                    eta = 0.1,
                    params = sample(c(-1,1), 8, replace = TRUE), #scalar multiple of value of k
                    variance = var(df$clinical),
                    annotations = df$clinical,
                    method = "sLDA")

# Extract top words for each of the topics
topics <- slda_mod$topics %>%
  top.topic.words(num.words = 5, by.score = TRUE) %>%
  apply(2, paste, collapse = ", ")

print(topics)

# Extract model coefficients for each topic
coefs <- data.frame(coef(summary(slda_mod$model)))
coefs <- cbind(coefs, topics = factor(topics,
                                      topics[order(coefs$Estimate, coefs$Std..Error)]))
coefs <- coefs[order(coefs$Estimate),]

# Write the coefficient table out. This is the whole reason for keeping an R arm
# alongside the Python pipeline: slda.em gives each topic an estimate with a
# standard error and a p-value, which tomotopy does not, and the paper's framing
# ("which themes are associated with severity") is an inferential claim that
# needs them. Previously they only ever reached a plot.
res_dir <- file.path(.script_dir, "results")
dir.create(res_dir, showWarnings = FALSE)
coef_out <- data.frame(outcome = OUTCOME,
                       topic_words = as.character(coefs$topics),
                       coefs[, setdiff(names(coefs), "topics")],
                       row.names = NULL)
write.csv(coef_out, file.path(res_dir, paste0(OUTCOME, "_topic_coefficients.csv")),
          row.names = FALSE)
cat("\n---- topic coefficients:", OUTCOME, "----\n")
print(coef_out, digits = 3)

# Visualize top words per topic
fig_dir <- file.path(.script_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE)
p_coefs <- coefs %>% ggplot(aes(topics, Estimate, colour = Estimate)) + 
  geom_point() + 
  geom_errorbar(width = 0.5, 
                aes(ymin = Estimate - 1.96 * Std..Error, 
                    ymax = Estimate + 1.96 * Std..Error)) +
  coord_flip() + theme_bw()
ggsave(file.path(fig_dir, paste0(OUTCOME, "_coefficients.png")),
       p_coefs, width = 9, height = 5, dpi = 150)


# Testing whether topics are associated with the outcome ----------------

# Topic proportions per document, one column per topic.
df2 <- t(slda_mod$document_sums) / colSums(slda_mod$document_sums)
df2 <- cbind(df2, df$clinical)
df2 <- data.frame(df2)
colnames(df2) <- c(paste0("topic", 1:8), "clinical")

# The no-intercept fit, kept because it is what slda.em reports internally and
# what the plot above draws. Each coefficient is the fitted outcome for a
# document made entirely of that topic, so the estimates are readable as a
# severity gradient -- but every p-value tests the estimate against ZERO, not
# against the other topics. For a score bounded well above zero they are all
# significant by construction and say nothing about association.
lmod_nointercept <- lm(clinical ~ . - 1, data = df2)

# The inferential version. Topic proportions sum to 1, so with an intercept one
# topic drops out and becomes the reference; every remaining coefficient is then
# the difference in outcome between a document wholly in that topic and one
# wholly in the reference, which is the contrast the question actually asks.
# The reference is the lowest-scoring topic, so the estimates read as increments
# above the mildest theme.
ref <- names(sort(coef(lmod_nointercept)))[1]
others <- setdiff(paste0("topic", 1:8), ref)
form <- as.formula(paste("clinical ~", paste(others, collapse = " + ")))
lmod <- lm(form, data = df2)

# Omnibus test first: does topic composition explain any variance at all?
null_mod <- lm(clinical ~ 1, data = df2)
omni <- anova(null_mod, lmod)
cat("\n---- do topics explain the outcome?", OUTCOME, "----\n")
cat(sprintf("  reference topic       : %s (%s)\n", ref,
            topics[as.integer(sub("topic", "", ref))]))
cat(sprintf("  F(%d, %d) = %.2f, p = %.3g\n",
            omni$Df[2], omni$Res.Df[2], omni$F[2], omni$`Pr(>F)`[2]))
cat(sprintf("  in-sample adjusted R^2: %.3f\n", summary(lmod)$adj.r.squared))

contrasts_tbl <- data.frame(outcome = OUTCOME,
                            reference = ref,
                            coef(summary(lmod)),
                            row.names = NULL)
contrasts_tbl$term <- rownames(coef(summary(lmod)))
contrasts_tbl$topic_words <- c(NA, topics[as.integer(sub("topic", "",
                                 rownames(coef(summary(lmod)))[-1]))])
write.csv(contrasts_tbl,
          file.path(res_dir, paste0(OUTCOME, "_topic_contrasts.csv")),
          row.names = FALSE)
cat("\n  differences from the reference topic (this is the testable claim):\n")
print(contrasts_tbl[, c("term", "Estimate", "Std..Error", "Pr...t..", "topic_words")],
      digits = 3)

# Prediction on unseen data ---------------------------
df <- for_prediction

# Same preparation, same drop set, from the training side.
df <- prepare_documents(df, OUTCOME, drop_tokens = drop_tokens)$docs
cat("test documents:", nrow(df), "from", length(unique(df$pid)), "participants\n")

# Index the test documents against the TRAINING vocabulary. lexicalize() with
# no vocab argument builds a fresh index in first-seen order, so the word ids it
# returns address different columns of slda_mod$topics than the ones the model
# was fitted on: the predictions were being read off unrelated word-topic
# associations. Passing vocab= reuses the training index and drops
# out-of-vocabulary words, which is what slda.predict expects.
test_docs <- lexicalize(df$text, vocab = train_input$vocab)

# Restricting to the training vocabulary leaves a handful of test documents with
# no in-vocabulary words at all. slda.predict returns NaN for those, which then
# poisons every summary statistic computed over the vector. Drop them and keep
# the outcome aligned, reporting how many went.
n_empty <- sum(vapply(test_docs, ncol, integer(1)) == 0)
if (n_empty > 0) {
  keep <- vapply(test_docs, ncol, integer(1)) > 0
  cat("dropping", n_empty, "test documents with no in-vocabulary words\n")
  test_docs <- test_docs[keep]
  df <- df[keep, ]
}

# Do prediction using first model (no explanatory variables aside from topics)
yhat <- slda.predict(test_docs,
                     slda_mod$topics,
                     slda_mod$model,
                     alpha = 1,
                     eta = 0.1)

# Print full coefficient values
#cat(coef(summary(slda_mod$model)), "\n")
#print(yhat)

p_dens <- ggplot(data.frame(yhat = yhat, actual = df$clinical)) +
  geom_density(aes(yhat), fill = 1, alpha = I(0.5)) +
  geom_density(aes(actual), fill = 2, alpha = I(0.5)) +
  labs(x = "Predicted vs actual", y = "Density") +
  theme_bw() + theme(legend.position = "none")
ggsave(file.path(fig_dir, paste0(OUTCOME, "_predictions.png")),
       p_dens, width = 7, height = 4, dpi = 150)

#### Held-out performance -------------------------------------------------
# The original line here was
#
#     R_squared_variance <- var(yhat) / var(df$clinical)
#
# which is not an R-squared. It is the ratio of the spread of the predictions to
# the spread of the outcome and never looks at whether a prediction lands near
# its own observation, so it is unchanged if yhat is randomly permuted and can
# exceed 1. The four numbers previously reported from these scripts (0.181,
# 0.092, 0.086, 0.062) were that ratio. Reported below is the out-of-sample
# R-squared, 1 - SSE/SST, alongside Pearson r; the old ratio is kept, labelled
# for what it is, so the earlier figures remain traceable.
y <- df$clinical
stopifnot(length(y) == length(yhat))

sse <- sum((y - yhat)^2)
sst <- sum((y - mean(y))^2)
r2_oos <- 1 - sse / sst
pearson <- suppressWarnings(cor(y, yhat))
var_ratio <- var(yhat) / var(y)

cat("\n---- held-out performance:", OUTCOME, "----\n")
cat(sprintf("  n test documents      : %d\n", length(y)))
cat(sprintf("  Pearson r             : %+.3f\n", pearson))
cat(sprintf("  out-of-sample R^2     : %+.3f\n", r2_oos))
cat(sprintf("  RMSE                  : %.3f  (sd of y = %.3f)\n",
            sqrt(mean((y - yhat)^2)), sd(y)))
cat(sprintf("  var(yhat)/var(y)      : %.3f   <- the old \"R-squared\", not one\n",
            var_ratio))

# Binary outcomes additionally get an AUC, computed from the rank statistic so
# no extra package is needed, for comparability with the Python table.
if (length(unique(y)) == 2) {
  pos <- yhat[y == max(y)]; neg <- yhat[y == min(y)]
  auc <- (mean(rank(c(pos, neg))[seq_along(pos)]) -
            (length(pos) + 1) / 2) / length(neg)
  cat(sprintf("  AUC                   : %.3f  (majority class = %.3f)\n",
              auc, max(mean(y == max(y)), mean(y == min(y)))))
}
cat("--------------------------------------------\n")