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

# Load data
df <- read.csv("C:/Users/arya1999/Desktop/sLDA AVH/datasets_final/baselinetext_training.csv")
for_prediction <- read.csv("C:/Users/arya1999/Desktop/sLDA AVH/datasets_final/baselinetext_test.csv")


# Set seed for reproducibility
set.seed(2022)


# Prepare dataset ------------------------------------

# Keep only the variables that will be used for modelling
df <- df %>% 
  select(used.treatment, text_manual)

# Rename variables
df <- df %>%
  rename(clinical = used.treatment,
         text = text_manual)

# Assuming custom_stop_words is a vector of your custom stop words
custom_stop_words <- c("i", "yeah", "um", "uh", "like", "you")

# Combine custom stop words with the existing stop_words dataset
combined_stop_words <- bind_rows(stop_words, tibble(word = custom_stop_words))

# Convert to lowercase, remove punctuation and special characters, remove stop words
df <- df %>%
  mutate(diary_num = row_number()) %>%
  unnest_tokens(output = word, input = text) %>%
  anti_join(combined_stop_words) %>%
  group_by(diary_num) %>% 
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>% 
  select(-word) %>% 
  distinct()

# Fix review number variable
df <- df %>% 
  select(-diary_num) %>% 
  mutate(diary_num = row_number())

# Perform sLDA ------------------------------------

# Create input that is in the format the model is expecting
sLDA_input <- lexicalize(df$text)

slda_mod <- slda.em(documents = sLDA_input$documents,
                    K = 8,#number of topics
                    vocab = sLDA_input$vocab,
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

view(topics)

# Extract model coefficients for each topic
coefs <- data.frame(coef(summary(slda_mod$model)))
coefs <- cbind(coefs, topics = factor(topics,
                                      topics[order(coefs$Estimate, coefs$Std..Error)]))
coefs <- coefs[order(coefs$Estimate),]

# Visualize top words per topic
coefs %>% ggplot(aes(topics, Estimate, colour = Estimate)) + 
  geom_point() + 
  geom_errorbar(width = 0.5, 
                aes(ymin = Estimate - 1.96 * Std..Error, 
                    ymax = Estimate + 1.96 * Std..Error)) +
  coord_flip() + theme_bw()


# Add an explanatory variable to the model ----------------

# Generate model coefficients without using slda.em function
df2 <- t(slda_mod$document_sums) / colSums(slda_mod$document_sums)
df2 <- cbind(df2, df$clinical)
df2 <- data.frame(df2)
colnames(df2) <- c(paste0("topic", 1:8), "clinical")
lmod <- lm(clinical ~ . -1, data = df2)
coef(lmod)

# Prediction on unseen data ---------------------------
df <- for_prediction

# Keep only the variables that will be used for modelling
df <- df %>% 
  select(used.treatment, text_manual)

# Rename variables
df <- df %>%
  rename(clinical = used.treatment,
         text = text_manual)

# Assuming custom_stop_words is a vector of your custom stop words
custom_stop_words <- c("i", "um", "uh", "like", "you")

# Combine custom stop words with the existing stop_words dataset
combined_stop_words <- bind_rows(stop_words, tibble(word = custom_stop_words))

# Convert to lowercase, remove punctuation and special characters, remove stop words
df <- df %>%
  mutate(diary_num = row_number()) %>%
  unnest_tokens(output = word, input = text) %>%
  anti_join(combined_stop_words) %>%
  group_by(diary_num) %>% 
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>% 
  select(-word) %>% 
  distinct()

# Fix review number variable
df <- df %>% 
  select(-diary_num) %>% 
  mutate(diary_num = row_number())

# Create input that is in the format the model is expecting
sLDA_input <- lexicalize(df$text)

# Do prediction using first model (no explanatory variables aside from topics)
yhat <- slda.predict(sLDA_input$documents,
                     slda_mod$topics,
                     slda_mod$model,
                     alpha = 1,
                     eta = 0.1)

# Print full coefficient values
#cat(coef(summary(slda_mod$model)), "\n")
#print(yhat)

qplot(yhat,
      xlab = "Predicted Overall Rating",
      ylab = "Density", fill = 1,
      alpha=I(0.5), geom = "density") +
  theme_bw() +
  theme(legend.position = "none") +
  geom_density(aes(df$clinical), fill = 2, alpha = I(0.5))

#### R-squared
variance_yhat <- var(yhat)
variance_y <- var(df$clinical)
R_squared_variance <- variance_yhat / variance_y
print(paste0("R-squared (variance):", R_squared_variance))