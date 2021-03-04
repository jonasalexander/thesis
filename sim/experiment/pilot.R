library(tidyr)
library(dplyr)

## ETL

MODE <- "bz"

raw_df <- read.csv("~/Dropbox/College/Thesis/Words Pilot/Words Memory Pilot_March 4, 2021_10.18.csv")
if (MODE == "ay") {
  words <- read.csv("~/Dropbox/College/Thesis/Words Pilot/ay.csv")$word
  experiment_versions <- c("v1", "pilot")
} else if (MODE == "bz") {
  words <- read.csv("~/Dropbox/College/Thesis/Words Pilot/bz.csv")$word
  experiment_versions <- c("v2")
}

df <- raw_df %>%
  subset(as.logical(Finished) == TRUE) %>%
  subset(Status == "IP Address") %>%
  subset(version %in% experiment_versions) %>%
  select(ResponseId, workerId, Duration..in.seconds., Q2, Q7_Page.Submit, Q22_1, Q22_2, Q22_3, Q22_4, Q22_5, Q22_6, Q22_7, Q22_8, Q22_9, Q22_10, Q22_11, Q22_12, Q22_13, Q35_Page.Submit, bonus) %>%
  rename(total_seconds=Duration..in.seconds., sentence=Q2, sentence_time=Q7_Page.Submit, recall_time=Q35_Page.Submit) %>%
  transform(bonus = as.numeric(bonus), recall_time = as.numeric(recall_time), sentence_time = as.numeric(sentence_time)) %>%
  mutate_if(is.numeric, round, digits=2) %>%
  subset(grepl("\\.", df$sentence) == TRUE)

df.long <- df %>%
  select(workerId, Q22_1, Q22_2, Q22_3, Q22_4, Q22_5, Q22_6, Q22_7, Q22_8, Q22_9, Q22_10, Q22_11, Q22_12, Q22_13) %>%
  pivot_longer(!workerId, names_to="entry", values_to="word") %>%
  mutate(word = toupper(word)) %>%
  subset(word %in% words)

word_counts <- as.data.frame(table(df.long$word)) %>% rename(word = Var1, count = Freq) %>% mutate(percentage = 100 * count / length(df$workerId))

## ANALYSIS

plot(word_counts$word, word_counts$percentage, ylim=c(0, 100))
word_counts

hist(df$bonus, xlim=c(0, 0.7))

