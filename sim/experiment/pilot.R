library(tidyr)
library(dplyr)

## ETL
raw_df <- read.csv("~/Dropbox/College/Thesis/Words Pilot/Words Memory Pilot_March 4, 2021_10.18.csv")

transform_raw <- function(experiment_versions, words) {
  df <- raw_df %>%
    subset(as.logical(Finished) == TRUE) %>%
    subset(Status == "IP Address") %>%
    subset(version %in% experiment_versions) %>%
    select(ResponseId, workerId, Duration..in.seconds., Q2, Q7_Page.Submit, Q22_1, Q22_2, Q22_3, Q22_4, Q22_5, Q22_6, Q22_7, Q22_8, Q22_9, Q22_10, Q22_11, Q22_12, Q22_13, Q35_Page.Submit, bonus) %>%
    rename(total_seconds=Duration..in.seconds., sentence=Q2, sentence_time=Q7_Page.Submit, recall_time=Q35_Page.Submit) %>%
    transform(bonus = as.numeric(bonus), recall_time = as.numeric(recall_time), sentence_time = as.numeric(sentence_time)) %>%
    mutate_if(is.numeric, round, digits=2)
  
  df.long <- df %>%
    subset(grepl("\\.", df$sentence) == TRUE) %>%
    select(workerId, Q22_1, Q22_2, Q22_3, Q22_4, Q22_5, Q22_6, Q22_7, Q22_8, Q22_9, Q22_10, Q22_11, Q22_12, Q22_13) %>%
    pivot_longer(!workerId, names_to="entry", values_to="word") %>%
    mutate(word = toupper(word)) %>%
    subset(word %in% words)
  
  return(df.long %>%
           mutate(rank = match(df.long$word, words)))
}

ay.words <- read.csv("~/Dropbox/College/Thesis/Words Pilot/ay.csv")$word
ay.experiment_versions <- c("v1", "pilot")
ay.long <- transform_raw(ay.experiment_versions, ay.words)
ay.word_counts <- as.data.frame(table(ay.long$word)) %>% rename(word = Var1, count = Freq) %>% mutate(percentage = 100 * count / length(unique(ay.long$workerId)))

bz.words <- rev(read.csv("~/Dropbox/College/Thesis/Words Pilot/bz.csv")$word) # reverse because closer to A is better here
bz.experiment_versions <- c("v2")
bz.long <- transform_raw(bz.experiment_versions, bz.words)
bz.word_counts <- as.data.frame(table(bz.long$word)) %>% rename(word = Var1, count = Freq) %>% mutate(percentage = 100 * count / length(unique(bz.long$workerId)))

combined_word_counts <- rbind(ay.long, bz.long)
word_counts <- as.data.frame(table(combined_word_counts$rank)) %>% rename(rank = Var1, count = Freq) %>% mutate(percentage = 100 * count / length(unique(combined_word_counts$workerId)))

## ANALYSIS

plot(word_counts$rank, word_counts$percentage, ylim=c(0, 100), xlab="rank", ylab="recall frequency")
word_counts
