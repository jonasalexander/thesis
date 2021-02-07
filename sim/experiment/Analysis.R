require(dplyr)
require(dfidx)
require(glm)
require(ggplot2)

mode <- "EXPERIMENT"
N <- 1000

raw_df <- read.csv("~/Desktop/thesis/sim/experiment/data/Adams_experiment_cleaned_all.csv")
months <- raw_df %>% group_by(word, s2_value) %>% summarise()

if (mode == "EXPERIMENT") {
  df <- raw_df %>%
    mutate(last = !as.logical(raw_df$did_continue_eval)) %>%
    mutate(subject.id = as.numeric(factor(raw_df$subject))) %>%
    select(last, subject.id, s2_value) %>%
    rename(value = s2_value)
} else if (mode == "SIMULATE RANDOM") {
  mean_num_drawn <- length(df$subject.id)/length(unique(df$subject.id))
  # for each of N subjects: randomly draw one of the remaining words and then stop with probability 1/mean_num_drawn
  subject.id <- c(1:N)
  # Need to cap at 1, 12?
  # Need to multiply probability to get to similar expected values as in real life?
  num_drawn <- pmax(pmin(rgeom(x, 1/mean_num_drawn*0.75), rep(c(12), N)), 1)
  months_drawn <- sapply(num_drawn, function (x) sample(months$s2_value, x, replace = FALSE)) %>% unlist()
  df <- data.frame(rep(subject.id, num_drawn), months_drawn)
  colnames(df) <- c("subject.id", "value")
  last <- rep(c(F), length(df$subject.id))
  last[cumsum(num_drawn)] <- T
  df <- df %>% mutate(last = last)
} else if (mode == "SIMULATE INCREASING") {
  # for each of N subjects: randomly draw one of the remaining words and calculate p(v) where v is the value of the word, then stop with probability p(v)
  # raw_df <- 
  # everything the same as above, just set prob for the sampling function
  
}


summary(glm(last ~ log(value) + factor(subject.id), data=df, family="binomial"))


df.avg <- df %>% group_by(subject.id, last) %>% summarise_each(funs(mean))

# Can exclude people with only one entry or make them have same value for both cases
one_entry <- (count(df.avg, subject.id) %>% filter(n == 1))$subject.id
df.avg[df.avg$subject.id %in% one_entry,]
df.avg.include <- rbind(df.avg, df.avg[df.avg$subject.id %in% one_entry,] %>% mutate(last = FALSE))
ggplot(df.avg.include, aes(value, fill=last)) + geom_density(alpha = 0.2)

df.avg.exclude <- df.avg[!df.avg$subject.id %in% one_entry,]
ggplot(df.avg.exclude, aes(value, fill=last)) + geom_density(alpha = 0.2)


value_increase_last <- df.avg.exclude$value[df.avg.exclude$last == TRUE] - df.avg.exclude$value[df.avg.exclude$last == FALSE]



df.hazard <- df %>% mutate(event = as.integer(df$last)) %>% mutate(exit = ) %>% select(subject.id, value, event, exit) %>% 



# dflogit <- df %>% mlogit.data(choice = "last", shape = "long", id.var = "subject.id", alt.var = "option.id", chid.var = "subject.id")
# # dflogit <- df %>% dfidx(choice = "last", idx=c("subject.id", "option.id"))
# 
# m.last <- mlogit(last ~ s2_value | -1 + option.id, dflogit, panel = T,
#                   rpar = c(s2_value = "n"), halton = NA, R = 1000, tol = .001)
# 
# #m.choice.null = mlogit(chosen ~ s2_value | -1 + word_ind, df, panel = T,
# #                       rpar = c(s2_value = "n"), halton = NA, R = 1000, tol = .001)
