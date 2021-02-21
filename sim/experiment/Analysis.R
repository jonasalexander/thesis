require(dplyr)
require(glm)
require(ggplot2)
require(jtools)
require(effects)
require(survival)
require(lme4)

# DATA

mode <- "EXPERIMENT"
N <- 1000

raw_df <- read.csv("~/Desktop/thesis/sim/experiment/data/Adams_experiment_cleaned_all.csv")
months <- raw_df %>% group_by(word, s2_value) %>% summarise()

if (mode == "EXPERIMENT") {
  df <- raw_df %>%
    mutate(last = !as.logical(raw_df$did_continue_eval)) %>%
    mutate(subject.id = as.numeric(factor(raw_df$subject))) %>%
    select(last, subject.id, s2_value, order) %>%
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

# SIMPLE

## Want to use (1|subject.id) instead of full factor(subject.id) because
## the latter would treat subjects as having been randomly sampled which means that
## regressing to the mean makes sense, which is not the case for non-randomly
## sampled. Also something about the structure of the covariance matrix?

model.logvalue <- glmer(last ~ log(value) + (1|subject.id), data=df, family="binomial")
summary(model.logvalue)

# Has significant result, but is more robust to compare to baseline model that doesn't get value
# -> don't interpret p-values but do model comparison using anova
model.baseline <- glmer(last ~ (1|subject.id), data=df, family="binomial")
anova(model.logvalue, model.baseline)

# model.logvalue has variance 0 on the random effects, which is a problem
# try excluding subjects that stop after the first item:
one_entry <- (count(df.avg, subject.id) %>% filter(n == 1))$subject.id
df.exclude <- df[!df$subject.id %in% one_entry,]
summary(glmer(last ~ log(value) + (1|subject.id), data=df.exclude, family="binomial"))


# LOGIT

# Logit regression without individual-level analysis
# see https://www.rensvandeschoot.com/tutorials/discrete-time-survival/

df.hazard <- df %>%
  mutate(event = as.integer(df$last)) %>%
  group_by(subject.id) %>%
  mutate(time = order) %>%
  ungroup() %>%
  select(subject.id, value, event, time) 

df.hazard %>%
  group_by(value) %>%
  summarise(event = sum(event),
            total = n()) %>%
  mutate(hazard = event/total) %>%
  ggplot(aes(x = value, y = log(hazard/(1-hazard)))) +
  geom_point() +
  geom_smooth()

df.hazard %>%
  group_by(time) %>%
  summarise(event = sum(event),
            total = n()) %>%
  mutate(hazard = event/total) %>%
  ggplot(aes(x = time, y = log(hazard/(1-hazard)))) +
  geom_point() +
  geom_smooth()

model.time <- glm(formula = event ~ time,
                               family = binomial(link = "logit"),
                               data = df.hazard)
summary(model.time)

df.hazard %>%
  group_by(value) %>%
  summarise(event = sum(event),
            total = n()) %>%
  mutate(hazard = event/total) %>%
  ggplot(aes(x = value, y = log(hazard/(1-hazard)))) +
  geom_point() +
  geom_smooth()

model.value <- glm(formula = event ~ time + value,
                           family = binomial(link = "logit"),
                           data = df.hazard)

summary(model.value)
summ(model.value, exp = T)
plot_summs(model.value, exp = T)

# Logit Multi-level discrete-time survival analysis
# see https://www.rensvandeschoot.com/tutorials/discrete-time-survival/
model.multi <- glmer(formula = event ~ time + value + (1|subject.id),
                     family = binomial(logit),
                     data = df.hazard)
coef(summary(model.multi))


# SURVIVAL
df.hazard <- df %>%
  mutate(event = as.integer(df$last)) %>%
  group_by(subject.id) %>%
  mutate(time = order) %>%
  ungroup() %>%
  select(subject.id, value, event, time) 

model.coxph <- coxph(Surv(time,event)~ log(value), df.hazard, cluster=subject.id)
model.coxph
# This already includes time, can't have it as an additional variable on the right hand side
# coxph(Surv(time,event)~ log(value) + time, df.hazard, cluster=subject.id, iter.max=200)
zph <- cox.zph(model.coxph)
zph

plot(zph[1],lwd=2)
abline(0,0, col=1,lty=3,lwd=2)
abline(h= model.coxph$coef[1], col=3, lwd=2, lty=2) > legend("topleft",
                                                        legend=c('Reference line for null effect', "Average hazard over time", "Time-varying hazard"),
                                                        lty=c(3,2,1), col=c(1,3,1), lwd=2)

# COMPARING AVGs
df.avg <- df %>% group_by(subject.id, last) %>% summarise_each(funs(mean))

# Can exclude people with only one entry or make them have same value for both cases
one_entry <- (count(df.avg, subject.id) %>% filter(n == 1))$subject.id
df.avg.include <- rbind(df.avg, df.avg[df.avg$subject.id %in% one_entry,] %>% mutate(last = FALSE))
ggplot(df.avg.include, aes(value, fill=last)) + geom_density(alpha = 0.2)

df.avg.exclude <- df.avg[!df.avg$subject.id %in% one_entry,]
ggplot(df.avg.exclude, aes(value, fill=last)) + geom_density(alpha = 0.2)

value_increase_last <- df.avg.exclude$value[df.avg.exclude$last == TRUE] - df.avg.exclude$value[df.avg.exclude$last == FALSE]
hist(value_increase_last) # can we turn this into a test statistic?

