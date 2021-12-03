require(dplyr)
require(survival)
require(survminer)

# DATA
raw_df <- read.csv("~/Desktop/thesis/data/Adams_experiment_cleaned_filtered.csv")
months <- raw_df %>% group_by(word, s2_value) %>% summarise()

df <- raw_df %>%
  mutate(last = !as.logical(raw_df$did_continue_eval)) %>%
  mutate(subject.id = as.numeric(factor(raw_df$subject))) %>%
  select(last, subject.id, s2_value, order) %>%
  rename(value = s2_value)

plot(aggregate(last ~ order, df, mean))

# TODO: Get simulated data created in Python
filename <- "~/Desktop/thesis/data/thesis_full_clean_long.csv"
df <- read.csv(filename) %>% mutate(last = as.logical(last))

plot(aggregate(last ~ order, df, mean))
plot(aggregate(last ~ value, df, mean),
     ylim=c(0,0.4),
     ylab="Probability of Exiting Decision Process",
     xlab="Value of Most Recent Word Considered",
     main="Effect of Value on Option Generation") + abline(lm(last ~ value, df), col="red")
plot(aggregate(last ~ value, df, mean),
     ylim=c(0,0.4),
     ylab="Probability of Exiting Decision Process",
     xlab="Value of Most Recent Word Considered",
     main="Effect of Value on Option Generation")

# COX
df.hazard <- df %>%
  mutate(event = as.integer(df$last)) %>%
  group_by(subject.id) %>%
  mutate(time = order) %>%
  ungroup() %>%
  select(subject.id, value, event, time) %>%
  mutate(value.centered = value - mean(df$value))

model.coxph <- coxph(Surv(time,event) ~ log(value), data = df.hazard)
summary(model.coxph)
ggsurvplot(survfit(model.coxph, data=df.hazard),
           ggtheme = theme_minimal())

# LOGIT

df %>%
  group_by(value) %>%
  summarise(event = sum(last),
            total = n()) %>%
  mutate(hazard = event/total) %>%
  ggplot(aes(x = value, y = log(-log(1-hazard)))) +
  geom_point() +
  geom_smooth()

df %>%
  group_by(order) %>%
  summarise(event = sum(last),
            total = n()) %>%
  mutate(hazard = event/total) %>%
  ggplot(aes(x = order, y = log(-log(1-hazard)))) +
  geom_point() +
  geom_smooth()

model.glm <- glm(last ~ order + log(value),
                family = binomial(link = "logit"),
                data = df)

summary(model.glm)

summary(glm(last ~ log(value), data=df, family="binomial"))


# EXTRA: Examining Cox Assumption
zph <- cox.zph(model.coxph)
zph

# Unfortunately does not look like the ratio between these stays constant
# time coefficient is messing with us
grouped <- aggregate(event ~ time + value, df.hazard, mean)
grouped %>% ggplot(aes(x=time, y=event, color=as.character(value))) + geom_line()

plot(zph[1],lwd=2)
abline(0,0, col=1,lty=3,lwd=2)
abline(h= model.coxph$coef[1], col=3, lwd=2, lty=2) > legend("topleft",
                                                             legend=c('Reference line for null effect', "Average hazard over time", "Time-varying hazard"),
                                                             lty=c(3,2,1), col=c(1,3,1), lwd=2)
