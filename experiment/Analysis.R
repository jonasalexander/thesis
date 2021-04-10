require(dplyr)
require(survival)
require(survminer)

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

# SURVIVAL ANALYSIS
df.hazard <- df %>%
  mutate(event = as.integer(df$last)) %>%
  group_by(subject.id) %>%
  mutate(time = order) %>%
  ungroup() %>%
  select(subject.id, value, event, time) %>%
  mutate(value.centered = value - mean(df$value))

# Use this for time-varying analysis
model.coxph <- coxph(Surv(time,event) ~ value, data = df.hazard)
summary(model.coxph)
ggsurvplot(survfit(model.coxph, data=df.hazard),
           ggtheme = theme_minimal())

zph <- cox.zph(model.coxph)
zph

plot(zph[1],lwd=2)
abline(0,0, col=1,lty=3,lwd=2)
abline(h= model.coxph$coef[1], col=3, lwd=2, lty=2) > legend("topleft",
                                                        legend=c('Reference line for null effect', "Average hazard over time", "Time-varying hazard"),
                                                        lty=c(3,2,1), col=c(1,3,1), lwd=2)

# Unfortunately does not look like the ratio between these stays constant
# time coefficient is messing with us
grouped <- aggregate(event ~ time + value, df.hazard, mean)
grouped %>% ggplot(aes(x=time, y=event, color=as.character(value))) + geom_line()





# Use this for non-time-varying analyses (log-rank test)
model.survfit <- survfit(Surv(time,event) ~ value, df.hazard)
model.survfit
summary(model.survfit)$table
ggsurvplot(model.survfit,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           surv.median.line = "hv", # Specify median survival
           ggtheme = theme_bw())

model.survdiff <- survdiff(Surv(time, event) ~ value, df.hazard)
model.survdiff


