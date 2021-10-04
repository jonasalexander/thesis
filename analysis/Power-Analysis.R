require(dplyr)
require(survival)
require(survminer)

# DATA
n <- 100 # participants
N <- 100 # number of samples to determine power

filename.create_data_script <- "~/Desktop/thesis/simulation/generate-data.py"
filename.simulated_data_gaussian_optimal <- paste("~/Desktop/thesis/data/generated_gaussian_optimal_n=", as.character(n) ,".csv", sep="")
filename.simulated_data_random <- paste("~/Desktop/thesis/data/generated_random_n=", as.character(n) ,".csv", sep="")

# Initialize
p_values.positive.cox <- c()
p_values.positive.logit <- c()

for (trial in c(1:N)) {
  system(paste("source /Users/jonas/.virtualenvs/thesis/bin/activate && python3 ", filename.create_data_script, "-n ", n))
  df <- read.csv(filename.simulated_data_gaussian_optimal) %>% mutate(last = as.logical(last))
  
  # COX
  df.hazard <- df %>%
    mutate(event = as.integer(df$last)) %>%
    group_by(subject.id) %>%
    mutate(time = order) %>%
    ungroup() %>%
    select(subject.id, value, event, time) %>%
    mutate(value.centered = value - mean(df$value))
  model.coxph <- coxph(Surv(time,event) ~ value, data = df.hazard)
  p_values.positive.cox <- append(p_values.positive.cox, summary(model.coxph)$logtest[3])
  
  # LOGIT
  model.glm <- glm(last ~ order + value,
                   family = binomial(link = "logit"),
                   data = df)
  p_values.positive.logit <- append(p_values.positive.logit, coef(summary(model.glm))[2,4])
}

hist(p_values.positive.cox)
hist(p_values.positive.logit)

# Initialize
p_values.negative.cox <- c()
p_values.negative.logit <- c()

for (trial in c(1:N)) {
  system(paste("source /Users/jonas/.virtualenvs/thesis/bin/activate && python3 ", filename.create_data_script, "-n ", n, " --random"))
  df <- read.csv(filename.simulated_data_random) %>% mutate(last = as.logical(last))
  
  # COX
  df.hazard <- df %>%
    mutate(event = as.integer(df$last)) %>%
    group_by(subject.id) %>%
    mutate(time = order) %>%
    ungroup() %>%
    select(subject.id, value, event, time) %>%
    mutate(value.centered = value - mean(df$value))
  model.coxph <- coxph(Surv(time,event) ~ value, data = df.hazard)
  p_values.negative.cox <- append(p_values.negative.cox, summary(model.coxph)$logtest[3])
  
  # LOGIT
  model.glm <- glm(last ~ order + value,
                   family = binomial(link = "logit"),
                   data = df)
  p_values.negative.logit <- append(p_values.negative.logit, coef(summary(model.glm))[2,4])
}

hist(p_values.negative.cox)
hist(p_values.negative.logit)
