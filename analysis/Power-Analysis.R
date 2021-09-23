require(dplyr)
require(survival)
require(survminer)

# DATA
n <- 200 # participants
N <- 10 # number of samples to determine power

filename.create_data_script <- "~/Desktop/thesis/simulation/generate-data.py"
filename.simulated_data <- "~/Desktop/thesis/data/generated_script.py"

# Initialize
p_values.positive.cox <- c()
p_values.positive.logit <- c()

for (trial in c(1:N)) {
  system(paste("python3 ", filename.create_data_script, "-n ", n))
  df <- read.csv(filename.simulated_data) %>% mutate(last = as.logical(last))
  
  # COX
  df.hazard <- df %>%
    mutate(event = as.integer(df$last)) %>%
    group_by(subject.id) %>%
    mutate(time = order) %>%
    ungroup() %>%
    select(subject.id, value, event, time) %>%
    mutate(value.centered = value - mean(df$value))
  p_values.positive.cox <- append(p_values.cox, summary(model.coxph)$logtest[3])
  
  # LOGIT
  model.glm <- glm(last ~ order + log(value),
                   family = binomial(link = "logit"),
                   data = df)
  p_values.positive.logit <- append(p_values.logit, coef(summary(model.glm))[2,4])
}

# Initialize
p_values.negative.cox <- c()
p_values.negative.logit <- c()

for (trial in c(1:N)) {
  system(paste("python3 ", filename.create_data_script, "-n ", n, " --random"))
  df <- read.csv(filename.simulated_data) %>% mutate(last = as.logical(last))
  
  # COX
  df.hazard <- df %>%
    mutate(event = as.integer(df$last)) %>%
    group_by(subject.id) %>%
    mutate(time = order) %>%
    ungroup() %>%
    select(subject.id, value, event, time) %>%
    mutate(value.centered = value - mean(df$value))
  p_values.negative.cox <- append(p_values.cox, summary(model.coxph)$logtest[3])
  
  # LOGIT
  model.glm <- glm(last ~ order + log(value),
                   family = binomial(link = "logit"),
                   data = df)
  p_values.negative.logit <- append(p_values.logit, coef(summary(model.glm))[2,4])
}
