#pkgs <- c("keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
#install.packages(pkgs)

# Load libraries
library(keras)
library(lime)          
library(tidyquant)     # dplyr(Data manipulation), ggplot2 and tidyverse     
library(rsample)       # sampling, splitting datasets
library(recipes)       # preprocessing
library(yardstick)     # measuring model performance
library(corrr)         # correlation among the predictors
library(dplyr)
library(ggplot2)
library(forcats)

# Install Keras if you have not installed before
#install_keras()

# https://s3.amazonaws.com/hackerday.datascience/114/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Import data
#setwd('/Users/pradmishra/Downloads')

churn_data_raw <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(churn_data_raw)

# Remove unnecessary data
churn_data_tbl <- churn_data_raw %>%
  select(-customerID) %>%
  drop_na() %>%
  select(Churn, everything())

glimpse(churn_data_tbl)

# Split test/training sets
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split) 

# Determine if log transformation improves correlation 
# between TotalCharges and Churn
train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(
    #Churn = Churn %>% as.factor() %>% as.numeric(),
    Churn = Churn %>% as.numeric(),
    LogTotalCharges = log(TotalCharges)
  ) %>%
  correlate() %>%
  focus(Churn) %>% # like a select function
  fashion()        # change style format to read easily

# Create recipe
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

# Print the recipe object
rec_obj

# Predictors
x_train_tbl <- bake(rec_obj, new_data = train_tbl)
x_test_tbl  <- bake(rec_obj, new_data = test_tbl)

glimpse(x_train_tbl)

# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()
model_keras %>% 
  # First hidden layer
  layer_dense(
    units              = 70, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl[,-1])) %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(
    units              = 35, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  # Compile ANN
  compile(
    optimizer = 'adagrad',              # Adaptive Subgradient Method
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )
model_keras
## Model
## 
# Fit the keras model to the training data
fit_keras <- fit(object = model_keras, 
                 x= as.matrix(x_train_tbl[,-1]), 
                 y = as.numeric(y_train_vec),
                 batch_size = 50,
                 epochs = 1500,
                 validation_split = 0.30)
# Print the final model
fit_keras

#################################################################
# Plot the training/validation history of our Keras model
plot(fit_keras) +
  theme_tq() +
  scale_color_tq() +
  scale_fill_tq() +
  labs(title = "Deep Learning Training Results")



# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, 
                                        x = as.matrix(x_test_tbl[,-1])) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, 
                                      x = as.matrix(x_test_tbl[,-1])) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)

# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)

# Precision
tibble(
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate)
)

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)

class(model_keras)

# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}

# Setup lime::predict_model() function for keras
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}

class(model_keras)

# Test our predict_model() function

predict_model(x = model_keras, newdata = x_test_tbl[,-1], type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl[,-1], 
  model          = model_keras, 
  bin_continuous = FALSE)

# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[1:10,-1], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
  kernel_width = 0.5)

plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

# Feature correlations to Churn
corrr_analysis <- x_train_tbl %>%
  mutate(Churn = y_train_vec) %>%
  correlate() %>%
  focus(Churn) %>%
  rename(feature = rowname) %>%
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature)) 

corrr_analysis

# Correlation visualization
corrr_analysis %>%
  ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
  geom_point() +
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[2]], 
               data = corrr_analysis %>% filter(Churn > 0)) +
  geom_point(color = palette_light()[[2]], 
             data = corrr_analysis %>% filter(Churn > 0)) +
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[1]], 
               data = corrr_analysis %>% filter(Churn < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = corrr_analysis %>% filter(Churn < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  theme_tq() +
  labs(title = "Churn Correlation Analysis",
       subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
       y = "Feature Importance")
