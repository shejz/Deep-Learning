
#https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.data
#https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.test

# Adult: UCI ML Repository 
# Income: >50K$ and < 50K$
# Predictors
# Deep Learning Models using R-Studio and H2O, MXnet
# What is Deep Learning
# How Deep learning work?
# What is Optimization in Deep Learning
# Implementation using H2O
# Implementation using MXnet

train <- read.csv('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.data',
                  header = F)
test <- read.csv('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.test',
                 skip = 1,header = F)

colnames(train) <- c('age',
                     'workclass',
                     'fnlwgt',
                     'education',
                     'education-num',
                     'marital-status',
                     'occupation',
                     'relationship',
                     'race',
                     'sex',
                     'capital-gain',
                     'capital-loss',
                     'hours-per-week',
                     'native-country',
                     'target')
colnames(test) <- c('age',
                     'workclass',
                     'fnlwgt',
                     'education',
                     'education-num',
                     'marital-status',
                     'occupation',
                     'relationship',
                     'race',
                     'sex',
                     'capital-gain',
                     'capital-loss',
                     'hours-per-week',
                     'native-country',
                    'target')
# classification
# regression
# clustering
# dimension reduction

# feed-forward: information flows one-directional
# back-propagation: information flows from input-output-input-output-input

# back-propagation algorithm optimizes the network performance using a cost function

# cost function is minimized using gradient descent algorithm

# H2O understanding:
str(train)
# hidden layers and neurons in it
# epochs
# learning rate
# activation function

# Data Sanity check:
dim(train)
dim(test) 

library(data.table)

table(is.na(train))

summary(train)

library(Hmisc)
describe(train)

# missing patterns
native-country
occupation
workclass

repl <- function(x){
  ifelse(x == ' ?',NA,x)
}

repl(train$`native-country`)

train$`native-country`
df = train[ifelse(train$`native-country`==' ?',NA,train$`native-country`),]

na.omit(df)

##################
library(data.table)
train_df <- read.table('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.data',
                  header = F,
                  sep=',',
                  na.strings = c(" ?"),
                  stringsAsFactors = F)
test_df <- read.table('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/24-04-18-11-31-04/adult.test',
                 skip = 1,
                 header=F,
                 sep=',',
                 na.strings = c(" ?"),
                 stringsAsFactors = F)

colnames(train_df) <- c('age',
                     'workclass',
                     'fnlwgt',
                     'education',
                     'education-num',
                     'marital-status',
                     'occupation',
                     'relationship',
                     'race',
                     'sex',
                     'capital-gain',
                     'capital-loss',
                     'hours-per-week',
                     'native-country',
                     'target')
colnames(test_df) <- c('age',
                    'workclass',
                    'fnlwgt',
                    'education',
                    'education-num',
                    'marital-status',
                    'occupation',
                    'relationship',
                    'race',
                    'sex',
                    'capital-gain',
                    'capital-loss',
                    'hours-per-week',
                    'native-country',
                    'target')

# check for missing values:
sapply(train_df,function(x) sum(is.na(x))/length(x))*100

sapply(test_df,function(x) sum(is.na(x))/length(x))*100

# set all missing values as missing
train_df[is.na(train_df)] <- 'Missing'

test_df[is.na(test_df)] <- 'Missing'

# summary
describe(train_df)
test_df$target = gsub("[.]","",test_df$target)
#remove leading whitespace
library(stringr)
char_col <- colnames(train_df)[sapply(train_df,is.character)]
char_col <- colnames(test_df)[sapply(test_df,is.character)]

for(i in char_col) set(train_df,j=i,value = str_trim(train_df[[i]],side = "left"))
for(i in char_col) set(test_df,j=i,value = str_trim(test_df[[i]],side = "left"))

#set all character variables as factor
fact_col <- colnames(train_df)[sapply(train_df,is.character)]
fact_col <- colnames(test_df)[sapply(test_df,is.character)]

for(i in fact_col) set(train_df,j=i,value = factor(train_df[[i]]))
for(i in fact_col) set(test_df,j=i,value = factor(test_df[[i]]))


#impute missing values
#imp1 <- impute(data = train_df,target = "target",
 #              classes = list(integer = imputeMedian(), 
  #                            factor = imputeMode()))
#imp2 <- impute(data = test_df,target = "target",
 #              classes = list(integer = imputeMedian(), 
  #                            factor = imputeMode()))

#load the package
require(h2o)

#start h2o
localH2o <- h2o.init(nthreads = -1, max_mem_size = "4G")

#load data on H2o
trainh2o <- as.h2o(train_df)
testh2o <- as.h2o(test_df)

#set variables
y <- "target"
x <- setdiff(colnames(trainh2o),y)

#train the model - without hidden layer
deepmodel <- h2o.deeplearning(x = x
                              ,y = y
                              ,training_frame = trainh2o
                              ,standardize = T
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 100
                              ,seed = 1
                              ,nfolds = 5
                              ,variable_importances = T)

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T)

1 - (5305/32561)
#0.837075 

# Hyper Parameter Optimization
deepmodel <- h2o.deeplearning(x = x
                              ,y = y
                              ,training_frame = trainh2o
                              ,standardize = T
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 100
                              ,seed = 1
                              ,nfolds = 3
                              ,hidden = 100
                              ,variable_importances = T)

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T)

1 - (5139/32561)
#0.837075 # 10 neurons in hidden layer 1
1 - (5633/32561)
#0.8270016 # 100 neurons in hidden layer

deepmodel <- h2o.deeplearning(x = x
                              ,y = y
                              ,training_frame = trainh2o
                              ,validation_frame = testh2o
                              ,standardize = T
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 100
                              ,seed = 1
                              ,hidden = 5
                              ,variable_importances = T)

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,valid = T)

# accuracy
1-(3456/16281)
#0.787728 test accuracy

# random gridsearch to find out the best parameters
# should have 2 hidden layers
# multiple activation functions
# bring regularization (l1,l2)

# steps in grid search
# define GRID
activation_opt <- c('Rectifier','RectifierWithDropout',
                    'Maxout','MaxoutWithDropout')
hidden_opt <- list(c(10,10),c(20,10),c(50,50,50))
l1_opt <- c(0,1e-3,1e-5)
l2_opt <- c(0,1e-3,1e-5)

hyper_params <- list(activation = activation_opt,
                     hidden = hidden_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)
# define the search criteria
search_criteria <- list(strategy='RandomDiscrete',max_models=10)
#
dl_grid <- h2o.grid('deeplearning'
                    ,grid_id = 'deep_learn'
                    ,hyper_params = hyper_params
                    ,search_criteria = search_criteria
                    ,training_frame = trainh2o
                    ,x=x
                    ,y=y
                    ,nfolds=5
                    ,epochs=100)

# get the best model
d_grid <- h2o.getGrid('deep_learn',
                      sort_by = 'accuracy')

best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])

h2o.performance(best_dl_model, xval = T)
