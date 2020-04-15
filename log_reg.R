# Logistic Regression
rm(list=ls())
# Importing the dataset
dataset = read.csv('train.csv')

dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
dataset[ dataset == "na" ] <- NA

#Factor like columns
dataset$job=as.integer(as.factor(dataset$job))
dataset$marital=as.integer(as.factor(dataset$marital))
dataset$education=as.integer(as.factor(dataset$education))
dataset$device=as.integer(as.factor(dataset$device))
dataset$outcome_old=as.integer(as.factor(dataset$outcome_old))
dataset[is.na(dataset)] <- 0

# Encoding the target feature as factor

dataset<-dataset[,c("device","marital","time_spent", "education","age", "outcome_old", "month","y")]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[,1:7] = scale(training_set[,1:7])
test_set[-8] = scale(test_set[-8]) #removes third column alone

#fitting logistic regression to the training set
classifier = glm(formula = y ~ .,
                 family = binomial, #for logistic reg mention binomial
                 data = training_set)

#predicting the test set results
prob_pred = predict(classifier, type = 'response',newdata = test_set[-8])#use type = response for logistic reg
prob_pred                                                          #that will give the prob listed in the single vector
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred

#making the confusion matrix
cm = table(test_set[,8], y_pred)
cm

print("=====================================Logistic Regression=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

