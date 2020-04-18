#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')
summary(dataset)

#Changing the data
dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
levels(dataset$education)
dataset$outcome_old[dataset$outcome_old == "other"] <- "failure"
dataset$outcome_old[dataset$outcome_old == "failure"] <- "na"
dataset$marital[dataset$marital == "divorced"] <- "single"

#Testing these
dataset$job[dataset$job == "salesman"] <- "industrial_worker"
dataset$job[dataset$job == "industrial_worker"] <- "housekeeper"
dataset$job[dataset$job == "industrial_worker"] <- "na"


dataset$job[dataset$job == "high_school"] <- "university"
dataset$job[dataset$job == "university"] <- "na"

dataset$job[dataset$job == "desktop"] <- "na"

dataset[ dataset == "na" ] <- NA

#Factor like columns
dataset$job=as.integer(dataset$job)
dataset$marital=as.integer(dataset$marital)
dataset$education=as.integer(dataset$education)
dataset$device=as.integer(dataset$device)
dataset$outcome_old=as.integer(dataset$outcome_old)
dataset[is.na(dataset)] <- 0

summary(dataset)
# Encoding the target feature as factor
dataset$y = factor(dataset$y, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling #for higher resolution visualisation only we are using feature scaling,RF doesnt need feature scaling
summary(training_set)

library(randomForest)
library(caret)
library(C50)
classifier = randomForest(x = training_set[,-17],
                          y = training_set$y)         #, ntree = 500)  

# Predicting the Test set results
y_pred = predict(fit, newdata = test_set)
y_pred

# Making the Confusion Matrix
cm = table(test_set[,17], y_pred)
cm
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#accuracy ~ 86.87%, balanced accuracy ~ 86.42

