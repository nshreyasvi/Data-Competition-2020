#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')

#Changing the data
dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
dataset[ dataset == "na" ] <- NA
dataset$outcome_old[dataset$outcome_old == "other"] <- "failure"
dataset$outcome_old[dataset$outcome_old == "failure"] <- NA

#Factor like columns
dataset$job=as.integer(as.factor(dataset$job))
dataset$marital=as.integer(as.factor(dataset$marital))
dataset$education=as.integer(as.factor(dataset$education))
dataset$device=as.integer(as.factor(dataset$device))
dataset$outcome_old=as.integer(as.factor(dataset$outcome_old))
dataset[is.na(dataset)] <- 0

#changing outcome to levels 1 and 0
dataset$outcome_old[dataset$outcome_old == 4] <- 1

#Changing device to 0, 1 and 2
dataset$device[dataset$device == 3] <- 2

summary(dataset)
#Removing variabes with less dependency
#dataset<-dataset[,c("device","marital","time_spent", "education","age", "outcome_old", "month","y")]

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

library(randomForest)
classifier = randomForest(x = training_set[-17],
                          y = training_set$y)         #, ntree = 500)  


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set)
y_pred

# Making the Confusion Matrix
cm = table(test_set[,17], y_pred)
cm
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#accuracy ~ 86.68%, balanced accuracy ~ 86.24

