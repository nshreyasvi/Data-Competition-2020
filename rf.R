#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')

summary(dataset)
plot(dataset$days_elapsed_old)

#Convert age to categorical variable

levels(dataset$education)

summary(lm(dataset$y~dataset$days_elapsed_old))
#important ones --> student, retired, unemployed

#Changing the data
#dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
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

#Trying to check if the days_elapsed_old can be changed to categorical for better results
#dataset$days_elapsed_old[dataset$days_elapsed_old <= 50] <- 0
#dataset$days_elapsed_old[dataset$days_elapsed_old >= 50 & dataset$days_elapsed_old<=150] <- 1
#dataset$days_elapsed_old[dataset$days_elapsed_old >= 150 & dataset$days_elapsed_old<=250 ] <- 2
#dataset$days_elapsed_old[dataset$days_elapsed_old >= 250] <- 0

dataset[ dataset == "na" ] <- NA

#Factor like columns
dataset$job=as.integer(as.factor(dataset$job))
dataset$marital=as.integer(as.factor(dataset$marital))
dataset$education=as.integer(as.factor(dataset$education))
dataset$device=as.integer(as.factor(dataset$device))
dataset$outcome_old=as.integer(as.factor(dataset$outcome_old))
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

#accuracy ~ 86.87%, balanced accuracy ~ 86.42

