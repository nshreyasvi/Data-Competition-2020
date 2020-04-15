#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Changing the data
dataset$age<-sqrt(dataset$age)
dataset$time_spent<-sqrt(dataset$time_spent)

dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
dataset[ dataset == "na" ] <- NA

#Factor like columns
dataset$job=as.integer(as.factor(dataset$job))
dataset$marital=as.integer(as.factor(dataset$marital))
dataset$education=as.integer(as.factor(dataset$education))
dataset$device=as.integer(as.factor(dataset$device))
dataset$outcome_old=as.integer(as.factor(dataset$outcome_old))
dataset[is.na(dataset)] <- 0

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
                          y = training_set$y, 
                          ntree = 550)  


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

