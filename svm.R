#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')

#Changing the data
dataset$age<-sqrt(dataset$age)
#dataset$time_spent<-sqrt(sqrt(dataset$time_spent))

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
dataset<-dataset[,c("banner_views_old","days_elapsed_old", "banner_views", "education","outcome_old","device","marital","day","X1","X2","X3","X4","time_spent","y")]

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
#training_set[-14] = scale(training_set[-14])
#test_set[-14] = scale(test_set[-14])


library(randomForest)
classifier = randomForest(x = training_set[-14],
                          y = training_set$y, 
                          ntree = 1000)  


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set)
y_pred

# Making the Confusion Matrix
cm = table(test_set[,14], y_pred)
cm
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

