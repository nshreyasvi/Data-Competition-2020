#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')
dataset_1 = read.csv('test.csv')

#Changing the data
#dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
#dataset[ dataset == "na" ] <- NA

#Plotting to check outliers
summary(dataset)

#dataset$age=sqrt(dataset$age)
#dataset$time_spent=sqrt(dataset$age)

#Factor like columns
dataset$job=as.integer(as.factor(dataset$job))
dataset$marital=as.integer(as.factor(dataset$marital))
dataset$education=as.integer(as.factor(dataset$education))
dataset$device=as.integer(as.factor(dataset$device))
dataset$outcome_old=as.integer(as.factor(dataset$outcome_old))

#dataset[is.na(dataset)] <- 0

#Less dependent variables (X2, month, day, education marital job age i.e. 1,2,3,4,6,7,14)
cor(dataset)

#Removing variabes with less dependency
#dataset<-dataset[-c(1,6,7,14)]

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
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])


library(randomForest)
classifier = randomForest(x = training_set[-17],
                          y = training_set$y, 
                          ntree = 750)  


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set)
y_pred

#write.csv(data.frame(ID=1:4263, y=y_pred), file='prediction.csv', row.names=FALSE)

# Making the Confusion Matrix
#cm = table(dataset_1, y_pred)
cm = table(test_set[,17], y_pred)
cm
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

