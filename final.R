#==========================Model 1.1 (C5.0)============================================
#accuracy ~ 87.15%, balanced accuracy ~ 86.76
#================================================================================================================
rm(list=ls())
library(caret)
library(C50)
library(caTools)
set.seed(123)
dataset <- read.csv('train.csv')
dataset_1 <- read.csv('test.csv')
dataset_1 <- dataset_1[,-c(1)]

dataset$y = factor(dataset$y, levels = c(0, 1))

split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

nfolds <- 5
trControl <- trainControl(method  = "cv",
                          number  = nfolds)
fit <- train(form=y ~ .,
             data = training_set,
             method     = "C5.0", 
             trControl  = trControl,
             tuneLength = 5, #5
             control = C5.0Control(earlyStopping = FALSE),
             metric     = "Accuracy")

plot(fit)

# Predicting the Test set results
y_pred = predict(fit, newdata = test_set)
#y_pred = predict(fit, newdata = dataset_1)

y_pred


#write.csv(data.frame(ID=1:4263, y=y_pred), file='prediction.csv', row.names=FALSE)
# Making the Confusion Matrix
cm = table(test_set[,17], y_pred)
cm
print("=====================================C5.0=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#==========================Model 1.2 (C5.0)============================================
#accuracy ~ 88.18%, balanced accuracy ~ 87.74
#================================================================================================================
rm(list=ls())
library(caret)
library(C50)
library(caTools)
set.seed(123)
dataset <- read.csv('train.csv')
dataset_1 <- read.csv('test.csv')
dataset_1 <- dataset_1[,-c(1)]


dataset$days_elapsed_old <- log(dataset$days_elapsed_old)
dataset_1$days_elapsed_old <- log(dataset_1$days_elapsed_old)

dataset[is.na(dataset)] <- 0
dataset_1[is.na(dataset_1)] <- 0

dataset$y = factor(dataset$y, levels = c(0, 1))

split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

nfolds <- 5
trControl <- trainControl(method  = "cv",
                          number  = nfolds)
fit <- train(form=y ~ .,
             data = training_set,
             method     = "C5.0", 
             trControl  = trControl,
             tuneLength = 5, #5
             control = C5.0Control(earlyStopping = FALSE),
             metric     = "Accuracy")

plot(fit)

# Predicting the Test set results
y_pred = predict(fit, newdata = test_set)
#y_pred = predict(fit, newdata = dataset_1)

y_pred


#write.csv(data.frame(ID=1:4263, y=y_pred), file='prediction.csv', row.names=FALSE)
# Making the Confusion Matrix
cm = table(test_set[,17], y_pred)
cm
print("=====================================C5.0=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#==================Model 2 (Random Forest)=============================
#accuracy ~ 86.87%, balanced accuracy ~ 86.42
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
classifier = randomForest(x = training_set[,-17],
                          y = training_set$y)         #, ntree = 500)  

plot(classifier)
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
#==================Model 2 (XGB Tree)=============================
#accuracy ~ 84.43%, balanced accuracy ~ 87.03
#================================================================================================================
rm(list=ls())
# Importing the dataset
dataset = read.csv('train.csv')
dataset_1 = read.csv('test.csv')
# Encoding the target feature as factor
dataset$y= factor(dataset$y, levels = c(0, 1))

dataset$days_elapsed_old <- log(dataset$days_elapsed_old)
dataset_1$days_elapsed_old <- log(dataset_1$days_elapsed_old)

dataset[is.na(dataset)] <- 0
dataset_1[is.na(dataset_1)] <- 0

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)

split = sample.split(dataset$y, SplitRatio = 0.75)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting SVM to the Training set
classifier <- train(y~., data = training_set, method = "xgbTree", 
              trControl = trainControl(method  = "cv",number  = 5), tuneLength = 5)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-17])
y_pred

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)
cm
print("======================================XGBTree=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

