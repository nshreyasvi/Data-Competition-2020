rm(list=ls())
library(caret)
library(C50)
library(caTools)
set.seed(123)
dataset <- read.csv('train.csv')
dataset_1 <- read.csv('test.csv')

dataset$y = factor(dataset$y, levels = c(0, 1))
dataset$age <- log(dataset$age)

#dataset_1$age <- log(dataset$age)
dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0

split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

nfolds <- 4
trControl <- trainControl(method  = "cv",
                          number  = nfolds)

fit <- train(form = y ~ .,
             data = training_set,
             method     = "C5.0", 
             trControl  = trControl,
             tuneLength = 4, #5
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
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#87.15 percent, 86.76% balanced accuracy

