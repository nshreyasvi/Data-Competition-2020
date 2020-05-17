rm(list=ls())
library(caret)
library(caTools)
library(h2o)
library(keras)

#set.seed(123)
dataset <- read.csv('train.csv')

dataset_1 <- dataset_1[,-c(1)]

summary(dataset)
dataset$days_num = 1
dataset$days_num[dataset$days_elapsed_old==-1] <- 0

dataset$outcome_old[ dataset$outcome_old == "other" ] <- "na"
dataset$outcome_old[ dataset$outcome_old == "na" ] <- "failure"

dataset$time_spent = sqrt(dataset$time_spent)

dataset$days_elapsed_old[dataset$days_elapsed_old==-1] <- 0
dataset$days_elapsed_old <- sqrt(dataset$days_elapsed_old)

dataset$y = factor(dataset$y, levels = c(0, 1))
dataset$X1 = factor(dataset$X1, levels = c(0, 1))
dataset$X2 = factor(dataset$X2, levels = c(0, 1))
dataset$X3 = factor(dataset$X3, levels = c(0, 1))
dataset$days_num = factor(dataset$days_num, levels = c(0, 1))

split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)
trData<-as.h2o(training_set)
tsData<-as.h2o(test_set)

test<-read.csv("test.csv")
test_h2o<-as.h2o(test)

res.dl <- h2o.deeplearning(x = 1:16, y = 17, trData, activation = "Tanh", hidden=rep(160,5), use_all_factor_levels = FALSE,  distribution = c("AUTO"),epochs = 20)
plot(res.dl)

#use model to predict testing dataset
pred.dl<-h2o.predict(object=res.dl, newdata=tsData)
pred.dl.df<-as.data.frame(pred.dl)

#write.csv(data.frame(ID=1:4263, y=pred.dl.df$predict), file='submission.csv', row.names=FALSE)

summary(pred.dl)
test_labels<-test_set[,17]

#calculate number of correct prediction
sum(diag(table(test_labels, pred.dl.df[,1])))/nrow(test_set)
#confusionMatrix(table(test_labels,pred.dl))

# shut down virtual H2O cluster
h2o.shutdown(prompt = FALSE)

