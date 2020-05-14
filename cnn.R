rm(list=ls())
library(caret)
library(caTools)
library(h2o)
library(keras)

#set.seed(123)
dataset <- read.csv('train.csv')

dataset$y = factor(dataset$y, levels = c(0, 1))
dataset$age = log(dataset$age)

dataset$days_elapsed_old[dataset$days_elapsed_old<1] <- 0
dataset$banner_views_old <- log(dataset$banner_views_old)
dataset$banner_views <- log(dataset$banner_views)

split = sample.split(dataset$y, SplitRatio = 0.85)
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

#================================================================================
# read test.csv

# convert H2O format into data frame and save as csv
df.test <- as.data.frame(pred.dl.test)
df.test <- data.frame(ImageId = seq(1,length(df.test$predict)), Label = df.test$predict)
write.csv(df.test, file = "submission.csv", row.names=FALSE)

# shut down virtual H2O cluster
h2o.shutdown(prompt = FALSE)

