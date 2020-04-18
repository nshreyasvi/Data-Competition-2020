rm(list=ls())
#==============================================================================
#Running Ridge lasso based logistic regression on the variables
#Converting everything to inteager format and then running
library(MASS)
library(tidyverse)
set.seed(42)

dataset<-read.csv('train.csv')
dataset$y = factor(dataset$y, levels = c(0, 1))

dataset$job=as.integer(dataset$job)
dataset$marital=as.integer(dataset$marital)
dataset$education=as.integer(dataset$education)
dataset$device=as.integer(dataset$device)
dataset$outcome_old=as.integer(dataset$outcome_old)

split = sample.split(dataset$y, SplitRatio = 0.75)
ziptrain = subset(dataset, split == TRUE)
ziptest = subset(dataset, split == FALSE)

library(glmnet)

ridge.fit <- glmnet(x=as.matrix(ziptrain[,-17]),y=ziptrain[,17],
                    family='binomial',alpha=1)

plot(ridge.fit,xvar='lambda',label=TRUE)

nlam<-length(ridge.fit$lambda)
ridge.pred.tr<-predict(ridge.fit,newx=as.matrix(ziptrain[,-17]),
                       type = 'class')
ridge.pred.te<-predict(ridge.fit,newx=as.matrix(ziptest[,-17]),
                       type='class')

ridge.train <- ridge.test <- numeric(nlam)

for (i in 1:nlam){
  ridge.train[i] <- mean(!(ridge.pred.tr[,i]==ziptrain$y))
  ridge.test[i] <- mean(!(ridge.pred.te[,i]==ziptest$y))
}

plot(log(ridge.fit$lambda),ridge.train,type='l')
lines(log(ridge.fit$lambda),ridge.test,col='red')
lines(log(ridge.fit$lambda),rep(0,nlam),lty='dotdash')

ridge.cv <- cv.glmnet(x=as.matrix(ziptrain[,-17]),y=ziptrain[,17],
                      family='binomial',alpha=1,nfolds=5)
plot(ridge.cv)

#Running on the best lambda found
ridge.cv$lambda.min

ridge_coeffs <- coef(ridge.cv, s="lambda.min")

pred_ridge.te <- predict(ridge.cv,newx=as.matrix(ziptest[,-17]),type='class')
pred_ridge.tr <- predict(ridge.cv,newx=as.matrix(ziptrain[,-17]),type='class')

1-mean(!(predict(ridge.cv,newx = as.matrix(ziptest[,-17]),type='class')==
           ziptest$y))

1-mean(!(predict(ridge.cv,newx = as.matrix(ziptrain[,-17]),type='class')==
           ziptrain$y))

#=========================================================================
rm(list=ls())
#Without converting categorical variables
library(MASS)
library(tidyverse)
library(glmnet)
set.seed(123)

dataset<-read.csv('train.csv')
dataset$y = factor(dataset$y, levels = c(0, 1))

split = sample.split(dataset$y, SplitRatio = 0.75)
ziptrain = subset(dataset, split == TRUE)
ziptest = subset(dataset, split == FALSE)

x_train<-model.matrix(y~.,ziptrain)[,-17]
x_test<-model.matrix(y~.,ziptest)[,-17]
#y <- ifelse(ziptrain$y == "pos",1, 0)

#Ridge and Lasso 
ridge.fit <- glmnet(x=x_train,y=ziptrain[,17],
                    family='binomial',alpha=1)#,lambda = ridge.cv$lambda.min)

plot(ridge.fit,xvar='lambda',label=TRUE)

nlam<-length(ridge.fit$lambda)
ridge.pred.tr<-predict(ridge.fit,newx=x_train,
                       type = 'class')
ridge.pred.te<-predict(ridge.fit,newx=x_test,
                       type='class')

ridge.train <- ridge.test <- numeric(nlam)

for (i in 1:nlam){
  ridge.train[i] <- mean(!(ridge.pred.tr[,i]==ziptrain$y))
  ridge.test[i] <- mean(!(ridge.pred.te[,i]==ziptest$y))
}

plot(log(ridge.fit$lambda),ridge.train,type='l')
lines(log(ridge.fit$lambda),ridge.test,col='red')
lines(log(ridge.fit$lambda),rep(0,nlam),lty='dotdash')

ridge.cv <- cv.glmnet(x=x_train,y=ziptrain[,17],
                      family='binomial',alpha=1,nfolds=5)
plot(ridge.cv)

#Running on the best lambda found
ridge.cv$lambda.min

ridge_coeffs <- coef(ridge.cv, s="lambda.min")

pred_ridge.te <- predict(ridge.cv,newx=x_test,type='class')
pred_ridge.tr <- predict(ridge.cv,newx=x_train,type='class')

1-mean(!(predict(ridge.cv,newx = x_test,type='class')==
           ziptest$y))

1-mean(!(predict(ridge.cv,newx = x_train,type='class')==
           ziptrain$y))

#1% increase when categorical variables are fed using this method
#===============================================================================================================================
#Accuracy of the KNN model at different k values

rm(list=ls())
library(caret)
classSim <- read.csv('train.csv')
classSim$y = factor(classSim$y, levels = c(0, 1))

nfolds <- 10
trControl <- trainControl(method  = "cv",
                          number  = nfolds)
max_k <- 100
fit <- train(form = y ~ .,
             data = classSim,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:max_k),
             trControl  = trControl,
             metric     = "Accuracy")

palette = c(tolBlue = "#4477AA",
            tolRed = "#EE6677",
            tolGreen = "#228833",
            tolYellow = "#CCBB44",
            tolCyan = "#66CCEE",
            tolPurple = "#AA3377",
            tolGrey = "#BBBBBB") %>% unname()
plot(fit, col = palette[1])


