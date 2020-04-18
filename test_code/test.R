rm(list=ls())
# Importing the dataset
library(tree)
library(randomForest)
library(tidyverse)

ziptrain <- read.csv('train.csv')
ziptest <- read.csv('test.csv')

ziptrain$y <- as.factor(ziptrain$y)

set.seed(1)

idx.val <- sample(1:nrow(ziptrain),2000,replace=FALSE)
zipval <-ziptrain[idx.val,]
ziptrain <-ziptrain[-idx.val,]

tree.full <- tree(y~.,data=ziptrain,split="deviance",
                  control=tree.control(nobs=nrow(ziptrain),minsize=1,
                                       mindev=0))
plot(tree.full)

tree.pruned <- prune.tree(tree=tree.full,k=1:200,newdata = zipval)
plot(tree.pruned)

min(tree.pruned$dev)

best.tune <- which.min(tree.pruned$dev)
best.tune

best.tree <- prune.misclass(tree.full,best=best.tune)
plot(best.tree)

bag.digits=randomForest(y~.,data=ziptrain, importance=TRUE,
                        na.action=na.omit, ntree=500,xtest=zipval[,-17],
                        ytest=zipval[,17])

plot(bag.digits)
which.min(bag.digits$err.rate[,1])
min(bag.digits$err.rate[,1])

tree.pred <- predict(best.tree,ziptrain[,-17],type='class')
1-sum(diag(table(tree.pred,ziptrain[,17])))/nrow(ziptrain)

tree.pred_val <- predict(best.tree,zipval[,-17],type='class')
sum(diag(table(tree.pred_val,zipval[,17])))/nrow(zipval)

1-bag.digits$test$err.rate[which.min(bag.digits$err.rate[,1]),1]

#Random filling ===============================================================
rm(list=ls())

ziptrain <- read.csv('train.csv')
ziptest <- read.csv('test.csv')
ziptrain$outcome_old[ ziptrain$outcome_old == "na" ] <- "other"
ziptrain[ ziptrain == "na" ] <- NA

set.seed(222)

#ziptrain.imputed <- rfImpute(y~.,ziptrain)
summary(ziptrain.imputed)

idx.val <- sample(1:nrow(ziptrain),2000,replace=FALSE)
zipval <-ziptrain[idx.val,]
ziptrain <-ziptrain[-idx.val,]


dataset = read.csv('train.csv')

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

dataset[ dataset == "na" ] <- NA

#Factor like columns
dataset$job=as.integer(dataset$job)
dataset$marital=as.integer(dataset$marital)
dataset$education=as.integer(dataset$education)
dataset$device=as.integer(dataset$device)
dataset$outcome_old=as.integer(dataset$outcome_old)
dataset[is.na(dataset)] <- 0

library("gbm")
set.seed(77850)

gbm.fit <- gbm(y~.,
                distribution="bernoulli",
                data=dataset,
                n.trees=750,
                interaction.depth=4,
                shrinkage=0.01,cv.folds = 3)

summary(gbm.fit)

opt_n_trees <- gbm.perf(gbm.fit, method="cv")
opt_n_trees

#=========================================================================================
rm(list=ls())

AutoHp <- read.csv('train.csv')
AutoHp <- AutoHp[,c(17,1:16)]
AutoHp$y = factor(AutoHp$y, levels = c(0, 1))

AutoHp$job=as.integer(AutoHp$job)
AutoHp$marital=as.integer(AutoHp$marital)
AutoHp$education=as.integer(AutoHp$education)
AutoHp$device=as.integer(AutoHp$device)
AutoHp$outcome_old=as.integer(AutoHp$outcome_old)


library(tidyverse)
library(ggplot2)
library(GGally)

library(mthemer)
library(ISLR)

# Define palette with colors for later plots
palette = c(tolBlue = "#4477AA",
            tolRed = "#EE6677",
            tolGreen = "#228833",
            tolYellow = "#CCBB44",
            tolCyan = "#66CCEE",
            tolPurple = "#AA3377",
            tolGrey = "#BBBBBB") %>% unname()

# Look at data
head(Auto)

# Create powers of HP
#AutoHp <- data.frame(matrix(nrow=nrow(Auto), ncol=16))

# 2) Write function
xval_lm <- function(dat, nfolds){
  ## data.frame integer -> numeric_vector
  ## produces data.frame with the cross-validation errors across the folds 
  ## ASSUME: dat has already shuffled rows
  ## ASSUME: nfolds > 1
  
  # split data
  groups_id <- rep(1:nfolds, length=NROW(dat))
  split_rows <- split(1:NROW(dat), groups_id)
  
  # Perform nfolds runs of cross validation
  J <- numeric(nfolds)
  
  for (r in 1:nfolds){
    # Take training and validation
    train_id <- unname(unlist(split_rows[-r]))
    dat_train <- dat[train_id, ]
    dat_val <- dat[-train_id, ]
    y_val <- dat_val$y
    
    # Fit parameter
    my_fit <- lm(y ~ ., data = dat_train)
    
    # Evaluate on validation set
    J[r] <- mean((y_val - predict(my_fit, newdata = dat_val))^2)
    
  }
  
  # Return vector with cross validation errors
  return(J)
}

# 3) Run 10-fold cross validation

# Shuffle rows *before* doing cross validation
shuffled_rows <- sample(1:nrow(AutoHp), replace=FALSE)
AutoHp <- AutoHp[shuffled_rows, ]


head(AutoHp)
# Do xval for all models k = 1, ..., 10
nfolds <- 16
nmodels <- 16

xval_perf <- matrix(nrow = nmodels, ncol = 2)

for (k in 1:nmodels){
  dat <- AutoHp[, c(1:k, 16)]
  perf <- xval_lm(dat, nfolds)
  
  # Average xval error
  xval_perf[k, 1] <- mean(perf)
  
  # Standard error of xval error
  xval_perf[k, 2] <- sd(perf) / sqrt(nfolds)
}


# Plot results
xval_perf <- as_tibble(xval_perf) %>% 
  rename(mean_error = V1, se = V2) %>% 
  mutate(model_id = 1:n())

# Find model according to the 1 standard error rule
min_model <- which.min(xval_perf$mean_error)
min_misclass_plus_se <- xval_perf$mean_error[min_model] + 
  xval_perf$se[min_model]

plt +
  geom_hline(yintercept = min_misclass_plus_se, linetype = 2)

best_model <- min(which(xval_perf$mean_error < min_misclass_plus_se))

plt <- ggplot(data = Auto) +
  geom_point(mapping = aes(x = horsepower, y = mpg), alpha = 0.5, size = 1.5)

# Fit model with k = 2
k <- 2
dat <- AutoHp[, c(1:k, 11)]
lm.fit_k <- lm(y ~ ., data = dat)

# Make predictions on new data
y_pred <- predict(lm.fit_k, newdata = dat)

# Plot best model
pred_to_plot <- tibble(x = dat$X1, y = y_pred)
plot(pred_to_plot)
plt +
  geom_line(data = pred_to_plot,
            mapping = aes(x = x, y = y), col = palette[1], size = 1, alpha = 1)
#================================================================================================================
rm(list=ls())
library(class)
library(caret)

classSim <- read.csv('train.csv')
classSim$y = factor(classSim$y, levels = c(0, 1))

classSim$job=as.integer(classSim$job)
classSim$marital=as.integer(classSim$marital)
classSim$education=as.integer(classSim$education)
classSim$device=as.integer(classSim$device)
classSim$outcome_old=as.integer(classSim$outcome_old)

train_data=classSim[,1:16]
train_labels=classSim[,17]

maxK = 100
klist <- 1:maxK

missClassificationError=array(dim=maxK)

for (k in klist){
  y_hat=knn(train=train_data,test=train_data,cl=train_labels, k = k)
  
  missClassificationError[k]=mean(y_hat!=train_labels)
}
plot(x = missClassificationError, y =NULL, xlab="k",ylab="Misclassification error for KNN classifier", type = "l")

set.seed(11)

plotAverage_misclassificationError<-function(numberOffolds){
  n=nrow(classSim)
  id = sample(1:n)
  train_data=classSim[id,1:16]
  train_labels=classSim[id,17]
  randomIndexes=sample(1:n)
  listOf_indexesOfElementsInFold=split(randomIndexes,rep(1:numberOffolds,length=n))
  missClassificationError=matrix(nrow=numberOffolds,ncol=maxK)
  indexOfFold=1
  for(indexesOfElementsInFold in listOf_indexesOfElementsInFold){
    x_valTest=train_data[indexesOfElementsInFold,]
    y_valTest=train_labels[indexesOfElementsInFold]
    
    x_valTrain=train_data[-indexesOfElementsInFold,]
    y_valTrain=train_labels[-indexesOfElementsInFold]
    
    for (k in klist){
      y_valTest_hat=knn(train = x_valTrain, test=x_valTest, cl=y_valTrain,k=k)
      
      missClassificationError[indexOfFold,k]=mean(y_valTest_hat!=y_valTest)
    }
    
    indexOfFold=indexOfFold+1
  }

  mean_misclassificationError<-apply(missClassificationError,MARGIN = 2,mean)
  standardDeviation_misclassificationError<-apply(missClassificationError,MARGIN = 2,sd)/sqrt(numberOffolds)
  
  par(pty="s")
  plotCI(mean_misclassificationError, y=NULL, standardDeviation_misclassificationError,standardDeviation_misclassificationError,
         xlab="k",ylab="Average misclassification error with kNN")
  abline(h=min(mean_misclassificationError+standardDeviation_misclassificationError))
}

numberOffolds=10
maxK=100
plotAverage_misclassificationError(numberOffolds)

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

#===============================================================================================================================
#Ridge Lasso stuff
rm(list=ls())
library(glmnet)

dat <- read.csv('train.csv')
dat$y = factor(dat$y, levels = c(0, 1))

dat$job=as.integer(dat$job)
dat$marital=as.integer(dat$marital)
dat$education=as.integer(dat$education)
dat$device=as.integer(dat$device)
dat$outcome_old=as.integer(dat$outcome_old)

x<-as.matrix(dat[,-17])
y<-as.matrix(dat[,17])

grid <- 10^(seq(4,-2,length=61))

#Ridge
fit.ridge<- glmnet(x,y,alpha = 0, lambda = grid,family = "binomial", standardize = TRUE)
plot(fit.ridge,xvar="lambda")
cv.fit.ridge<- cv.glmnet(x,y,alpha = 0, lambda = grid,family = "binomial", standardize = TRUE)
plot(cv.fit.ridge,xvar="lambda")

#Lasso
fit.lasso<- glmnet(x,y,alpha = 1, lambda = grid,family = "binomial", standardize = TRUE)
plot(fit.lasso,xvar="lambda")
cv.fit.lasso<- cv.glmnet(x,y,alpha = 1, lambda = grid,family = "binomial", standardize = TRUE)
plot(cv.fit.lasso,xvar="lambda")

#===================================================================================
rm(list=ls())
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
                                family='binomial',alpha=0)

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
                      family='binomial',alpha=0,nfolds=5)
plot(ridge.cv)

ridge.cv$lambda.min

ridge_coeffs <- coef(ridge.cv, s="lambda.min")

pred_ridge.te <- predict(ridge.cv,newx=as.matrix(ziptest[,-17]),type='class')
pred_ridge.tr <- predict(ridge.cv,newx=as.matrix(ziptrain[,-17]),type='class')

1-mean(!(predict(ridge.cv,newx = as.matrix(ziptest[,-17]),type='class')==
         ziptest$y))

1-mean(!(predict(ridge.cv,newx = as.matrix(ziptrain[,-17]),type='class')==
         ziptrain$y))
#===============================================================================================
#Changing into LDA QDA
rm(list=ls())
library(tidyverse)
library(MASS)
classSim<-read.csv('train.csv')
classSim$y <-as_factor(classSim$y)

split = sample.split(classSim$y, SplitRatio = 0.75)
training_set = subset(classSim, split == TRUE)
test_set = subset(classSim, split == FALSE)

#change to lda() for linear qda() for quadratic
lda_fit <- lda(y~.,training_set)
qda_fit <- qda(y~.,training_set)

lda_pred <- predict(lda_fit,newdata=test_set)
qda_pred <- predict(lda_fit,newdata=test_set)

lda_cm = table(test_set[,17], lda_pred$class)
qda_cm = table(test_set[,17], qda_pred$class)

print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(lda_cm)
confusionMatrix(qda_cm)
