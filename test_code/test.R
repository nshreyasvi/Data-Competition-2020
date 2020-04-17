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

ziptrain.imputed <- rfImpute(y~.,ziptrain)
summary(ziptrain.imputed)

idx.val <- sample(1:nrow(ziptrain),2000,replace=FALSE)
zipval <-ziptrain[idx.val,]
ziptrain <-ziptrain[-idx.val,]

library("gbm")
set.seed(77850)

gbm.fit <- gbm(y~.,
                distribution="bernoulli",
                data=ziptrain,
                n.trees=750,
                interaction.depth=4,
                shrinkage=0.01,cv.folds = 3)

summary(gbm.fit)

opt_n_trees <- gbm.perf(gbm.fit, method="cv")
opt_n_trees
