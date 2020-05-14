rm(list=ls())
library(caret)
library(C50)
library(mlbench)

dataset <- read.csv('train.csv')
dataset_1 <- read.csv('test.csv')

summary(dataset)
dataset$y = factor(dataset$y, levels = c(0, 1))
#dataset$age <- log(dataset$age)
#dataset$days_elapsed_old <- log(dataset$days_elapsed_old)
#dataset$banner_views <- log(dataset$banner_views)
#dataset[is.na(dataset)] <- 0
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


C5CustomSort <- function(x) {
  
  x$model <- factor(as.character(x$model), levels = c("rules","tree"))
  x[order(x$trials, x$model, x$splits, !x$winnow),]
  
}

C5CustomLoop <- function (grid) 
{
  loop <- ddply(grid, c("model", "winnow","splits"), function(x) c(trials = max(x$trials)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$trials)) {
    index <- which(grid$model == loop$model[i] & grid$winnow == 
                     loop$winnow[i] & grid$splits == loop$splits[i])
    trials <- grid[index, "trials"]
    submodels[[i]] <- data.frame(trials = trials[trials != 
                                                   loop$trials[i]])
  }
  list(loop = loop, submodels = submodels)
}

C5CustomGrid <- function(x, y, len = NULL) {
  c5seq <- if(len == 1)  1 else  c(1, 10*((2:min(len, 11)) - 1))
  expand.grid(trials = c5seq, splits = c(2,10,20,50), winnow = c(TRUE, FALSE), model = c("tree","rules"))
}

C5CustomFit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  # add the splits parameter to the fit function
  # minCases is a function of splits
  
  theDots <- list(...)
  
  splits   <- param$splits
  minCases <- floor( length(y)/splits ) - 1
  
  if(any(names(theDots) == "control"))
  {
    theDots$control$winnow        <- param$winnow
    theDots$control$minCases      <- minCases
    theDots$control$earlyStopping <- FALSE
  }
  else
    theDots$control <- C5.0Control(winnow = param$winnow, minCases = minCases, earlyStopping=FALSE )
  
  argList <- list(x = x, y = y, weights = wts, trials = param$trials, rules = param$model == "rules")
  
  argList <- c(argList, theDots)
  
  do.call("C5.0.default", argList)
  
}

GetC5Info <- function() {
  
  # get the default C5.0 model functions
  c5ModelInfo <- getModelInfo(model = "C5.0", regex = FALSE)[[1]]
  
  # modify the parameters data frame so that it includes splits
  c5ModelInfo$parameters$parameter <- factor(c5ModelInfo$parameters$parameter,levels=c(levels(c5ModelInfo$parameters$parameter),'splits'))
  c5ModelInfo$parameters$label <- factor(c5ModelInfo$parameters$label,levels=c(levels(c5ModelInfo$parameters$label),'Splits'))
  c5ModelInfo$parameters <- rbind(c5ModelInfo$parameters,c('splits','numeric','Splits'))
  
  # replace the default c5.0 functions with ones that are aware of the splits parameter
  c5ModelInfo$fit  <- C5CustomFit
  c5ModelInfo$loop <- C5CustomLoop
  c5ModelInfo$grid <- C5CustomGrid
  c5ModelInfo$sort <- C5CustomSort
  
  return (c5ModelInfo)
  
}

c5info <- GetC5Info()

# Define the structure of cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10,  repeats = 10)

# create a custom cross validation grid
grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model=c("tree"), .splits=c(2,5,10,15,20,25,50,100) )

x <- training_set[,-17]
y <- training_set[,17]

# Tune and fit model
mdl<- train(x=x,y=y,tuneGrid=grid,trControl=fitControl,method=c5info,verbose=FALSE)

mdl

plot(mdl)

#===================================================================================================================================
rm(list=ls())
library(caret)
library(keras)
library(tidyverse)
library(caTools)

dataset <- read.csv('train.csv')
dataset_1 <- read.csv('test.csv')
dataset_1 <- dataset_1[,-c(1)]

summary(dataset)
dataset$y = factor(dataset$y, levels = c(0, 1))
dataset$X1 = factor(dataset$X1, levels = c(0, 1))
dataset$X2 = factor(dataset$X2, levels = c(0, 1))
dataset$X3 = factor(dataset$X3, levels = c(0, 1))

y_train <- to_categorical(dataset$y)[,2]

x_train_num <- dataset %>%
  select(-y, -month) %>%
  select_if(is.numeric) %>%
  as.matrix() %>%
  scale()

x_train_fac <- dataset %>%
  select(-y, -month) %>%
  select_if(is.factor)

dummy <- dummyVars("~.",x_train_fac)
x_train_fac <- predict(dummy, newdata=x_train_fac)

x_train <- cbind(x_train_num, x_train_fac, to_categorical(dataset$month))

model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation="relu", input_shape = c(ncol(x_train)),
              kernel_regularizer = regularizer_l1()) %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 60, activation = "relu",
              kernel_regularizer = regularizer_l1()) %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dense(units = 1, activation = "softmax")

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  x_train, y_train,
  shuffle = TRUE,
  epochs = 200,
  batch_size = 1000,
  validation_split = 0.2)
#  callback = list(
#    callback_early_stopping(monitor = "val_accuracy", model = "max",
#                            patience = 50, verbose = 1)
#  )
#)
