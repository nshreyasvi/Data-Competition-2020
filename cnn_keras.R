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
dataset$days_num = 1
dataset$days_num[dataset$days_elapsed_old==-1] <- 0

dataset$outcome_old[ dataset$outcome_old == "other" ] <- "na"
dataset$outcome_old[ dataset$outcome_old == "na" ] <- "failure"

dataset$days_elapsed_old[dataset$days_elapsed_old==-1] <- 0

dataset$y = factor(dataset$y, levels = c(0, 1))
dataset$X1 = factor(dataset$X1, levels = c(0, 1))
dataset$X2 = factor(dataset$X2, levels = c(0, 1))
dataset$X3 = factor(dataset$X3, levels = c(0, 1))
#dataset$outcome_old = factor(dataset$outcome_old, levels = c(0,1))
dataset$days_num = factor(dataset$X3, levels = c(0, 1))

y_train <- to_categorical(dataset$y)[,2]

x_train_num <- dataset %>%
  select(-y, -month, -day, - age) %>%
  select_if(is.numeric) %>%
  as.matrix() %>%
  scale()

x_train_fac <- dataset %>%
  select(-y, -month, -day, -age) %>%
  select_if(is.factor)

dummy <- dummyVars("~.",x_train_fac)
x_train_fac <- predict(dummy, newdata=x_train_fac)

x_train <- cbind(x_train_num, x_train_fac, to_categorical(dataset$month), to_categorical(dataset$age),to_categorical(dataset$day))

model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation="relu", input_shape = c(ncol(x_train)),
              kernel_regularizer = regularizer_l1()) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 60, activation = "relu",
              kernel_regularizer = regularizer_l1()) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dense(units = 60, activation = "relu",
              kernel_regularizer = regularizer_l1()) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  x_train, y_train,
  shuffle = TRUE,
  epochs = 1000,
  batch_size = 1000,
  validation_split = 0.2,
  callback = list(
    callback_early_stopping(monitor = "val_accuracy", mode = "max", 
                            restore_best_weights = TRUE,
                            patience = 50, verbose = 0))
)

