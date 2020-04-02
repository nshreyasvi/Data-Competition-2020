#here is the code with the variables all changed to numeric and also scaled

rm(list=ls())

library(ggplot2)
library(GGally)
library(tidyverse)
library(keras)
library(fastDummies)
library(caret)
library(tensorflow)


#Loading the default variables
train_set<-train <- read.csv("C:/Users/trasa/Desktop/2020 2nd semester/Machine Learning/Contest/train.csv",  stringsAsFactors = TRUE)
train_set<-as.data.frame(train_set, row.names = NULL, optional = FALSE)
test_set<-test <- read.csv("C:/Users/trasa/Desktop/2020 2nd semester/Machine Learning/Contest/test.csv",stringsAsFactors = TRUE)
str(train_set)

#mutate_if(train_set, is.integer, ~ as.numeric(.x))
summary(train_set)
summary(test_set)
str(train_set)

#Running the required things
dummy_train_data <- fastDummies::dummy_cols(train_set,remove_first_dummy = TRUE)
head(dummy_train_data)
keep <- c('age','day','month','time_spent','banner_views','banner_views_old','days_elapsed_old','X1','X2','X3','X4','y')
final <- dummy_train_data[keep]
head(final)
index <- createDataPartition(final$y, p=0.7, list=FALSE)
final.training<-dummy_train_data[index,]
final.test<-dummy_train_data[-index,]

str(final.test)
final.test<-final.test[,-c(2:5,12)]
final.test[] <- lapply(final.test, function(x) {
  if(is.integer(x)) as.numeric(as.character(x)) else x
})
sapply(final.test, class)
final.test<-cbind(final.test[,12], final.test[,-12])
names(final.test)[names(final.test) == "final.test[,12]"] <- "y"
str(final.test)

str(final.training)
final.training<-final.training[,-c(2:5,12)]
final.training[] <- lapply(final.training, function(x) {
  if(is.integer(x)) as.numeric(as.character(x)) else x
})
sapply(final.training, class)
final.training<-cbind(final.training[,12], final.training[,-12])
names(final.training)[names(final.training) == "final.test[,12]"] <- "y"
str(final.training)


#Defining the testing and training samples
str(final.training)

X_train<- final.training[,-1]
#names(X_train)[1:32] <- paste( 1:32, sep="")
summary(X_train)
  X_train<-scale(X_train[1:32])
  summary(X_train)
  str(X_train)
  y_train <- as.factor(final.training[,1])

  X_test<- final.test[,-1]
  summary(X_test)
  X_test<-scale(X_test[1:32])
  summary(X_test)
  str(X_test)
  y_test <- as.factor(final.test[,1])
#Defining the model parameters

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(X_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'sigmoid')

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)




##### Error: there is a bug here, cause he can't find the dtype. Is should be related to the dataframe converted in a matrix in py or tensorflow or keras


model %>% fit(
  X_train, y_train, 
  epochs = 100, 
  batch_size = 5,
  validation_split = 0.3
)

#Writing summaries for the model
model %>% evaluate(X_test, y_test)

#Loss graphs
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="orange", type="l")
lines(history$metrics$val_loss, col="skyblue")
legend("topright", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))

#Accuracy graphs

plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="orange", type="l")
lines(history$metrics$val_acc, col="skyblue")
legend("topleft", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))

#Predicting on our test
predictions <- model %>% predict_classes(X_test)
