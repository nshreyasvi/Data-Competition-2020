library(ggplot2)
library(GGally)
library(tidyverse)
library(keras)
library(fastDummies)
library(caret)

#Loading the default variables
train_set<-train
test_set<-test
summary(train_set)
summary(test_set)

#Running the required things
dummy_train_data <- fastDummies::dummy_cols(train_set,remove_first_dummy = TRUE)
head(dummy_train_data)

keep <- c('age','day','month','time_spent','banner_views','banner_views_old','days_elapsed_old','X1','X2','X3','X4','y')
final <- dummy_train_data[keep]
head(final)

index <- createDataPartition(final$y, p=0.7, list=FALSE)
final.training<-dummy_train_data[index,]
final.test<-dummy_train_data[-index,]

#Defining the testing and training samples
X_train <- final.training %>% 
  select(-y) %>% 
  scale()
y_train <- to_categorical(final.training$y)

X_test <- final.test %>% 
  select(-y) %>% 
  scale()
y_test <- to_categorical(final.test$y)

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


