rm(list=ls())
library(keras)
library(tensorflow)
library(caret)

set.seed(123)

dataset <- read.csv('train.csv')

dataset$y = factor(dataset$y, levels = c(0, 1))

indexes = createDataPartition(dataset$y, p = .85, list = F)

train = dataset[indexes,]
test = dataset[-indexes,]

xtrain = as.matrix(train[,-17])
ytrain = as.matrix(train[,17])
xtest = as.matrix(test[,-17])
ytest = as.matrix(test[, 17])

dim(xtrain)

dim(ytrain)

xtrain = array(xtrain, dim = c(nrow(xtrain), 16, 1))
xtest = array(xtest, dim = c(nrow(xtest), 16, 1))

dim(xtrain)
dim(xtest)

in_dim = c(dim(xtrain)[2:3])
print(in_dim)

model = keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 2,
                input_shape = in_dim, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  loss = "mse",
  optimizer = "adam")

model %>% summary()

model %>% fit(xtrain, ytrain, epochs = 100, batch_size=16, verbose = 0)
scores = model %>% evaluate(xtrain, ytrain, verbose = 0)
print(scores)

ypred = model %>% predict(xtest)

cat("RMSE:", RMSE(ytest, ypred))

x_axes = seq(1:length(ypred))

plot(x_axes, ytest, ylim = c(min(ypred), max(ytest)),
     col = "burlywood", type = "l", lwd = 2, ylab = "medv")
lines(x_axes, ypred, col = "red", type = "l", lwd = 2)
legend("topleft", legend = c("y-test", "y-pred"),
       col = c("burlywood", "red"), lty=1, cex=0.7, lwd=2, bty='n')