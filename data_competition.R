#Support Vector Machine
rm(list=ls())
# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))s
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))


# Encoding the target feature as factor
dataset$y= factor(dataset$y, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)

split = sample.split(dataset$y, SplitRatio = 0.75)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])

# Fitting SVM to the Training set
#install.packages('e1071')
library(e1071)
classifier = svm(formula = y ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-17],drop=TRUE)
y_pred

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)
cm
print("======================================SVM=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#================================================================================================================
rm(list=ls())
#Random Forest Classification

# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))

# Encoding the target feature as factor
dataset$y = factor(dataset$y, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling #for higher resolution visualisation only we are using feature scaling,RF doesnt need feature scaling
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])

# Fitting Random Forest Classification to the Training set
#install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-17],
                          y = training_set$y, 
                          ntree = 700)                 

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-17])
y_pred

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)
cm
print("=====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)


#=================================================================================================================
# Logistic Regression
rm(list=ls())
# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[,1:16] = scale(training_set[,1:16])
test_set[-17] = scale(test_set[-17]) #removes third column alone

#fitting logistic regression to the training set
classifier = glm(formula = y ~ .,
                 family = binomial, #for logistic reg mention binomial
                 data = training_set)

#predicting the test set results
prob_pred = predict(classifier, type = 'response',newdata = test_set[-17])#use type = response for logistic reg
prob_pred                                                          #that will give the prob listed in the single vector
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred

#making the confusion matrix
cm = table(test_set[,17], y_pred)
cm

print("=====================================Logistic Regression=====================================")

library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#=================================================================================================================
#Naive Bayes
rm(list=ls())
# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))

# Encoding the target feature as factor
dataset$y = factor(dataset$y, levels = c(0, 1)) #labels /levels -both are same

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])

# Fitting Naive Bayes to the Training set
library(e1071)
classifier = naiveBayes(x = training_set[-17],
                        y = training_set$y) 

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-17])

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)

print("=====================================Naive Bayes=====================================")

library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)


#=================================================================================================================
#Decision Tree Classification

rm(list=ls())

# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))

# Encoding the target feature as factor
dataset$y = factor(dataset$y, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling #no need to scale,but to visualise in high resolution if we scale, the results will be fast otherwise code may break
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])

# Fitting Decision TreeClassification to the Training set
library(rpart)
classifier = rpart(formula = y ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-17], type = 'class') 

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)

print("=====================================Decision Trees=====================================")

library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)

#=================================================================================================================
# k-nearest neighbors (K-NN)

rm(list=ls())

# Importing the dataset
dataset = read.csv('train.csv')

#Numeric like variables
dataset$age=as.numeric(as.factor(dataset$age))
dataset$day=as.numeric(as.factor(dataset$day))
dataset$month=as.numeric(as.factor(dataset$month))
dataset$time_spent=as.numeric(as.factor(dataset$time_spent))
dataset$banner_views=as.numeric(as.factor(dataset$banner_views))
dataset$banner_views_old=as.numeric(as.factor(dataset$banner_views_old))
dataset$days_elapsed_old=as.numeric(as.factor(dataset$days_elapsed_old))
dataset$X1=as.numeric(as.factor(dataset$X1))
dataset$X2=as.numeric(as.factor(dataset$X2))
dataset$X3=as.numeric(as.factor(dataset$X3))
dataset$X4=as.numeric(as.factor(dataset$X4))

#Factor like columns
dataset$job=as.numeric(as.factor(dataset$job))
dataset$marital=as.numeric(as.factor(dataset$marital))
dataset$education=as.numeric(as.factor(dataset$education))
dataset$device=as.numeric(as.factor(dataset$device))
dataset$outcome_old=as.numeric(as.factor(dataset$outcome_old))

# Encoding the target feature as factor #(the values are considered as numeric values i.e 1 > 0 but we don't want that. 
#Instead we want them to consider as factors i.e 1 and 0 as two different categories.)
dataset$y = factor(dataset$y, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-17] = scale(training_set[-17])
test_set[-17] = scale(test_set[-17])

# Fitting K-NN to the Training set and predicting the test set results
#install.packages('class')
library(class)
y_pred = knn(train = training_set[, -17],
             test = test_set[, -17],
             cl = training_set[, 17],
             k = 20)
y_pred

# Making the Confusion Matrix
cm = table(test_set[, 17], y_pred)
cm

print("=====================================KNN=====================================")

library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)
