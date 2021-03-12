---
title: "Practical ML Project - Exercise Prediction"
author: "Avijit"
date: "03^rd^ March 2021"
output:
    html_document:
        keep_md: yes
---

# Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. This project uses data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of your project is to predict the manner in which they did the exercise using the collected data. This feature is reflected in the "classe" variable in the training set.

## Importing libraries

```r
setwd("D:/R working directory/Practical ML Project")
library(RCurl)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
# Dataset
## Loading data

```r
trainFile0 <- "./pml-training.csv"
testFile0  <- "./pml-testing.csv"
if (!file.exists(trainFile0)) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                destfile=trainFile0, method="curl")}
pml.data <- read.csv(trainFile0)

if (!file.exists(testFile0)) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                destfile=testFile0, method="curl")}
pml.validation <- read.csv(testFile0)
```

## Viewing data

```r
dim(pml.data)
```

```
## [1] 19622   160
```

```r
dim(pml.validation)
```

```
## [1]  20 160
```

```r
w <- which(names(pml.data)!=names(pml.validation));w #160
```

```
## [1] 160
```

```r
names(pml.data)[w]
```

```
## [1] "classe"
```

```r
names(pml.validation)[w]
```

```
## [1] "problem_id"
```

```r
table(pml.data[,w])
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
pml.validation[,w]
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
```

## Cleaning data
Columns containing more than 90% missing values and those with user info and timestamp are removed.

```r
w <- which(colSums(is.na(pml.data)|pml.data=="")>0.9*dim(pml.data)[1])
as.vector(w)
```

```
##   [1]  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
##  [19]  30  31  32  33  34  35  36  50  51  52  53  54  55  56  57  58  59  69
##  [37]  70  71  72  73  74  75  76  77  78  79  80  81  82  83  87  88  89  90
##  [55]  91  92  93  94  95  96  97  98  99 100 101 103 104 105 106 107 108 109
##  [73] 110 111 112 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139
##  [91] 141 142 143 144 145 146 147 148 149 150
```

```r
pml.data.c <- pml.data[,-w]
pml.data.c <- pml.data.c[,-c(1:7)] #user info and timestamp
pml.data.c$classe <- as.factor(pml.data.c$classe)
dim(pml.data); dim(pml.data.c)
```

```
## [1] 19622   160
```

```
## [1] 19622    53
```
Same process is performed on validation dataset.

```r
w <- which(colSums(is.na(pml.validation)|pml.validation=="")>
               0.9*dim(pml.validation)[1])
pml.validation.c <- pml.validation[,-w]
pml.validation.c <- pml.validation.c[,-c(1:7)] #user info and timestamp
dim(pml.validation); dim(pml.validation.c)
```

```
## [1]  20 160
```

```
## [1] 20 53
```
After cleaning the data, number of columns are narrowed down from 160 to 53 relevant ones

## Splitting data
Data is split into training and testing datasets with a ratio of 70:30

```r
set.seed(666)
inTrain <- createDataPartition(y=pml.data.c$classe, p=0.7, list=F)
training <- pml.data.c[inTrain,]
testing <- pml.data.c[-inTrain,]
dim(training)
```

```
## [1] 13737    53
```

```r
dim(testing)
```

```
## [1] 5885   53
```

# Prediction Models
## 1. Random forest

```r
controlRf <- trainControl(method="cv", 5)
mod.rf <- train(classe ~ ., data=training, method="rf",
                 trControl=controlRf, ntree=250)
mod.rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 250, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 250
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.71%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    1    1    0    1 0.0007680492
## B   23 2628    7    0    0 0.0112866817
## C    0   14 2371   11    0 0.0104340568
## D    0    0   29 2220    3 0.0142095915
## E    0    1    4    3 2517 0.0031683168
```

```r
plot(mod.rf)
```

![](Practical-ML-Project_files/figure-html/random forest-1.png)<!-- -->

### Random Forest Testing

```r
pred.rf <- predict(mod.rf, testing)
cm.rf <- confusionMatrix(testing$classe, pred.rf)
cm.rf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    9 1128    2    0    0
##          C    0    5 1016    4    1
##          D    0    0   11  952    1
##          E    0    0    3    0 1079
```

```r
cm.rf$overall[1] # Accuracy of model = 99.35%
```

```
##  Accuracy 
## 0.9935429
```

```r
1 - as.numeric(cm.rf$overall[1]) # out-of-sample error is 0.65%
```

```
## [1] 0.006457094
```
Accuracy of Random Forest Model = 99.35%

## 2. Regression Trees

```r
mod.rpart <- rpart(classe ~ ., data=training, method="class")
prp(mod.rpart)
```

![](Practical-ML-Project_files/figure-html/rpart-1.png)<!-- -->

### Regression Trees Testing


```r
pred.rpart <- predict(mod.rpart, testing, type="class")
cm.rpart <- confusionMatrix(testing$classe, pred.rpart)
cm.rpart$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1513   56   47   25   33
##          B  185  625  109   92  128
##          C   21   50  825   63   67
##          D   57   79  150  603   75
##          E   17   63  121   64  817
```

```r
cm.rpart$overall[1] # Accuracy of model = 74.5%
```

```
##  Accuracy 
## 0.7447749
```

```r
1 - as.numeric(cm.rpart$overall[1]) # out-of-sample error is 25.5%
```

```
## [1] 0.2552251
```
Accuracy of Regression Trees Model = 74.5%

## 3. Gradient Boosting

```r
mod.gbm <- train(classe ~ ., data=training, verbose=F, method="gbm")
mod.gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 52 had non-zero influence.
```
### Gradient Boosting Testing

```r
pred.gbm <- predict(mod.gbm, testing)
cm.gbm <- confusionMatrix(testing$classe, pred.gbm)
cm.gbm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1651   12    4    6    1
##          B   48 1058   30    3    0
##          C    1   36  978   10    1
##          D    0    3   37  919    5
##          E    2    8   17    9 1046
```

```r
cm.gbm$overall[1] # Accuracy of model = 96%
```

```
##  Accuracy 
## 0.9604078
```

```r
1 - as.numeric(cm.gbm$overall[1]) # out-of-sample error is 4%
```

```
## [1] 0.03959218
```
Accuracy of Gradient Boosting Model = 96%

# Results
Out of the tested three machine learning prediction models, the random forest model stands out with the highest accuracy of prediction of 99.35%. So, we deploy random forest model to the validation dataset and obtain the following results.

```r
predict(mod.rf, pml.validation.c)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
After submitting to prediction assignment part of the project, we find that all 20/20 predicted values were correct.
