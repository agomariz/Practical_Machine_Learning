# Coursera Practical Machine Learning Peer Assesment Project

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Modeling

Firstly, we load and  preprocess the data in order to get consistent data to construct appropriate features for training the different models. Several things were done:
1. Strings *"NA"*, *"#DIV/0!"* as well as empty strings (*""*) were considered as `NA` values.
2. Convert empty strings to `NA` values.




Once we had a clear dataset, we have a dataset with 160 variables. Among those variables, some of them can be ruled out, so we study which of those can be removed. Taking into account that a model must generalize in a proper way from a training data set, discovering the actual signal in data and ignoring the noise of contained in them, we made the following decissions:

1. All of the columns that had NA values were removed.
2. There were some columns that seemed to be metadata. We decided to remove those columns since any relationship that a model could find with those data, would not be important and would make the model to be overfitted and perform badly. Those columns that were removed are: `row index`, `user_name`, `raw_timestamp_part_1`, `raw_timestamp_part_2`, `cvtd_timestamp`, `new_window`, `num_window`.
A summary of the final processed data can be seen in the figure of below:




```r
str(processed.data)
```

```
## 'data.frame':	19622 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```


## Cross Validation

In order to perform a good Cross validation phase, we split the already processed data in two independent sets: a training set and a test set. Below, we can see the associated code for making such partition:


```r
set.seed(98765)

in.train <- createDataPartition(processed.data$classe, p = 0.6, list = FALSE)
train <- processed.data[in.train[, 1], ]
dim(train)
```

```
## [1] 11776    53
```

```r
test <- processed.data[-in.train[, 1], ]
dim(test)
```

```
## [1] 7846   53
```


The partition was done according to the `classe` variable so as to ensure the training set and test set have examples of each one, with the same proportion than in the original set. 60% of the original data was allocated to the train set and the rest was the validation set.

## Prediction

Firstly, we decided to train a random forest with all the features of the train set, apart from the classe variable.


```r
set.seed(12345)
model.rf <- train(y = as.factor(train$classe), x = train[, -classeIndex], tuneGrid = data.frame(mtry = 3), 
    trControl = trainControl(method = "cv"), method = "parRF")
```

After we had our random forest model, we measured its accuracy in both the training and the test sets:


```r
trainCM <- confusionMatrix(predict(model.rf, newdata = train[, -classeIndex]), 
    factor(train$classe))
trainCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
testCM <- confusionMatrix(predict(model.rf, newdata = test[, -classeIndex]), 
    factor(test$classe))
testCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226   16    0    0    0
##          B    2 1499   12    0    0
##          C    0    3 1355   30    1
##          D    0    0    1 1255    3
##          E    4    0    0    1 1438
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.988, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.988         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.987    0.990    0.976    0.997
## Specificity             0.997    0.998    0.995    0.999    0.999
## Pos Pred Value          0.993    0.991    0.976    0.997    0.997
## Neg Pred Value          0.999    0.997    0.998    0.995    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.160    0.183
## Detection Prevalence    0.286    0.193    0.177    0.160    0.184
## Balanced Accuracy       0.997    0.993    0.993    0.988    0.998
```


We can see how, for the train set the accuracy is 100%, having a perfect prediction for that set. Since we know that the accuracy of the training set is very optimistic, we then use the validation set in order to check the accuraccy in an independent set. We can see how we also get a high value for accuracy for the test set, very close to 100%. Besides, we can see, for the test set, a Kappa-statistic of 0.9882.
Therefore, from this perform executed in the test sample data, we expect to have a very good prediction for any new observation that we may have. In other words, the accuracy that we get, as weel as the prediction rates, sensitivity, specifity and the kappa-statistic are estimators for us, that make us think that we are very close to a perfect prediction model for activity quality from activity monitors.
### example of decision tree used by the random forest

As we already know, a random forest consist of a set of decision trees that vote, in group, in order to predict the outcome of a new observation. Here we show an example of decision tree that could be used by our random forest model:

![plot of chunk decision_tree](figure/decision_tree.png) 


This simple decision tree does not perform a good prediction by itself. However, if we configure a battery of decision trees, similar to this ones, but trained in different ways, after a overall vote, the result becomes better. In our case we obtain a result very close to 100% of accuracy when we provide a forest of 252 trees that vote the final decision.

### Variable Importance
Regarding to the importance of each variable in the final model, we can see the ranking associated with the model in the following figure:

![plot of chunk variable_importance](figure/variable_importance.png) 


## Conclusion

After completing the whole model preprocessing of data, cross validation and model training we can see that we get a very good results for both train and test sets when we choose a random forest algorithm as predicting model. This model performs greatly the prediction of activities from accelerometers measurements and, according the the obtained results, it is capable of reading very well the signal from data, ignoring the possible noise that thay may have. 
