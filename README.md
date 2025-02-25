---
title: "Random Forest on Wine Quality"
author: "Reem Anwar"
date: "2025-01-08"
output:
  word_document: default
  html_document: default
---

# Introduction

Previously, I conducted a linear regression model on the dataset, and since many features seem to be non-linear, the model struggled to capture those relationships effectively.

**Goal:**
*I will try a random forest algorithm to build a better model and better understand the relationship between the features and the outcome.*

# 1. Loading Libraries

To begin, I will load the necessary libraries for the analysis:

```{r}
library(caret)
library(randomForest)
library(randomForestExplainer)
library(caTools)
```

*loading the dataset*

```{r}
setwd("/Users/reemanwar/Desktop/wine+quality")
red.dt <- read.csv("winequality-red.csv", header = TRUE, sep = ";")
```

*Inspect the dataset, looking for missing values and just checking out the structure*

```{r}
head(red.dt)
sum(is.na(red.dt))
str(red.dt)
```

outcome shows that there are no missing value sin the data set. there are 14 features of the data set, all of them seem to be numerical.

**Normalization and looking for outlines will not be performed for this algorithm "random forest" since it does not effect the model as much. So it is not necessary.**

*Splitting data set into testing and training*

```{r}
set.seed(42)
split.dt.red <- createDataPartition(red.dt$quality, p = 0.7, list = FALSE)
train.red <- red.dt[split.dt.red, ]
test.red <- red.dt[-split.dt.red, ]
```

*checking for highly correlated features*

```{r}
cor_matrix <- cor(red.dt[ , -ncol(red.dt)])
high_corr <- findCorrelation(cor_matrix, cutoff = 0.9)
print(high_corr)
```

This shows us there are no highly correlated columns, this is necessary to check before building a random forest model. We can now proceed to building the model.

**Building the random forest model for classification purposes**

*I will be setting the quality as a factor and produce a confusion matrix. If the quality is higher than 7, it will be classified as high and if it is lower than 7, it will be classified as low.*

```{r}
# prepare for classification 

train.red$quality_class <- as.factor(ifelse(train.red$quality >= 7, "High", "Low"))
test.red$quality_class <- as.factor(ifelse(test.red$quality >= 7, "High", "Low"))

# classification random regression model

rf_class <- randomForest(quality_class~., data = train.red[, -which(names(train.red) == "quality")], ntree = 500)
```

*Now that we built our model, we will evaluate the performance using the testing dataset*

```{r}
# using test data 
pred_class <- predict(rf_class, test.red[, -which(names(test.red) == "quality")])

# producing a confusion matrix
conf_matrix <- confusionMatrix(pred_class, test.red$quality_class)

print(conf_matrix)

```

This model is fairly good. It is better at predicting low quality wines than higher quality ones since the sensitivity is 43.1% for high quality red wines. Accuracy is fairly high (= 89%), although some tuning can be done to improve accuracy. 

**building regression model using random forest algorithm**

```{r}
# model
rf_reg <- randomForest(quality ~ ., data = train.red, ntree = 500)

# evaluate on test data
pred_reg <- predict(rf_reg, test.red)
rmse <- sqrt(mean((test.red$quality - pred_reg)^2))

r_squared <- 1-sum((test.red$quality - pred_reg)^2) / sum((test.red$quality - mean(test.red$quality))^2)

# printing results 

print(rf_reg)
print(paste("RMSE:", rmse))
print(paste("R Squared:", r_squared))

```

This is an average model since rmse is is 45% and r squared is 67%

*Lets analyze feature importance, to understand which factors are more effective on the quality of wine*

```{r}
# for calissification
varImpPlot(rf_class, main = "Feature Importance (Classification)")

varImpPlot(rf_reg, main = "Feature Importance (Regression)")


```

# Interpreting the graphs

**Key Takeaways**

Alcohol is the primary driver of wine quality in this model, since in the regression graph we can see it has the greatest value. It has a great influence on red wine quality.

volatile.acidity and sulphates also show a great importance to the model and contribute to the quality of red wine.

 *Typically, excessive volatile acidity is perceived as unpleasant, so its relationship with quality is crucial.*

*Sulphates can influence wine's mouth feel and preservation, so their levels likely impact perceived quality.*

residual sugar and sulfur dioxide provide little to no impact on wine quality. 
