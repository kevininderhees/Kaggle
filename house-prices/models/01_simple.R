# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: create a simple model to test reading in and scoring the data


#### Libraries ####
library(data.table)
library(dplyr)
library(rprojroot)


#### Inputs ####
base_dir <- find_root(criteria$is_git_root)
train_csv <- file.path(base_dir, "house-prices", "data", "train.csv")
test_csv  <- file.path(base_dir, "house-prices", "data", "test.csv")


#### Outputs ####
submission_csv <- file.path(base_dir, "house-prices", "models", "output", "01_submission.csv")


#### Code ####
train <- fread(train_csv, check.names = TRUE)
test  <- fread(test_csv, check.names = TRUE)


# Explore target variable
summary(train$SalePrice)

hist(train$SalePrice)
# log transform looks a lot more normally distributed
train[, logSalePrice := log(SalePrice)]
hist(train$logSalePrice)

# As a baseline, fit an intercept only model
int_mdl <- lm(data = train, formula = "logSalePrice ~ 1")

# Score the model
score_model <- function(mdl) {
  preds <- mdl$fitted.values
  
  # Scoring metric is RMSE of the log of the sales prices
  RMSE <- sqrt(mean((preds - train$logSalePrice)^2))
  return(RMSE)
}
score_model(int_mdl)
# 0.399


# Try a simple model using just GrLivArea

# What's the best square feet to use? 1st floor + 2nd floor?
# What's GrLivArea (Above grade (ground) living area square feet)?
train[, diff := X1stFlrSF + X2ndFlrSF - GrLivArea]
summary(train$diff)

# Some non-zero entries.  What column are we missing?
# Inspect all numeric columns.
train_numvars <- names(train)[vapply(train, is.numeric, logical(1))]
train[diff != 0, ..train_numvars]
# diff seems to line up with LowQualFinSF
train[, diff := X1stFlrSF + X2ndFlrSF + LowQualFinSF - GrLivArea]
summary(train$diff)
# All 0 now - good

mdl <- lm(data = train, formula = "logSalePrice ~ GrLivArea")

score_model(mdl)
# 0.285 - better than intercept


# Score the test dataset using our model
test[, SalePrice := exp(predict(mdl, newdata = test))]
summary(test$SalePrice)
fwrite(test[, .(Id, SalePrice)], file = submission_csv)
# Scores 0.29619 on leaderboard
