# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: test random forest
# R version: R version 3.5.0 (2018-04-23)


#### Libraries ####
library(data.table)
library(randomForest)
library(rprojroot)

base_dir <- find_root(criteria$is_git_root)
# load fit_crossfolds()
source(file.path(base_dir, "house-prices", "includes", "crossfold.R"))


#### Inputs ####
train_csv <- file.path(base_dir, "house-prices", "data", "train.csv")
test_csv  <- file.path(base_dir, "house-prices", "data", "test.csv")


#### Outputs ####
submission_csv <- file.path(base_dir, "house-prices", "models", "output"
                            , "03_submission.csv")


#### Code ####
set.seed(1234)

train <- fread(train_csv, check.names = TRUE)
test  <- fread(test_csv, check.names = TRUE)

train[, logSalePrice := log(SalePrice)]
# drop SalePrice so that randomForest doesn't use it
train[, SalePrice := NULL]

# randomForest seems happier when you do all variable manipulations in a single
# data.table
train[, fold := "train"]
test[, `:=`(
  fold = "test"
  , logSalePrice = 0
)]
data <- rbind(train, test)

# Convert integer columns to numeric to avoid warnings from mean() when imputing
intvars <- names(data)[vapply(data, is.integer, logical(1))]
data[, (intvars) := lapply(.SD, as.numeric), .SDcols = intvars]

# Impute missings
impute <- function(dt) {
  for (var in names(dt)) {
    if (any(is.na(dt[[var]]))) {
      if (is.numeric(dt[[var]])) {
        dt[is.na(dt[[var]]), (var) := mean(dt[[var]], na.rm = TRUE)]
      } else {
        # Impute to mode
        dt[is.na(dt[[var]])
           , (var) := names(sort(table(dt[[var]]), decreasing = TRUE)[1])]
      }
    }
  }
}
impute(data)

# randomForest requires factors.  Do this after imputing to avoid errors with NA
charvars <- names(data)[vapply(data, is.character, logical(1))]
data[, (charvars) := lapply(.SD, as.factor), .SDcols = charvars]

# Split train and test so that we can use fit_crossfolds
train <- data[fold == "train", ]
test <- data[fold == "test", ]
train[, fold := NULL]
test[, fold := NULL]

# Fit a random forest
fit_mdl <- function(train, test) {
  # Make a copy so that we can drop fold without modifying source table
  train <- copy(train)
  train[, fold := NULL]
  # Explicitly state randomForest:: so that foreach picks up on the dependency
  # when we run crossfolds in parallel
  mdl <- randomForest::randomForest(logSalePrice ~ ., data = train)
  return(predict(mdl, test))
}

system.time(
  crossfold_preds <- fit_crossfolds(train, fit_mdl)
)

# Helper function to score the model
score_model <- function(preds) {
  # Scoring metric is RMSE of the log of the sales prices
  RMSE <- sqrt(mean((preds - train$logSalePrice)^2))
  return(RMSE)
}
score_model(crossfold_preds)
# 0.1355

# Score the test dataset using our model
test[, SalePrice := exp(fit_mdl(train, test))]
summary(test$SalePrice)
fwrite(test[, .(Id, SalePrice)], file = submission_csv)
# Scores 0.14380 on leaderboard
