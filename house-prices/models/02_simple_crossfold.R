# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: test different models on crossfolds
# R version: R version 3.5.0 (2018-04-23)


#### Libraries ####
library(data.table)
library(dplyr)
library(rprojroot)

base_dir <- find_root(criteria$is_git_root)
# load fit_crossfolds()
source(file.path(base_dir, "house-prices", "includes", "crossfold.R"))


#### Inputs ####
train_csv <- file.path(base_dir, "house-prices", "data", "train.csv")
test_csv  <- file.path(base_dir, "house-prices", "data", "test.csv")


#### Outputs ####
submission_csv <- file.path(base_dir, "house-prices", "models", "output"
                            , "02_submission.csv")


#### Code ####
train <- fread(train_csv, check.names = TRUE)
test  <- fread(test_csv, check.names = TRUE)

train[, logSalePrice := log(SalePrice)]

# Fit an intercept only model
fit_int_mdl <- function(train, test) {
  mdl <- lm(data = train, formula = "logSalePrice ~ 1")
  return(predict(mdl, test))
}

crossfold_preds_int <- fit_crossfolds(train, fit_int_mdl)

# Helper function to score the model
score_model <- function(preds) {
  # Scoring metric is RMSE of the log of the sales prices
  RMSE <- sqrt(mean((preds - train$logSalePrice)^2))
  return(RMSE)
}
score_model(crossfold_preds_int)
# 0.3994

# Try our simple model from program 01
fit_simple_mdl <- function(train, test) {
  mdl <- lm(data = train, formula = "logSalePrice ~ GrLivArea")
  return(predict(mdl, test))
}

crossfold_preds_simple <- fit_crossfolds(train, fit_simple_mdl)

score_model(crossfold_preds_simple)
# 0.2855

# Try a slightly more complex model
train[, age := YrSold - YearBuilt]
fit_mdl <- function(train, test) {
  mdl_vars <- c("GrLivArea", "age", "CentralAir", "OverallQual", "OverallCond")
  mdl <- lm(data = train
            , formula = paste("logSalePrice ~ "
                              , paste(mdl_vars, collapse = "+")))
  return(predict(mdl, test))
}

crossfold_preds <- fit_crossfolds(train, fit_mdl)

score_model(crossfold_preds)
# 0.1762

# Score the test dataset using our model
test[, age := YrSold - YearBuilt]
test[, SalePrice := exp(fit_mdl(train, test))]
summary(test$SalePrice)
fwrite(test[, .(Id, SalePrice)], file = submission_csv)
# Scores 0.16425 on leaderboard
