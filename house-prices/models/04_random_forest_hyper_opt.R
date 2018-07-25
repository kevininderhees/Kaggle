# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: optimize random forest hyperparameters
# R version: R version 3.5.0 (2018-04-23)


#### Libraries ####
library(data.table)
library(randomForest)
library(rprojroot)
library(doRNG)
library(doParallel)

base_dir <- find_root(criteria$is_git_root)
# load fit_crossfolds()
source(file.path(base_dir, "house-prices", "includes", "crossfold.R"))
# load hyperparam_grid_search()
source(file.path(base_dir, "house-prices", "includes"
                 , "hyperparam_grid_search.R"))


#### Inputs ####
train_csv <- file.path(base_dir, "house-prices", "data", "train.csv")
test_csv  <- file.path(base_dir, "house-prices", "data", "test.csv")


#### Outputs ####
submission_csv <- file.path(base_dir, "house-prices", "models", "output"
                            , "04_submission.csv")


#### Code ####
# command to change priority on Windows (like Unix "nice"):
# wmic process where handle=13260 CALL setpriority 32

# idle: 64 (or "idle")
# below normal: 16384 (or "below normal")
# normal: 32 (or "normal")
# above normal: 32768 (or "above normal")
# high priority: 128 (or "high priority")
# real time: 256 (or "realtime")

cmd <- paste0("wmic process where handle=", Sys.getpid()
              , " CALL setpriority \"idle\"")
system(cmd)


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

fit_mdl <- function(train, test, hyper = NULL) {
  # Make a copy so that we can drop fold without modifying source table
  train <- copy(train)
  train[, fold := NULL]
  # Explicitly state randomForest:: so that foreach picks up on the dependency
  # when we run crossfolds in parallel
  if (is.null(hyper)) {
    mdl <- randomForest::randomForest(logSalePrice ~ ., data = train)
  } else {
    mdl <- do.call(randomForest::randomForest
                   , c(list(formula = logSalePrice ~ ., data = train), hyper))
  }
  return(predict(mdl, test))
}









fit_mdl <- function(train, test, hyper = NULL) {
  # Make a copy so that we can drop fold without modifying source table
  train <- copy(train)
  train[, fold := NULL]

  # Explicitly state randomForest:: so that foreach picks up on the dependency
  # when we run crossfolds in parallel
  ncores <- parallel::detectCores() - 1
  cl <- parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)

  # We need to divide ntree by ncores, so if we're optimizing the ntree
  # hyperparameter, pull it out of hyper and handle manually
  ntree <- 500
  if (!is.null(hyper)) {
    if (exists("ntree", hyper)) {
      ntree <- hyper[["ntree"]]
      hyper[["ntree"]] <- NULL
    }
  }

  rf_args <- list(formula = logSalePrice ~ .
                  , data = train
                  , ntree = ceiling(ntree / ncores))
  if (!is.null(hyper)) {
    rf_args <- c(rf_args, hyper)
  }

  mdl <- foreach(
    i = 1:ncores
    , .packages = c("data.table", "randomForest")
    , .combine = randomForest::combine
  ) %dorng% {
    do.call(randomForest::randomForest, rf_args)
  }

  parallel::stopCluster(cl)
  return(predict(mdl, test))
}

system.time(
  crossfold_preds <- fit_crossfolds(train, fit_mdl, parallel = FALSE)
)

# Helper function to score the model
score_model <- function(preds) {
  # Scoring metric is RMSE of the log of the sales prices
  RMSE <- sqrt(mean((preds - train$logSalePrice)^2))
  return(RMSE)
}
score_model(crossfold_preds)
# 0.1363 - slightly different from 03, since with a parallel fit_mdl our RNG is
# different

# Try different hyperparameters
system.time(
  res <- hyperparam_grid_search(
    train
    , fit_mdl
    , score_model
    , parallel = FALSE
    , hyper = list(mtry = c(5, 10, 26, 50)
                   , nodesize = c(2, 5, 10, 20, 40)
                   , ntree = c(250, 500, 1000)))
)
res[]
