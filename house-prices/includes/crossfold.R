# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: Export function for fitting crossfolds
# R version: R version 3.5.0 (2018-04-23)


#### Libraries ####
library(data.table)
library(doRNG)
library(doParallel)


# fit_crossfolds: return predictions for the given model when fit on crossfolds
#   indata: data.table holding input data
#   fit_mdl: function that takes a training dataset and test dataset, and
#            returns predictions on the test dataset
#   foldvar: character string holding the name of a column on indata that
#            specifies which fold each record is in
#   folds: integer specifying number of folds to use.  Ignored if foldvar is not
#          NULL
#   seed: seed to be passed to set.seed prior to generating folds
#   ...: parameters to be passed on to fit_mdl()
fit_crossfolds <- function(indata, fit_mdl, foldvar = NULL, folds = 10
                           , seed = 1234, ...) {
  if (is.null(foldvar)) {
    foldvar <- "fold"
    set.seed(seed)
    indata[, fold := ceiling(runif(n = .N) * folds)]
  }

  cl <- makeCluster(4)
  registerDoParallel(cl)

  fold_values <- unique(indata[[foldvar]])
  crossfold_preds <- foreach(fold = fold_values
                             , .packages = "data.table") %dorng% {
    idx <- indata[[foldvar]] != fold
    train <- indata[idx, ]
    test <- indata[!idx, ]
    test[, preds := fit_mdl(train, test, ...)]
    test[, .(Id, preds)]
  }

  stopCluster(cl)

  crossfold_preds <- rbindlist(crossfold_preds)
  return(crossfold_preds[order(Id), preds])
}
