# Project: Kaggle housing prices
# Copyright: Kevin Inderhees 2018
# Description: Export function for performing a grid search for optimal
#              hyperparameters, using cross validation
# R version: R version 3.5.0 (2018-04-23)


#### Libraries ####
library(data.table)
library(doRNG)
library(doParallel)

base_dir <- find_root(criteria$is_git_root)
# load fit_crossfolds()
source(file.path(base_dir, "house-prices", "includes", "crossfold.R"))


# hyperparam_grid_search: perform a grid search across all combinations of
#                         supplied hyperparameters using cross validation.
#                         Returns a data.table with hyperparameter values
#                         and model score.
#   indata: data.table holding input data
#   fit_mdl: function that takes a training dataset and test dataset, and
#            returns predictions on the test dataset
#   score_model: function that takes a list of predictions and returns a model
#                score
#   foldvar: character string holding the name of a column on indata that
#            specifies which fold each record is in
#   folds: integer specifying number of folds to use.  Ignored if foldvar is not
#          NULL
#   seed: seed to be passed to set.seed prior to generating folds
#   hyper: list of vectors of hyperparameters to be passed on to fit_mdl()
hyperparam_grid_search <- function(indata, fit_mdl, score_model, foldvar = NULL
                                   , folds = 10, seed = 1234, hyper) {
  hyper_comb <- as.data.table(do.call(expand.grid, hyper))
  hyper_comb[, model_score := vapply(
    1:dim(hyper_comb)[1]
    , function(i) {
      message(paste("Testing hyperparameter combination", i
                    , "out of", dim(hyper_comb)[1]))
      crossfold_preds <- fit_crossfolds(indata, fit_mdl, foldvar, folds, seed
                                        , hyper_comb[i, ])
      score_model(crossfold_preds)
    }
    , numeric(1)
  )]
  hyper_comb[, model_rank := rank(model_score)]
}
