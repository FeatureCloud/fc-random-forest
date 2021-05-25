# Random Forest FeatureCloud App

## Description
A Random Forest FeautureCloud app, allowing a federated computation of the random forest algorithm.
Supports both classification and regression.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data


## Output
- pred.csv containing the predicted class or value 
- prob.csv containing the prediction probability (classification only)
- train.csv containing the local training data
- test.csv containing the local test data

## Workflows
Can be combined with the following apps:
- Pre: Cross Validation, Normalization, Feature Selection
- Post: Regression Evaluation, Classification Evaluation

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_random_forest:
  input:
    train: "train.csv"
    test: "test.csv"
  output:
    pred: "pred.csv"
    proba: "proba.csv"
    test: "test.csv"
  format:
    sep: ","
    label: "Class"
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
  estimators: 100 # number of trees in the forest
  mode: classification # classification or regression
  random_state: 42 # random state for reproducibility
```
