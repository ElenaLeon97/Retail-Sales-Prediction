# Predicting Retail Sales Using Regularized Regression Models

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning/Preparation](#data-cleaningpreparation)
- [Data Analysis](#data-analysis)
- [Results/Findings](#resultsfindings)
- [Recommendations](#recommendations)
- [Limitations](#limitations)

### Project Overview
---
This project aims to predict sales for a multi-category retailer using regularized regression models. The approach includes model selection, validation, and prediction for each category, with comparisons to a baseline linear model.

### Data Sources
---
- **train.csv**: Contains 100 features (`X1` to `X100`), a target variable `y`, and a categorical variable `category`.
- **test.csv**: Includes the 100 features and `category`, used for making predictions.

### Tools
---
- R
  - `caret`
  - `leaps`
  - `glmnet`

### Data Cleaning/Preparation
---
- Dataset was split into 80% training and 20% validation.
- Data was separated by category to allow model tuning.
- Features were centered and scaled as part of preprocessing.

### Data Analysis
---
Five models were tested per category:
- Least Squares Regression (baseline)
- Fast Forward Regression (FFR)
- Lasso
- Ridge
- Elastic Net

Model performance was evaluated using RMSE and MAE. FFR achieved the best performance across all categories.

### Results/Findings
---
- **Best model**: Fast Forward Regression (FFR) for all categories.
- **Selected features**:
  - Category 1: 14
  - Category 2: 16
  - Category 3: 5
  - Category 4: 10
- **Performance** (FFR RMSE by category):
  - Category 1: 9.27
  - Category 2: 11.41
  - Category 3: 6.92
  - Category 4: 9.31

### Recommendations
---
- Adopt FFR as the primary modeling approach for each category.
- Pay special attention to improving predictions in Category 2.

### Limitations
---
- The approach only evaluates models from the regularization lecture.
- Category 2 yielded weaker results and may require alternative methods.
- Predictions are contingent on the representativeness of the training data.
