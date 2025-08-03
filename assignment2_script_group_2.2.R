###################  Libraries  ###################
library(caret)
library(leaps)
library(glmnet)

###################  Upload the data  ###################
train.df <- read.csv("train.csv")
test.df <- read.csv("test.csv")

####### Question 1: Apply regression methods  ####### 
# Divide the train data into estimation (80%) and validation (20%) samples ----
## Set seed 
set.seed(1234)
## Divide the train data set
train.df$estimation_sample <-rbinom(nrow(train.df), 1, 0.80)
train_20 <- train.df[train.df$estimation_sample==0,]
train_80 <- train.df[train.df$estimation_sample==1,]
## Remove the estimation_sample column from both data sets 
train_80 <- subset(train_80, select = -estimation_sample)
train_20 <- subset(train_20, select = -estimation_sample)

# Prepare the data ----
## Create data sets per categories. ----
# Train data set
train_filtered_c1 <- train_80[train_80$category == 1, ]
train_filtered_c2 <- train_80[train_80$category == 2, ]
train_filtered_c3 <- train_80[train_80$category == 3, ]
train_filtered_c4 <- train_80[train_80$category == 4, ]
# Train-validation data set
train_filtered_c1_v <- train_20[train_20$category == 1, ]
train_filtered_c2_v <- train_20[train_20$category == 2, ]
train_filtered_c3_v <- train_20[train_20$category == 3, ]
train_filtered_c4_v <- train_20[train_20$category == 4, ]

## Trying different parameter settings to see if we can get better results ----
parameters <- c(seq(0.1, 2, by =0.1) , seq(2, 5, 0.5) , seq(5, 25, 5))
parameters1 <- c(seq(0.1, 2, by =0.1) , seq(2, 5, 0.5) , seq(5, 25, 5)) 
parameters2 <- c(seq(0.01, 2, by =0.01) , seq(0.1, 10, 0.1) , seq(5, 25, 5))
parameters3 <- c(seq(0.01, 1, by = 0.01) , seq(0.1, 1, by = 0.1) , seq(1, 10, by = 1)) 
parameters4 <- c(seq(0.01, 1, by = 0.001) , seq(0.1, 1, by = 0.1) , seq(1, 10, by = 1)) 

## Conclude with the final settings for cross-validation and parameter ----
cv_5 = trainControl(method = "cv", number = 5)
parameters <- c(seq(0.1, 2, by =0.1) , seq(2, 5, 0.5) , seq(5, 25, 5))
alpha <- seq(0.00, 1, 0.1)

# Loop through all the categories and all the regularization techniques. ----
## Initialize an empty list to store results ----
results_list <- list()

## Create unique categories ----
categories <- unique(train.df$category)

## Loop through categories and save the results in results_df ----
suppressWarnings({
  for (c in categories) {
    # Subset the data for the current category
    train_category <- train_80[train_80$category == c, ]
    test_category <- train_20[train_20$category == c, ]
    # Initialize a table string
    table_string <- sprintf("| Regression  | RMSE      | MAE       |\n|-------------|-----------|-----------|\n")
    ###########  Least Square Regression (LSR) ############
    # Create the LSR
    lsr_train <- lm(y ~ ., data = train_category)
    # Make predictions and assess performance for LSR and Append LSR results to the table string
    pred_lsr_test <- predict(lsr_train, test_category)
    rmse_lsr <- RMSE(pred_lsr_test, test_category$y)
    mae_lsr <- MAE(pred_lsr_test, test_category$y)
    table_string <- sprintf("%s| LSR         | %9f | %9f |\n", table_string, rmse_lsr, mae_lsr)
    ###########  Fast Forward Regression (FFR)  ############
    # Remove 'category' variable for FFR
    train_category_ffr <- subset(train_category, select = -category)
    test_category_ffr <- subset(test_category, select = -category)
    # Run the FFR - 80% FFR
    ffr_train <- train(y ~ ., data = train_category_ffr, method = "leapForward", 
                       trControl = cv_5, preProc = c("center", "scale"), 
                       tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
    # Make predictions and assess performance for FFR on 20% and Append FFR results to the table string
    pred_ffr_test <- predict(ffr_train, newdata = test_category_ffr)
    rmse_ffr <- RMSE(pred_ffr_test, test_category_ffr$y)
    mae_ffr <- MAE(pred_ffr_test, test_category_ffr$y)
    table_string <- sprintf("%s| FFR         | %9f | %9f |\n", table_string, rmse_ffr, mae_ffr)
    ###########  Lasso  ############
    # Remove 'category' variable for Lasso
    train_category_lasso <- subset(train_category, select = -category)
    test_category_lasso <- subset(test_category, select = -category)
    # Run Lasso
    elnet_lasso_train <- train(
      y ~ ., data = train_category_lasso,
      method = "glmnet",
      trControl = cv_5,
      preProc = c("center", "scale"),
      tuneGrid = expand.grid(alpha = 1, lambda = parameters)
    )
    ###########  Testing Lasso  ############
    pred_lasso_test <- predict(elnet_lasso_train, test_category_lasso)
    # Performance Metrics for Lasso and Append Lasso results to the table string
    rmse_lasso <- RMSE(pred_lasso_test, test_category_lasso$y)
    mae_lasso <- MAE(pred_lasso_test, test_category_lasso$y)
    table_string <- sprintf("%s| Lasso       | %9f | %9f |\n", table_string, rmse_lasso, mae_lasso)
    ###########  Ridge  ############
    # Remove 'category' variable for Ridge
    train_category_ridge <- subset(train_category, select = -category)
    test_category_ridge <- subset(test_category, select = -category)
    # Run Ridge
    elnet_ridge_train <- train(
      y ~ ., data = train_category_ridge,
      method = "glmnet",
      trControl = cv_5,
      preProc = c("center", "scale"),
      tuneGrid = expand.grid(alpha = 0, lambda = parameters)
    )
    ## Performance metrics for Ridge and Append Ridge results to the table string
    est_elnet_ridge <- predict(elnet_ridge_train, test_category_ridge) 
    rmse_elnet_ridge_est <- RMSE(est_elnet_ridge, test_category_ridge$y) 
    mae_elnet_ridge_est <- MAE(est_elnet_ridge, test_category_ridge$y) 
    table_string <- sprintf("%s| Ridge       | %9f | %9f |\n", table_string, 
                            rmse_elnet_ridge_est, mae_elnet_ridge_est)
    ########### Elastic Net ############
    alpha <- seq(0.00, 1, 0.1)
    elnet_model <- train(
      y ~ ., data = train_category,
      method = "glmnet",
      trControl = cv_5,
      preProc = c("center", "scale"),
      tuneGrid = expand.grid(alpha = alpha, lambda = parameters)
    )
    # Assess in-sample performance
    est_elnet <- predict(elnet_model, train_category)
    rmse_elnet_est <- RMSE(est_elnet, train_category$y)
    mae_elnet_est <- MAE(est_elnet, train_category$y)
    # Assess out-of-sample performance
    pred_elnet <- predict(elnet_model, test_category)
    rmse_elnet <- RMSE(pred_elnet, test_category$y)
    mae_elnet <- MAE(pred_elnet, test_category$y)
    # Append Elastic Net results to the table string and Print the table for the current category
    table_string <- sprintf("%s| Elastic Net | %9f | %9f |\n", table_string, rmse_elnet, mae_elnet)
    cat(sprintf("Category %d\n%s\n", c, table_string))
    cat("\n")  # Separate tables with a newline
    #### Store results in a list####
    results_list[[as.character(c)]] <- list(
      category = c,
      rmse_lsr = rmse_lsr,
      mae_lsr = mae_lsr,
      rmse_ffr = rmse_ffr,
      mae_ffr = mae_ffr,
      rmse_lasso = rmse_lasso,
      mae_lasso = mae_lasso,
      rmse_elnet = rmse_elnet,
      mae_elnet = mae_elnet
    )}})

results_df <- do.call(rbind, results_list)

####### Question 2: Choose a model and explain its impact  ####### 
#Different model analyses were conducted to examine which method was the most appropriate one for 
#each of the four categories to provide predictions for the test data. More specifically, to assess 
#the goodness of the models (the difference between predicted values and actual values), loss functions 
#were used in a split data set (train) of  80% (in-sample data) and 20% (out-of-sample data) observations. 
#Precisely, 80% of the data was used to train the models and the rest to validate their fitness. 
#Hence, the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the 20% observations were taken 
#into account to determine the most optimal model for each category.  The models used were 
#“Least regression”, “Fast forward regression”, “Ridge regression”, “Lasso regression” and 
#“Elastic net regression”. Since the “Least regression” always gives the better fit, only the other 
#three models were considered for the final choice. The results with the lowest number were preferred, 
#meaning that they give higher accuracy, better performance of the predictive model and thus more reliable 
#predictions. According to them, “Fast forward regression” appears to be the best model for all the categories. More specifically, the RMSE and MAE results per category were the following ones:
#Category 1: RMSE= 9.267741    MAE= 7.406655
#Category 2: RMSE= 11.407585   MAE= 9.144513
#Category 3: RMSE=6.924911     MAE=5.547456
#Category 4: RMSE= 9.314774    MAE=7.400427
# Regarding the impact FFR has, there are a few things worth noting. First of all, it provides 
#us with the ideal number of features for each category, 14 got category 1, 16 for category 2, 
#5 for category 3 and 10 for category 4. Also, there is consistency across the categories, as shown 
#by the relatively consistent performance across categories. Category 3 stands out, delivering the 
#lowest RMSE (6.92) and MAE (5.55). This suggests that FFR excels in capturing patterns and making 
#accurate predictions in this specific category. However we found category 2 challenging, since they 
#have the highest RMSE (11.41) and MAE (9.14). This indicates potential limitations in accurately predicting 
#outcomes within this category.
# Plots for feature selection. ----
# Category 1----
## Run the FFR.
ffr_train_c1 = train(y  ~ ., data = train_filtered_c1, method = "leapForward", trControl =cv_5, preProc = c("center","scale"), tuneGrid = expand.grid(nvmax = seq(1,30,1)))
## Plot the results.
plot(ffr_train_c1) # We conclude with 14

# Category 2 ----
## Run the FFR.
ffr_train_c2 <- train(y ~ ., data = train_filtered_c2, method = "leapForward", trControl = cv_5, preProc = c("center", "scale"), tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
## Plot the results.
plot(ffr_train_c2) # We conclude with 16.

# Category 3 ----
## Run the FFR.
ffr_train_c3 = train(y ~ ., data = train_filtered_c3, method = "leapForward", trControl = cv_5, preProc = c("center", "scale"), tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
## Plot the results.
plot(ffr_train_c3) # We conclude with 5.

# Category 4 ----
## Run the FFR.
ffr_train_c4 = train(y ~ ., data = train_filtered_c4, method = "leapForward", trControl = cv_5, preProc = c("center", "scale"), tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
## Plot the results.
plot(ffr_train_c4) # We conclude with 10.
####### Question 3: Create prediction for the test.csv dataset  ####### 
# Divide test data set into categories ----
test_filtered_c1 <- test.df[test.df$category == 1, ]
test_filtered_c2 <- test.df[test.df$category == 2, ]
test_filtered_c3 <- test.df[test.df$category == 3, ]
test_filtered_c4 <- test.df[test.df$category == 4, ]

# Create the predictions for each category ----
## Category 1
ffr_model_c1 <- train(y ~ ., data = train_filtered_c1, method = "leapForward", 
                   trControl = cv_5, preProc = c("center", "scale"), 
                   tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
predictions_c1_ffr <- predict(ffr_model, test_filtered_c1)
## Category 2
ffr_model_c2 <- train(y ~ ., data = train_filtered_c2, method = "leapForward", 
                   trControl = cv_5, preProc = c("center", "scale"), 
                   tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
predictions_c2_ffr <- predict(ffr_model, test_filtered_c2)
## Category 3
ffr_model_c3 <- train(y ~ ., data = train_filtered_c3, method = "leapForward", 
                   trControl = cv_5, preProc = c("center", "scale"), 
                   tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
predictions_c3_ffr <- predict(ffr_model, test_filtered_c3)
## Category 4
ffr_model_c4 <- train(y ~ ., data = train_filtered_c4, method = "leapForward", 
                   trControl = cv_5, preProc = c("center", "scale"), 
                   tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
predictions_c4_ffr <- predict(ffr_model, test_filtered_c4)
# Combine predictions into a single Data Frame ----
all_predictions <- data.frame(
  predictions_group_x = c(predictions_c1_l, predictions_c2_l, predictions_c3_ffr, predictions_c4_elnet)
)
# Set row names of all_predictions to match the row numbers of test.df ----
rownames(all_predictions) <- rownames(test.df)

# Save the predictions to a CSV file ----
write.csv(all_predictions, file = "assignment2_predictions_group_2.2.csv", row.names = TRUE)

#Predictions are saved in another attached csv.

####### Question 4:  Replicate the best model with a plain linear regression  #######
#######                 for category 1 and make predictions.                  ####### 
# Estimate the Least Square Regression for Category 1 ----
## Train in 80% -
lsr_train_c1 <- lm(y ~ ., data = train_filtered_c1)
summary(lsr_train_c1)
##  Predict in test.df -
pred_lsr_c1 <- predict(lsr_train_c1, test.df)

# Estimate the Fast Forward Regression for Category 1 ----
## Train in train 80% -
ffr_model_c1 <- train(y ~ ., data = train_filtered_c1, method = "leapForward", 
                      trControl = cv_5, preProc = c("center", "scale"), 
                      tuneGrid = expand.grid(nvmax = seq(1, 30, 1)))
ffr_model_c1
##  Predict in test.df 
pred_ffr_c1 <- predict(ffr_model, test.df)

# Are they different? ----
## Calculate the differences
differences_c1 <- pred_lsr_c1 - pred_ffr_c1
## Summary statistics of differences
summary(differences_c1)
## Mean difference
mean_difference <- mean(differences_c1)
## Standard deviation of differences
sd_difference <- sd(differences_c1) 
## Plot the results
hist(differences_c1, main = "Differences between LSR and FFR Predictions", xlab = "Difference")
abline(v = mean_difference, col = "red", lwd = 2)
abline(v = mean_difference + sd_difference, col = "blue", lty = 2)
abline(v = mean_difference - sd_difference, col = "blue", lty = 2)
legend("topright", legend = sprintf("Mean Difference: %.2f\nSD: %.2f", mean_difference, sd_difference), col = c("red", "blue"), lwd = c(2, 1), lty = c(1, 2))
## Show the difference in predictions
plot(
  pred_lsr_c1, pred_ffr_c1,
  xlab = "Predictions from LSR", ylab = "Predictions from FFR",
  main = "Comparison of Predictions",
  col = c("blue", "green") )
# Add a diagonal line for reference (if predictions are identical, points will fall along this line)
abline(0, 1, col = "black")
#We compared LSR and FFR models for Category 1. The mean difference is -0.34 with a 
#standard deviation of 1.12, indicating our model does not differ that much from the BLUE model.
