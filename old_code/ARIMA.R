library(readr)
library(forecast)

day_data <- read.csv('agg_btc_day.csv')
hour_data <- read.csv('agg_btc_hour.csv')

calc_pct_diffs <- function (data_df, start_col=2) {
  df <- data.frame(index = 1:(length(data_df[[1]])-1))
  for (col in colnames(data_df)[start_col:(start_col+6)]) {
    column <- data_df[[col]]
    pct_diff <- 100 * diff(column)/column[-length(column)]
    df[col] <- pct_diff
  }
  return(df)
}

split_train_test <- function (data_df, factor) {
  n <- length(data_df$index)
  split_val <- n - (365 + 120) * factor
  split_t <- n - 365 * factor - 1
  train <- data.frame(index = 1:split_val)
  trainf <- data.frame(index = 1:split_t)
  val <- data.frame(index = (split_val + 1):split_t)
  test <- data.frame(index = (split_t+1):n)
  
  for (col in colnames(data_df)[2:length(colnames(data_df))]) {
    column <- data_df[[col]]
    train[col] <- column[1:split_val]
    trainf[col] <- column[1:split_t]
    val[col] <- column[(split_val + 1):split_t]
    test[col] <- column[(split_t+1):n]
  }
  return(list(train=train, trainf=trainf, val=val, test=test))
}

calc_invest_return <- function(frc, real, start_v = 1, t_cost=0.0015, use_threshold=TRUE) {
  T <- length(real)
  budget <- start_v
  path <- numeric(T)
  prev_strat = 1
  
  if (use_threshold) {
    lt <- mean(frc) - sd(frc)
    ut <- mean(frc) + sd(frc)
  } else {
    lt <- 0
    ut <- 0
  }
  
  for (t in 1:T) {
    strategy <- prev_strat
    if (frc[t] < lt) {
      strategy <- -1
    } else if (frc[t] > ut) {
      strategy <- 1
    }
    
    if (strategy != prev_strat) {
      budget <- budget * (1 - t_cost)
    }
    prev_strat <- strategy
    budget <- budget * (1 + real[t]/100 * strategy)
    path[t] <- budget
  }
  return (list(ret=((budget - start_v) / start_v), path=path))
}

day_ret <- calc_pct_diffs(day_data)

split_day_data <- split_train_test(day_ret, factor=1)
day_train <- split_day_data$train
day_trainf <- split_day_data$trainf
day_val <- split_day_data$val
day_test <- split_day_data$test

### Estimate the ARIMA model
day_trainX <- as.matrix(day_trainf[c(-1, -5)])[1:(length(day_trainf$close) - 1),]
day_testX <- as.matrix(day_test[c(-1, -5)])[1:(length(day_test$close) - 1),]
day_retX <- as.matrix(day_ret[c(-1, -5)])[1:(length(day_ret$close) - 1),]

ytrain <- day_trainf$close[(1 + 1):length(day_trainf$close)]
ytest <- day_test$close[(1 + 1):length(day_test$close)]
yfull <- day_ret$close[(1 + 1):length(day_ret$close)]

am <- auto.arima(ytrain)
#am <- auto.arima(ytrain)
print(summary(am))
forecast <- fitted(Arima(ytest, model=am))
#forecast <- fitted(Arima(ytest, model=am))

mean_mse <- (1/length(ytest)) * sum((ytest - mean(ytrain))^2)
mean_r <- calc_invest_return(rep(1, length(ytest)), ytest, use_threshold = FALSE)

budget_mse_dev <- calc_invest_return(forecast, ytest, use_threshold = FALSE)$path
optimal_dev <- calc_invest_return(ytest, ytest, use_threshold = FALSE)$path
holding_dev <- mean_r$path
shorting_dev <- calc_invest_return(rep(-1, length(ytest)), ytest, use_threshold = FALSE)$path

#write(forecast, file="txt_forecast/arima_day_test.txt")

plot(ytest, type='l', col='black')
lines(forecast, type='l', col='red')

plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, shorting_dev), max(budget_mse_dev, holding_dev, shorting_dev)))
lines(shorting_dev, type='l', col='blue')
lines(budget_mse_dev, type='l', col='red')

full_estim <- fitted(Arima(yfull, model=am, xreg=day_retX))
#full_estim <- fitted(Arima(yfull, model=am))
print(summary(am))

budget_mse_dev <- calc_invest_return(full_estim, yfull, use_threshold = TRUE)$path
optimal_dev <- calc_invest_return(yfull, yfull, use_threshold = FALSE)$path
holding_dev <- calc_invest_return(rep(1, length(yfull)), yfull, use_threshold = FALSE)$path
shorting_dev <- calc_invest_return(rep(-1, length(yfull)), yfull, use_threshold = FALSE)$path

plot(yfull, type='l', col='black')
lines(full_estim, type='l', col='red')

plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, shorting_dev), max(budget_mse_dev, holding_dev, shorting_dev)))
lines(shorting_dev, type='l', col='blue')
lines(budget_mse_dev, type='l', col='red')

write(full_estim, file="txt_forecast/arimaX_hour_all.txt")
