library(readr)
library(midasr)

day_data <- read.csv('agg_btc_day.csv')
hour_data <- read.csv('agg_btc_hour.csv')
min_data <- read.csv('agg_btc_min.csv')

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
  split <- n - 366 * factor
  train <- data.frame(index = 1:split)
  test <- data.frame(index = (split+1):n)
  
  for (col in colnames(data_df)[2:length(colnames(data_df))]) {
    column <- data_df[[col]]
    train[col] <- column[1:split]
    test[col] <- column[(split+1):n]
  }
  return(list(train=train, test=test))
}

calc_invest_return <- function(frc, real, start_v = 100) {
  T <- length(real)
  budget <- start_v
  prev_strat = 1
  transaction_cost = 0
  
  for (t in 1:T) {
    strategy <- 1
    if (frc[t] < 0) {
      strategy <- -1
    }
    
    if (strategy != prev_strat) {
      budget <- budget * (1 - transaction_cost)
    }
    prev_strat <- strategy
    budget <- budget * (1 + real[t]/100 * strategy)
  }
  return ((budget - start_v) / start_v)
}

day_ret <- calc_pct_diffs(day_data)
hour_ret <- calc_pct_diffs(hour_data)
min_ret <- calc_pct_diffs(min_data, start_col=3)

split_day_data <- split_train_test(day_ret, factor=1)
day_train <- split_day_data$train
day_test <- split_day_data$test

split_hour_data <- split_train_test(hour_ret, factor=24)
hour_train <- split_hour_data$train
hour_test <- split_hour_data$test

split_min_data <- split_train_test(min_ret, factor=12*24)
min_train <- split_min_data$train
min_test <- split_min_data$test

try_hlags <- 4:12*2
try_mlags <- 1:1*2

best_mse <- 1e10
best_forecast <- rep(0, length(day_test))
best_hlag <- 0
best_mlag <- 0

for (hlag in try_hlags) {
  for (mlag in try_mlags) {
        midas_model = midas_r(
          y ~ trend + 
              #fmls(x1, hlag, 24) + 
              #fmls(x2, hlag, 24) + 
              fmls(x3, hlag, 24) +
              fmls(x4, hlag, 24) + 
              #fmls(x5, hlag, 24) + 
              #fmls(x6, hlag, 24) + 
              fmls(x7, hlag, 24),
          data = list(
            y=day_train$close, 
            trend=1:length(day_train$close), 
            #x1=hour_train$open, 
            #x2=hour_train$high, 
            x3=hour_train$low, 
            x4=hour_train$close,
            #x5=hour_train$volume,
            #x6=hour_train$volumeNotional,
            x7=hour_train$tradesDone
          ),
          start = c(0)
        )
        
        midas_forecast <- forecast(
          midas_model, 
          list(
            trend=(1460 + 1):length(day_ret$close), 
            #x1=hour_test$open,
            #x2=hour_test$high, 
            x3=hour_test$low, 
            x4=hour_test$close,
            #x5=hour_test$volume,
            #x6=hour_test$volumeNotional,
            x7=hour_test$tradesDone
          )
        )
  
    mse = (1/length(day_test$close)) * sum((day_test$close - midas_forecast$mean)^2)
    print(mse)
    
    if (mse < best_mse) {
      best_mse <- mse
      best_forecast <- midas_forecast$mean
      best_hlag <- hlag
      best_mlag <- mlag
      }
    }
}

print("Best performing lag: ")
print(best_hlag)
print(best_mlag)
print(best_mse)
print((1/length(day_test$close)) * sum((day_test$close - mean(day_train$close))^2))
p <- plot.new()
plot(day_test$close, type='l', col='black')
lines(best_forecast, type='l', col='red')

best_r <- -1
best_rforecast <- rep(0, length(day_test))
best_rhlag <- 0
best_rmlag <- 0

for (hlag in try_hlags) {
  for (mlag in try_mlags) {
    midas_model = midas_r(
      y ~ trend + 
        #fmls(x1, hlag, 24) + 
        #fmls(x2, hlag, 24) + 
        fmls(x3, hlag, 24) +
        fmls(x4, hlag, 24) + 
        #fmls(x5, hlag, 24) + 
        #fmls(x6, hlag, 24) + 
        fmls(x7, hlag, 24),
      data = list(
        y=day_train$close, 
        trend=1:length(day_train$close), 
        #x1=hour_train$open, 
        #x2=hour_train$high, 
        x3=hour_train$low, 
        x4=hour_train$close,
        #x5=hour_train$volume,
        #x6=hour_train$volumeNotional,
        x7=hour_train$tradesDone
      ),
      start = c(0)
    )
    
    midas_forecast <- forecast(
      midas_model, 
      list(
        trend=(1460 + 1):length(day_ret$close), 
        #x1=hour_test$open,
        #x2=hour_test$high, 
        x3=hour_test$low, 
        x4=hour_test$close,
        #x5=hour_test$volume,
        #x6=hour_test$volumeNotional,
        x7=hour_test$tradesDone
      )
    )
    
    r = calc_invest_return(midas_forecast$mean, day_test$close)
    print(r)
    
    if (r > best_r) {
      best_r <- r
      best_rforecast <- midas_forecast$mean
      best_rhlag <- hlag
      best_rmlag <- mlag
    }
  }
}

print("Best performing lag: ")
print(best_rhlag)
print(best_rmlag)
print(best_r)
print(day_test$close[length(day_test$close)] / day_test$close[1] - 1)

T <- length(day_test$close)
budget_mse <- 1
budget_r <- 1
holding <- 1
budget_mse_dev <- numeric(T)
budget_r_dev <- numeric(T)
holding_dev <- numeric(T)
prev_mse_strat = 1
prev_r_strat = 1
transaction_cost = 0

for (t in 1:T) {
  # Deal with MSE selected
  strategy_mse <- 1
  if (best_forecast[t] < 0) {
    strategy_mse <- -1
  }
  
  if (strategy_mse != prev_mse_strat) {
    budget_mse <- budget_mse * (1 - transaction_cost)
  }
  prev_mse_strat <- strategy_mse
  
  # Deal with r selected
  strategy_r <- 1
  if (best_rforecast[t] < 0) {
    strategy_r <- -1
  }
  
  if (strategy_r != prev_r_strat) {
    budget_r <- budget_r * (1 - transaction_cost)
  }
  prev_r_strat <- strategy_r
  
  
  budget_mse <- budget_mse * (1 + day_test$close[t]/100 * strategy_mse)
  budget_mse_dev[t] <- budget_mse
  budget_r <- budget_r * (1 + day_test$close[t]/100 * strategy_r)
  budget_r_dev[t] <- budget_r
  holding <- holding * (1 + day_test$close[t]/100)
  holding_dev[t] <- holding
}
p <- plot.new()
plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, budget_r_dev), max(budget_mse_dev, holding_dev, budget_r_dev)))
lines(budget_mse_dev, type='l', col='red')
lines(budget_r_dev, type='l', col='green')

calc_invest_return(best_forecast, day_test$close)
