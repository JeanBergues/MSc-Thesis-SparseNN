library(readr)
library(forecast)

day_data <- read.csv('pct_btc_day.csv')

split_train_test <- function (data_df, factor) {
  n <- length(data_df$open)
  print(n)
  split_val <- n - (365 + 120) * factor
  split_t <- n - 365 * factor - 1
  train <- data.frame(index = 1:split_val)
  trainf <- data.frame(index = 1:split_t)
  val <- data.frame(index = (split_val + 1):split_t)
  test <- data.frame(index = (split_t+1):n)
  
  for (col in colnames(data_df)[1:length(colnames(data_df))]) {
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

split_day_data <- split_train_test(day_data, factor=1)
day_train <- split_day_data$train
day_trainf <- split_day_data$trainf
day_val <- split_day_data$val
day_test <- split_day_data$test

### Estimate the ARIMA model
day_trainX <- as.matrix(day_trainf[c(-1, -5)])[1:(length(day_trainf$close) - 1),]
day_testX <- as.matrix(day_test[c(-1, -5)])[1:(length(day_test$close) - 1),]

ytrain <- day_trainf$close[(1 + 1):length(day_trainf$close)]
ytest <- day_test$close[(1 + 1):length(day_test$close)]

am <- auto.arima(ytrain)
print(summary(am))

forecast <- fitted(Arima(ytest, model=am))
write(forecast, file="final_R_forecasts/arima_day_test.txt", ncolumns=1)
forecast <- fitted(Arima(ytrain, model=am))
write(forecast, file="final_R_forecasts/arima_day_train.txt", ncolumns=1)

am <- auto.arima(ytrain, xreg=day_trainX)
print(summary(am))

forecast <- fitted(Arima(ytest, model=am, xreg=day_testX))
write(forecast, file="final_R_forecasts/arimaX_day_test.txt", ncolumns=1)
forecast <- fitted(Arima(ytrain, model=am, xreg=day_trainX))
write(forecast, file="final_R_forecasts/arimaX_day_train.txt", ncolumns=1)
