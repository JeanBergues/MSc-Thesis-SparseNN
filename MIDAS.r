library(readr)
library(midasr)

day_data <- read.csv('pct_btc_day.csv')
hour_data <- read.csv('pct_btc_hour.csv')

split_train_test <- function (data_df, factor) {
  n <- length(data_df$open)
  split_val <- n - (366 + 120) * factor
  split_t <- n - 366 * factor
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

calc_invest_return <- function(frc, real, start_v = 1, t_cost=0.0010, use_threshold=TRUE) {
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

day_ret <- day_data
hour_ret <- hour_data
#min_ret <- calc_pct_diffs(min_data, start_col=3)

split_day_data <- split_train_test(day_ret, factor=1)
day_train <- split_day_data$train
day_trainf <- split_day_data$trainf
day_val <- split_day_data$val
day_test <- split_day_data$test

split_hour_data <- split_train_test(hour_ret, factor=24)
hour_train <- split_hour_data$train
hour_trainf <- split_hour_data$trainf
hour_val <- split_hour_data$val
hour_test <- split_hour_data$test

#split_min_data <- split_train_test(min_ret, factor=12*24)
#min_train <- split_min_data$train
#min_test <- split_min_data$test

estimate_midas_model <- function(Xt_day, Xv_day, Xt_hour, Xv_hour, hlag=1, mlag=1, summ=FALSE, freq=24) {
  midas_model = midas_r(
    y ~ 
      trend
      + y1
      #+ fmls(x1, hlag, freq, nealmon) 
      #+ fmls(x2, hlag, freq, nealmon) 
      #+ fmls(x3, hlag, freq, nealmon)
      + fmls(x4, hlag, freq, nealmon) 
      #+ fmls(x5, hlag, freq, nealmon) 
      #+ fmls(x6, hlag, freq, nealmon) 
      #+ fmls(x7, hlag, freq, nealmon)
      ,
    data = list(
      y=Xt_day$close[2:length(Xt_day$close)] 
      ,trend=1:(length(Xt_day$close) - 1)
      ,y1=Xt_day$close[1:(length(Xt_day$close) - 1)]
      ,x1=Xt_hour$open[1:(length(Xt_hour$open) - freq)] 
      ,x2=Xt_hour$high[1:(length(Xt_hour$open) - freq)]
      ,x3=Xt_hour$low[1:(length(Xt_hour$open) - freq)] 
      ,x4=Xt_hour$close[1:(length(Xt_hour$open) - freq)]
      ,x5=Xt_hour$volume[1:(length(Xt_hour$open) - freq)]
      ,x6=Xt_hour$volumeNotional[1:(length(Xt_hour$open) - freq)]
      ,x7=Xt_hour$tradesDone[1:(length(Xt_hour$open) - freq)]
    ),
    start = list(
      #x1=rep(0, 3), 
      #x2=rep(0, 3), 
      #x3=rep(0, 3),
      x4=rep(0, 3)
      #x5=rep(0, 3),
      #x6=rep(0, 3),
      #x7=rep(0, 3)
    )
  )
  
  midas_forecast <- forecast(
    midas_model, 
    list(
      trend=(length(Xt_day$close) + 1):(length(Xt_day$close) + length(Xv_day$close)-1)
      ,y1=Xv_day$close[1:(length(Xv_day$close)-1)]
      ,x1=Xv_hour$open[1:(length(Xv_hour$close) - freq)]
      ,x2=Xv_hour$high[1:(length(Xv_hour$close) - freq)]
      ,x3=Xv_hour$low[1:(length(Xv_hour$close) - freq)]
      ,x4=Xv_hour$close[1:(length(Xv_hour$close) - freq)]
      ,x5=Xv_hour$volume[1:(length(Xv_hour$close) - freq)]
      ,x6=Xv_hour$volumeNotional[1:(length(Xv_hour$close) - freq)]
      ,x7=Xv_hour$tradesDone[1:(length(Xv_hour$close) - freq)]
    )
  )
  
  ytest <- Xv_day$close[2:length(Xv_day$close)]
  mse <- (1/length(ytest)) * sum((ytest - midas_forecast$mean)^2)
  rt <- calc_invest_return(midas_forecast$mean, ytest)$ret
  comb <- midas_forecast$mean * ytest
  co <- length(comb[comb >= 0]) / length(ytest)
  
  if (summ) {
    print(summary(midas_model))
  }
  
  return (list(mse=mse, r=rt, frc=midas_forecast$mean, co=co, model=midas_model))
}

try_hlags <- 1:24
try_mlags <- 0

best_mse <- 1e10
best_r <- -100

best_mse_hlag <- 0
best_mse_mlag <- 0
best_r_hlag <- 0
best_r_mlag <- 0

for (hlag in try_hlags) {
  for (mlag in try_mlags) {
    total_mse <- 0
    total_r <- 0
    
    result <- estimate_midas_model(day_train, day_val, hour_train, hour_val, hlag=hlag, mlag=mlag)
    total_mse <- result$mse
    total_r <- result$r
    print(total_mse)
    print(total_r)
    #print("")
    
    if (total_mse < best_mse) {
      best_mse <- total_mse
      best_mse_hlag <- hlag
      best_mse_mlag <- mlag
    }
    if (total_r > best_r) {
      best_r <- total_r
      best_r_hlag <- hlag
      best_r_mlag <- mlag
    }
  }
}

mean_mse <- (1/length(day_test$close)) * sum((day_test$close - mean(day_train$close))^2)
mean_r <- calc_invest_return(rep(1, length(day_test$close)), day_test$close, use_threshold = FALSE)

full_mse <- estimate_midas_model(day_trainf, day_test, hour_trainf, hour_test, hlag=best_mse_hlag, mlag=best_mse_mlag, summ=FALSE)
full_ret <- estimate_midas_model(day_trainf, day_test, hour_trainf, hour_test, hlag=best_r_hlag, mlag=best_r_mlag, summ=FALSE)
print(full_ret$co)

write(full_mse$frc, file="final_R_forecasts/MIDAS_test.txt")
#write(full_ret$frc, file="txt_forecast/amidas_day_r_test.txt")

print(summary(full_mse$model))

ytest <- day_test$close[2:(length(day_test$close) - 1)]
budget_mse_dev <- calc_invest_return(full_mse$frc, ytest, use_threshold = FALSE)$path
budget_r_dev <- calc_invest_return(full_ret$frc, ytest, use_threshold = FALSE)$path
holding_dev <- mean_r$path
shorting_dev <- calc_invest_return(rep(-1, length(ytest)), ytest, use_threshold = FALSE)$path

plot(day_test$close, type='l', col='black')
lines(full_mse$frc, type='l', col='red')
lines(full_ret$frc, type='l', col='green')

plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, budget_r_dev, shorting_dev), max(budget_mse_dev, holding_dev, budget_r_dev, shorting_dev)))
lines(shorting_dev, type='l', col='blue')
lines(budget_mse_dev, type='l', col='red')
lines(budget_r_dev, type='l', col='green')
