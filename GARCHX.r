library(readr)
library(rugarch)

df <- read.csv('pct_btc_day.csv')
#df <- read.csv('pct_btc_hour.csv')

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

# Actually perform GARCH estimation
rdf <- df$close
Xdf <- as.matrix(df[c(-4)])
split <- 365
val <- 120

rtr <- rdf[1:(length(rdf)-split - val)]
Xtf <- df[c(-4)][1:(length(rdf)-split),]
rva <- rdf[(val+1):split]
rtf <- rdf[1:(length(rdf)-split)]
rts <- rdf[(length(rdf)-split+1):length(rdf)]

best_mse <- 1e10
best_p <- 0
best_o <- 0
best_q <- 0
best_a <- 0
best_b <- 0

try_p <- 1
try_o <- 1
try_q <- 1
try_a <- 3:4
try_b <- 3:4

estimate_garch_model <- function(yt, yv, p, o, q, a, b, summ=FALSE, type='gjrGARCH') {
  spec <- ugarchspec(
    variance.model = list(model = type, garchOrder = c(p, o, q)),
    mean.model = list(armaOrder = c(a, b), include.mean = TRUE, archm = TRUE, external.regressors=Xdf),
    distribution.model = "std"
  )
  
  model <- ugarchfit(
    data=yt,
    out.sample = length(yv),
    spec=spec,
    ex
  )
  
  if (summ) {print(model)}
  
  forc_results <- ugarchforecast(model, n.ahead=1, n.roll=length(yv)-1)
  forc <- as.numeric(fitted(forc_results))
  
  mse <- (1/length(yv)) * sum((yv - forc)^2)
  r <- calc_invest_return(forc, yv)
  
  return (list(r=r, mse=mse, forc=forc, fit=model, vol=as.numeric(forc_results@forecast$sigmaFor)))
}

estimate_rolling_garch_model <- function(yt, yv, p, o, q, a, b, summ=FALSE) {
  spec <- ugarchspec(
    variance.model = list(model = "gjrGARCH", garchOrder = c(p, o, q)),
    mean.model = list(armaOrder = c(a, b), include.mean = TRUE, archm = TRUE),
    distribution.model = "std"
  )
  
  roll <- ugarchroll(spec=spec, data=yt, forecast.length=365, refit.every=30)
  forc <- roll@forecast$density$Mu
  
  mse <- (1/length(yv)) * sum((yv - forc)^2)
  r <- calc_invest_return(forc, yv)
  
  return (list(r=r, mse=mse, forc=forc, fit=roll, fullf=roll@forecast))
}

if(FALSE) {
  for (p in try_p) {
    for (o in try_o) {
      for (q in try_q) {
        for (a in try_a) {
          for (b in try_b) {
            #result <- estimate_garch_model(rtf, rva, p, o, q, a, b)
            
            if (result$mse < best_mse) {
              best_mse <- result$mse
              best_p <- p
              best_o <- o
              best_q <- q
              best_a <- a
              best_b <- b
            }
            
            print(result$mse)
          }
        }
      }
    }
  }
}


# Compare against predicting mean
#fm <- estimate_garch_model(rdf, rts, best_p, best_o, best_q, best_a, best_b, summ=TRUE)
fm <- estimate_garch_model(rdf, rts, 1, 1, 1, 1, 0, summ=TRUE, type='gjrGARCH')
#rm <- estimate_rolling_garch_model(rdf, rts, 1, 1, 1, 1, 0, summ=TRUE)

print((1/split) * sum((rts - fm$forc)^2))
print((1/split) * sum((rts - mean(rtr))^2))

budget_mse_dev <- calc_invest_return(fm$forc, rts, use_threshold=FALSE)$path
holding_dev <- calc_invest_return(rep(1, length(rts)), rts, use_threshold=FALSE)$path
shorting_dev <- calc_invest_return(rep(-1, length(rts)), rts, use_threshold=FALSE)$path

plot(rts, type='l', col='black')
lines(fm$forc, type='l', col='red')
lines(fm$vol, type='l', col='blue')

plot(rtf, type='l', col='black')
lines(fm$fit@fit$fitted.values, type='l', col='red')
lines(fm$fit@fit$sigma, type='l', col='blue')

plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, shorting_dev), max(budget_mse_dev, holding_dev, shorting_dev)))
lines(budget_mse_dev, type='l', col='red')
lines(shorting_dev, type='l', col='blue')

write(fm$forc, file="final_R_forecasts/garchX_test.txt", ncolumns=1)
#write(fm$vol, file="final_R_forecasts/garchX_vol.txt", ncolumns=1)
#write(rm$fullf$density$Mu, file="final_R_forecasts/roll_garchX_test.txt", ncolumns=1)
#write(rm$fullf$density$Sigma, file="final_R_forecasts/roll_garchX_vol.txt", ncolumns=1)
#write(fm$fit@fit$fitted.values, file="final_R_forecasts/garchX_train.txt", ncolumns=1)
#write(fm$fit@fit$sigma, file="final_R_forecasts/garchX_train_vol.txt", ncolumns=1)
print("DONE")
