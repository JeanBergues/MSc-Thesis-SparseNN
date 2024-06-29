library(readr)
library(rugarch)

df <- read.csv('agg_btc_day.csv')
#hour_data <- read.csv('agg_btc_hour.csv')
#min_data <- read.csv('agg_btc_min.csv')

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
rdf <- 100 * diff(df$close)/df$close[-length(df$close)]
split <- 365
val <- 100

rtr <- rdf[1:(length(rdf)-split - val)]
rva <- rdf[(val+1):split]
rtf <- rdf[1:split]
rts <- rdf[(length(rdf)-split+1):length(rdf)]

best_mse <- 1e10
best_p <- 0
best_o <- 0
best_q <- 0
best_a <- 0
best_b <- 0

try_p <- 1:3
try_o <- 0:2
try_q <- 0:3
try_a <- 0:0
try_b <- 0:0

estimate_garch_model <- function(yt, yv, p, o, q, a, b, summ=FALSE) {
  spec <- ugarchspec(
    variance.model = list(model = "gjrGARCH", garchOrder = c(p, o, q)),
    mean.model = list(armaOrder = c(a, b), include.mean = TRUE, archm = TRUE),
    distribution.model = "std"
  )
  
  model <- ugarchfit(
    data=yt,
    out.sample = length(yv),
    spec=spec
  )
  
  if (summ) {print(summary(model))}
  
  forc_results <- ugarchforecast(model, n.ahead=length(yv))
  forc <- as.numeric(fitted(forc_results))
  
  mse <- (1/length(yv)) * sum((yv - forc)^2)
  r <- calc_invest_return(forc, yv)
  
  return (list(r=r, mse=mse, forc=forc))
}

for (p in try_p) {
  for (o in try_o) {
    for (q in try_q) {
      for (a in try_a) {
        for (b in try_b) {
          result <- estimate_garch_model(rtf, rva, p, o, q, a, b)
          
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


# Compare against predicting mean
fm <- estimate_garch_model(rdf, rts, best_p, best_o, best_q, best_a, best_b, summ=TRUE)

print((1/split) * sum((rts - fm$forc)^2))
print((1/split) * sum((rts - mean(rtr))^2))

budget_mse_dev <- calc_invest_return(fm$forc, rts)$path
holding_dev <- calc_invest_return(rep(1, length(rts)), rts)$path
shorting_dev <- calc_invest_return(rep(-1, length(rts)), rts)$path

plot(rts, type='l', col='black')
lines(fm$forc, type='l', col='red')

plot(holding_dev, type='l', col='black', ylim=c(min(budget_mse_dev, holding_dev, shorting_dev), max(budget_mse_dev, holding_dev, shorting_dev)))
lines(budget_mse_dev, type='l', col='red')
print("DONE")
