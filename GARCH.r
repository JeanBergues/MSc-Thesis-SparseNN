library(readr)
library(rugarch)

df <- read.csv('agg_btc_min.csv')
#hour_data <- read.csv('agg_btc_hour.csv')
#min_data <- read.csv('agg_btc_min.csv')

# Actually perform GARCH estimation
rdf <- 100 * diff(df$close)/df$close[-length(df$close)]
split <- floor(0.2 * length(rdf))
rtr <- rdf[1:split]
rts <- rdf[(length(rdf)-split+1):length(rdf)]

spec <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE, archm = TRUE),
  distribution.model = "std"
)

model <- ugarchfit(
  data=rdf,
  out.sample = split,
  spec=spec
)

forc_results <- ugarchforecast(model, n.ahead=split)
forc <- as.numeric(fitted(forc_results))

# Compare against predicting mean
p <- plot.new()
plot(rts, type='l', col='black')
lines(forc, type='l', col='red')
print((1/split) * sum((rts - forc)^2))
print((1/split) * sum((rts - mean(rtr))^2))


# Simulate investments
budget <- 10000
holding <- 10000
budget_dev <- numeric(split)
holding_dev <- numeric(split)
prev_strat = 1
transaction_cost = 0

for (t in 1:split) {
  strategy <- 1
  if (forc[t] < 0) {
    strategy <- -1
  }
  
  if (strategy != prev_strat) {
    budget <- budget * (1 - transaction_cost)
  }
  prev_strat <- strategy
  budget <- budget * (1 + rts[t]/100 * strategy)
  budget_dev[t] <- budget
  holding <- holding * (1 + rts[t]/100)
  holding_dev[t] <- holding
}
p <- plot.new()
plot(holding_dev, type='l', col='black', ylim=c(min(budget_dev, holding_dev), max(budget_dev, holding_dev)))
lines(budget_dev, type='l', col='red')
