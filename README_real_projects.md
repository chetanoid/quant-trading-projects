# Real‑Data Quantitative Trading Projects

This folder contains three small projects that showcase practical skills
in quantitative trading and data analysis using *real* financial data.
Each project is self‑contained and heavily commented.  To run any
project, ensure you have Python 3.8+ and install the required
packages listed at the top of each script (typically `pandas`,
`scikit‑learn` and optionally `yfinance`).  You can install them via:

```bash
pip install pandas scikit‑learn yfinance
```

## 1 Real‑Data Market Making Simulator (`real_data_market_maker.py`)

This simulation uses a sequence of mid prices from actual markets to
drive a simple limit order book and market‑making strategy.  The
script first tries to download daily adjusted closes for the SPY ETF
using `yfinance`.  If the network or `yfinance` is unavailable, it
falls back to a small series of genuine S&P 500 levels from the early
twentieth century (sourced from the open *dow‑sp500‑100‑years*
repository).  The market maker quotes bid/ask prices around each
mid‑price, simulates random market order arrivals, matches orders and
marks its inventory to market.  At the end of the simulation it prints
the final P&L and inventory.  If `matplotlib` is installed, the script
also generates two diagnostic plots: one showing the mark‑to‑market P&L
trajectory and another showing how the market maker’s inventory evolves
over time.  These images are saved to `market_maker_pnl.png` and
`market_maker_inventory.png` in the current directory.  You can
adjust the spread, lot size and inventory target by modifying the
parameters passed to `simulate_market_with_real_data`.

### Running

```bash
python3 real_data_market_maker.py
```

You will see output like:

```
Simulation complete using 252 price points.
Final mark‑to‑market P&L: 12.34
Final inventory: 10
```

## 2 Real‑Data Trading Strategy Backtester (`real_data_strategy_backtest.py`)

This script fetches historical SPY prices (2015–2025 by default) via
`yfinance` and evaluates two simple strategies:

* **Momentum** – go long when the average return over the last 20
  days is positive; otherwise stay flat.
* **Mean‑reversion** – go long when the average return over the last
  20 days is negative; otherwise stay flat.

If data cannot be downloaded, the script uses a small sample of real
S&P 500 prices embedded as a CSV string.  The backtester calculates
daily strategy returns, computes cumulative return, annualised
volatility and a Sharpe‑like ratio, prints a summary to the console
and saves the time‑series of returns to `strategy_real_returns.csv`.
When `matplotlib` is available, it also plots cumulative returns for
the buy‑and‑hold, momentum and mean‑reversion strategies.  The chart
is saved as `strategy_cumulative_returns.png`.

### Running

```bash
python3 real_data_strategy_backtest.py
```

Example output:

```
Performance Summary (approx. annualised):
buy_and_hold     - Cumulative: 57.32%, Volatility: 16.24%, Sharpe: 1.21
momentum         - Cumulative: 32.10%, Volatility: 10.11%, Sharpe: 1.01
mean_reversion   - Cumulative: 15.55%, Volatility: 8.90%,  Sharpe: 0.55
Daily returns saved to strategy_real_returns.csv
```

## 3 Real‑Data Sentiment Analysis (`real_data_sentiment_analysis.py`)

This project uses the **Financial News Sentiment** dataset from the
open‐source repository [`sentiment-analysis-for-financial-news`](https://github.com/isaaccs/sentiment-analysis-for-financial-news).
The file `all-data.csv` (bundled in this repo) contains ~6k
sentences labelled as *positive*, *neutral* or *negative*.  The
script performs the following steps:

1. Load the dataset and map textual labels to integers (1, 0 and –1).
2. Convert the text into a TF‑IDF matrix of token frequencies.
3. Split into training and test sets.
4. Train a logistic regression classifier and a random forest classifier.
5. Print accuracy and a full classification report for each model.
6. If `matplotlib` is installed, generate confusion matrix heatmaps for
   the logistic regression and random forest classifiers and save them
   as `logistic_confusion_matrix.png` and
   `random_forest_confusion_matrix.png`.

You can adjust the `max_features` parameter of `TfidfVectorizer` or
test other models (e.g. SVM, XGBoost) to experiment with
performance.

### Running

```bash
python3 real_data_sentiment_analysis.py
```

Example output:

```
Logistic Regression Accuracy: 74.85%
Classification Report (Logistic Regression):
              precision    recall  f1-score   support

    negative       0.76      0.71      0.74       400
     neutral       0.70      0.72      0.71       401
    positive       0.78      0.81      0.79       401

Random Forest Accuracy: 72.10%
Classification Report (Random Forest):
              precision    recall  f1-score   support

    negative       0.71      0.68      0.69       400
     neutral       0.68      0.70      0.69       401
    positive       0.75      0.78      0.76       401
```

## Notes & Next Steps

* The embedded price sample is deliberately small to keep the repository
  lightweight.  For deeper analysis, replace it with more recent or
  longer time series downloaded via `yfinance` or another data source.
* The order book simulator uses simplified assumptions (no order
  cancellations, constant lot size and spread).  Extending it with
  dynamic spread management, cancellation queues or real intraday
  order book snapshots would provide richer behaviour.
* The sentiment model can be improved by adding domain‑specific
  pre‑processing, n‑grams, or using transformer models like BERT via
  `transformers`.

We hope these examples help you build a compelling portfolio for
quant trading or research roles.  Feel free to fork and modify the
projects to suit your interests.

## 4 Portfolio Optimisation (`portfolio_optimization.py`)

This script downloads daily adjusted close prices for a handful of
liquid tickers (default: AAPL, MSFT, GOOGL, AMZN and SPY) and
computes daily returns.  It then generates thousands of random
portfolios to explore the risk–return trade‑off.  For each
portfolio it calculates expected annualised return, annualised
volatility and the Sharpe ratio.  The portfolios with the highest
Sharpe ratio and the lowest volatility are highlighted, and a
scatter plot of all portfolios is saved to
`portfolio_optimisation.png`.  All portfolio statistics are also
written to `portfolio_optimisation.csv`.  A small synthetic price
dataset is embedded as a fallback if live data cannot be fetched.

### Running

```bash
python3 portfolio_optimization.py
```

Example output:

```
Maximum Sharpe Ratio Portfolio:
return         0.31
volatility     0.18
sharpe         1.60
Name: 1234, dtype: float64
Weights:
{'AAPL': 0.34, 'MSFT': 0.25, 'GOOGL': 0.18, 'AMZN': 0.12, 'SPY': 0.11}

Minimum Volatility Portfolio:
return         0.20
volatility     0.15
sharpe         1.20
Name: 567, dtype: float64
Weights:
{'AAPL': 0.10, 'MSFT': 0.15, 'GOOGL': 0.25, 'AMZN': 0.30, 'SPY': 0.20}
Portfolio statistics saved to portfolio_optimisation.csv
Efficient frontier plot saved to portfolio_optimisation.png
```

## 5 Monte Carlo Option Pricing (`monte_carlo_option_pricing.py`)

This project prices a European call option using Monte Carlo
simulation.  It can estimate volatility from recent SPY returns via
`yfinance` or use a default volatility if that fails.  The script
simulates geometric Brownian motion price paths, computes the
discounted payoff of the option, and outputs an estimated price with
its standard error.  If `matplotlib` is installed, it also saves
plots of sample price paths and the distribution of discounted
payoffs to PNG files (`option_price_paths.png` and
`option_payoff_distribution.png`).

### Running

```bash
python3 monte_carlo_option_pricing.py
```

Example output:

```
Estimated call option price: 10.4321 ± 0.1234 (standard error)
Price path and payoff distribution plots saved to option_price_paths.png and option_payoff_distribution.png

## 6 Cointegration Pairs Trading (`cointegration_pairs_trading.py`)

This project implements a simple statistical arbitrage strategy based on the
concept of cointegration. It does the following:

* **Data acquisition:** Downloads daily adjusted close prices for two tickers (default: KO and PEP) via
  `yfinance`. If the library is unavailable or data cannot be fetched,
  it generates two correlated random walk series as a fallback.
* **Cointegration test:** Uses the Engle–Granger two‑step test to check whether the two price series
  are cointegrated and estimates the hedge ratio by regressing one series on
  the other.
* **Spread construction:** Constructs the spread and computes its z‑score. When the z‑score deviates
  beyond a specified threshold (default ±1.0), the strategy opens a mean‑reversion
  trade (long one asset and short the other) and exits when the spread reverts.
* **Backtesting:** Simulates the strategy to produce an equity curve, prints the final equity
  and total return, and saves a plot of the z‑score with trading bands as well
  as the equity curve to `pairs_trading_results.png`.

### Running

```bash
python3 cointegration_pairs_trading.py --symbols KO PEP --start 2018-01-01 --end 2024-01-01 --threshold 1.5
```

Example output:

```
Downloaded 1500 rows of price data for ['KO', 'PEP'].
Cointegration test statistic: -3.45, p-value: 0.02
Hedge ratio (beta): 0.85
Final equity: 1.12 (Total return: 12.00%)
Plots saved to 'pairs_trading_results.png'.
```

## 7 Value at Risk & CVaR Simulation (`value_at_risk_simulation.py`)

This script demonstrates how to compute Value at Risk (VaR) and Conditional
Value at Risk (CVaR or Expected Shortfall) for a portfolio of equities using
both parametric (Gaussian) and historical methods. The key steps are:

* **Data acquisition:** Download daily adjusted close prices for a set of tickers (default: AAPL,
  MSFT, GOOGL and AMZN) via `yfinance`. If the download fails, it
  generates synthetic returns and constructs a price series from them as a
  fallback.
* **Return calculation:** Compute daily portfolio returns using equal weighting (weights can be
  customised). Calculate the mean and standard deviation of returns.
* **VaR and CVaR computation:** For the parametric method, assume returns follow a normal distribution and
  compute VaR and CVaR using the appropriate quantile and normal density.
  For the historical method, sort the empirical returns and compute VaR as
  the percentile and CVaR as the average of the worst returns.
* **Visualisation:** Plot the distribution of portfolio returns with vertical lines indicating
  the VaR and CVaR thresholds; the plot is saved to
  `portfolio_var_distribution.png`.

### Running

```bash
python3 value_at_risk_simulation.py --symbols AAPL MSFT GOOGL AMZN --start 2018-01-01 --end 2023-12-31 --confidence 0.99 --method parametric
```

Example output:

```
Downloaded 1500 rows of price data for ['AAPL', 'MSFT', 'GOOGL', 'AMZN'].
Parametric VaR (99%): 2.13%
Parametric CVaR: 2.75%
Plot saved to 'portfolio_var_distribution.png'.

## 8 Risk Parity Portfolio (`risk_parity_portfolio.py`)

Risk parity aims to allocate capital such that each asset contributes an
equal share of total portfolio risk. This script constructs a
risk‑parity portfolio from a basket of equities and compares its
performance to a simple equal‑weighted portfolio.

Key steps include:

* **Data acquisition:** Download daily adjusted prices for a set of
  tickers (defaults: AAPL, MSFT, GOOGL, AMZN) via `yfinance`.
  If the download fails, the script generates synthetic correlated
  returns and builds a price series as a fallback.
* **Return and covariance calculation:** Compute log returns and the
  covariance matrix.
* **Risk parity optimisation:** Use an iterative multiplicative update
  method to find weights that equalise each asset’s risk
  contribution. This method converges quickly and ensures the
  weights sum to 1.
* **Performance comparison:** Calculate annualised return,
  volatility and Sharpe ratio for both the risk parity portfolio and
  an equal‑weight portfolio, and display each asset’s weight and risk
  contribution.
* **Visualisation:** Plot a bar chart of risk parity weights and
  save it to `risk_parity_weights.png`.

### Running

```bash
python3 risk_parity_portfolio.py --symbols AAPL MSFT GOOGL AMZN --start 2018-01-01 --end 2024-01-01
```

Example output:

```
Risk Parity Weights:
  AAPL: weight=0.446, risk contribution=0.007
  MSFT: weight=0.000, risk contribution=0.000
  GOOGL: weight=0.000, risk contribution=0.000
  AMZN: weight=0.554, risk contribution=0.007

Risk Parity Portfolio: Return=3.06%, Volatility=22.55%, Sharpe=0.14
Equal Weight Portfolio: Return=2.59%, Volatility=16.56%, Sharpe=0.16
Plot saved to 'risk_parity_weights.png'.
```

## 9 Option Pricing & Greeks Calculator (`option_greeks_calculator.py`)

Understanding how option prices respond to changes in market
conditions is essential for hedging and risk management. This script
computes the Black–Scholes price for European options and their
sensitivity measures ("Greeks") such as Delta, Gamma, Vega, Theta and
Rho.

Key features:

* **User‑specified inputs:** Specify whether the option is a call or
  put, the underlying spot price, strike price, risk‑free rate,
  volatility and time to maturity. Default values are provided.
* **Analytical Greeks:** Compute the option price and all five major
  Greeks using closed‑form Black–Scholes expressions. Option price,
  Delta, Gamma, Vega, Theta and Rho are printed to the console.
* **Visualisation:** Generate a figure showing how each Greek varies
  with the underlying price (from 50% to 150% of the current spot).
  The plot is saved to `option_greeks.png`.

### Running

```bash
python3 option_greeks_calculator.py --type call --spot 100 --strike 100 --rate 0.02 --vol 0.2 --t 1
```

Example output:

```
Call Option Price: 8.9160
Delta: 0.5793
Gamma: 0.019552
Vega: 39.1043
Theta: -6.0491
Rho: 49.0099
Greek plots saved to 'option_greeks.png'.

## 10 Kalman Filter Pairs Trading (`kalman_pairs_trading.py`)

This project extends the pairs trading framework by allowing the
hedge ratio between two assets to vary over time. A Kalman filter
estimates the time‑varying beta that relates the price of a
dependent asset to that of an independent asset. Trading signals are
generated based on the z‑score of the spread (prediction residual).

Main features:

* **Adaptive hedge ratio:** Use a state‑space model where the hedge
  ratio follows a random walk and is updated each day via a Kalman
  filter. This captures changing market relationships more
  effectively than a static regression.
* **Signal logic:** Compute the spread and its rolling z‑score. Enter a
  short position when the z‑score exceeds the entry threshold and a
  long position when it drops below the negative entry threshold.
  Exit positions when the spread reverts within the exit band. The
  position sizes maintain dollar neutrality using the current
  estimated hedge ratio.
* **Fallback data:** If `yfinance` cannot fetch prices for the chosen
  tickers, the script generates synthetic correlated prices for
  demonstration. This ensures the code runs offline while still
  illustrating the algorithm.
* **Performance tracking:** Calculate cumulative equity, total return
  and number of trades. Plot the evolving hedge ratio, z‑score with
  entry/exit thresholds and the equity curve in a single figure
  saved to `kalman_pairs_trading_results.png`.

### Running

```bash
python3 kalman_pairs_trading.py --y_ticker SPY --x_ticker QQQ \
    --start 2015-01-01 --end 2024-01-01 --entry 2.0 --exit 0.5
```

Example output:

```
Downloaded 1600 rows of price data for ['SPY', 'QQQ'].
Final return: 5.13% with 18 trades.
Results plot saved to 'kalman_pairs_trading_results.png'.
```

## 11 GARCH Volatility Modelling and VaR (`garch_volatility_model.py`)

This project demonstrates how to estimate time‑varying volatility using a
GARCH(1,1) model and how to forecast risk measures like Value at Risk (VaR).
The script fetches daily prices for a single asset (by default SPY),
computes log returns and fits a GARCH model to the return series. If the
network or `yfinance` is unavailable it uses a small sample of genuine
historical closing prices as a fallback.  The optimisation routine
estimates the parameters *(ω, α, β)* by maximising the log‑likelihood,
ensuring the process is stationary (α + β < 1).  Once the model is
fitted, it produces a series of conditional variances and forecasts
future volatility for the next *n* days.  Using the predicted variance
it calculates one‑day 95 percent VaR (assuming normally distributed
returns) and prints the results.

Key features:

* **Parameter estimation:** Use maximum likelihood estimation via
  `scipy.optimize.minimize` to estimate the GARCH(1,1) parameters, with
  parameter bounds and stationarity constraints.
* **Volatility forecasting:** Compute the conditional variance series
  and forecast volatility for a user‑specified number of days. Print
  the predicted annualised volatility for each forecast horizon.
* **Risk measurement:** Calculate the one‑day 95 percent VaR using
  the forecasted volatility under the normality assumption. The script
  prints the VaR alongside the estimated parameters and average
  historical volatility.
* **Visualisation:** Generate a plot showing the historical conditional
  variance and forecast horizon, saved to `garch_volatility_model.png`.

### Running

```bash
python3 garch_volatility_model.py --ticker SPY --start 2010-01-01 --end 2024-01-01 --forecast_days 20
```

Example output:

```
Estimated parameters: omega=3.42e-06, alpha=0.1000, beta=0.8000
Average annualised volatility: 12.05%
Forecast next 20 days volatility:
  Day 1: 9.41% annualised
  Day 2: 9.40% annualised
  ...
  Day 20: 9.30% annualised
One-day 95% VaR (assuming normal returns): 1.16%
Plot saved to 'garch_volatility_model.png'.
```

```