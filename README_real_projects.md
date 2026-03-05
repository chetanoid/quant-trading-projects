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
marks its inventory to market.  At the end of the simulation it
prints the final P&L and inventory.  You can adjust the spread,
lot size and inventory target by modifying the parameters passed to
`simulate_market_with_real_data`.

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