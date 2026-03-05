# Quantitative Trading Project Portfolio

This repository contains three standalone Python scripts that demonstrate fundamental concepts in quantitative trading and finance.  Each project focuses on a different aspect of the trading ecosystem and is designed for educational purposes.  All scripts are self‑contained and use synthetic data so that they can be run offline.

## 1. Limit Order Book & Market Making Simulator (`limit_order_book_simulator.py`)

This script implements a simple limit order book (LOB) with a market‑making strategy.

### Features

* **Event‑Driven Matching** — Incoming limit and market orders are matched based on price priority using heaps to maintain bid and ask queues.
* **Inventory‑Based Quote Logic** — The market maker posts bid/ask quotes around a mid price and nudges quotes up or down to manage inventory.
* **Geometric Brownian Motion Price Simulation** — The underlying asset price evolves via a random walk with configurable volatility.
* **P&L Calculation** — At each timestep the market maker’s cash, inventory, and mark‑to‑market P&L are updated and written to a text file.

### How to run

Execute the script with Python to simulate 200 timesteps with a base price of 100.0:

```bash
python3 limit_order_book_simulator.py
```

Upon completion the script prints summary statistics and writes a `limit_order_book_results.txt` file with the price and P&L history.  You can adjust the number of timesteps, volatility, or lot sizes by editing the parameters in the `simulate_market` function call in the script.

## 2. Trading Strategies Research (`trading_strategies_research.py`)

This script generates synthetic price data and backtests two simple strategies: a momentum strategy and a mean‑reversion strategy.

### Features

* **Synthetic Data Generation** — Prices follow a geometric Brownian motion with drift and volatility parameters.
* **Momentum Strategy** — Goes long (short) when the current price is above (below) its value 10 periods ago.
* **Mean‑Reversion Strategy** — Goes long (short) when the price is below (above) a 20‑period moving average.
* **Performance Metrics** — Computes cumulative return, annualised volatility, a simplified Sharpe ratio, and maximum drawdown for each strategy versus a buy‑and‑hold benchmark.
* **CSV Output** — Saves the returns from each strategy and the benchmark to `strategy_returns.csv` for further analysis.

### How to run

Run the script with Python:

```bash
python3 trading_strategies_research.py
```

The script prints a performance summary and writes the returns series to a CSV file in the current directory.  You can modify the length of the time series or the lookback windows by changing the arguments in the `run_backtest` function.

## 3. Sentiment Analysis & Market Direction Prediction (`sentiment_analysis_market_prediction.py`)

This project demonstrates a basic NLP pipeline for classifying sentiment in short text snippets and predicting whether news or social media posts might indicate a positive or negative outlook for markets.

### Features

* **Synthetic Text Dataset** — A small list of market‑related headlines and social media‑style comments labelled as positive or negative.
* **TF‑IDF Vectorisation** — Converts raw text into numerical features using term frequency–inverse document frequency.
* **Classification Models** — Trains both a logistic regression model and a random forest classifier on the dataset.
* **Model Evaluation** — Reports accuracy on a test split and prints predictions for a handful of unseen example headlines.

### How to run

Run the script with Python:

```bash
python3 sentiment_analysis_market_prediction.py
```

You’ll see the accuracy of each model and the predicted sentiment for several new headlines.  To experiment with the pipeline, edit the list of sentences in `create_dataset` or try adding your own examples in the `test_examples` list.

## Notes

These projects are meant to illustrate methodologies rather than produce profitable trading strategies.  They are deliberately simplified and rely on synthetic data to keep the focus on the algorithms and analysis.  For real‑world applications, you would replace the synthetic components with actual market data feeds, use more sophisticated modelling techniques, and implement rigorous validation and risk management.