"""
Trading Strategies Research
==========================

This script demonstrates the construction and backtesting of two simple algorithmic trading strategies
on synthetic time series data: a momentum strategy and a mean‑reversion strategy.  Because external
data access is unavailable in this environment, we generate synthetic price data via a random walk
with drift.  The script computes trades, cumulative returns, drawdowns, and a rudimentary Sharpe ratio
for each strategy and compares their performance to a long‑only buy‑and‑hold benchmark.

The goal is to illustrate how to structure a basic backtesting loop, calculate common performance
metrics, and contrast the behaviours of trend‑following and mean‑reversion approaches.  The code
is self‑contained and can be easily adapted to real market data when available.

Author: OpenAI assistant
"""

import math
import numpy as np
import pandas as pd


def generate_synthetic_prices(n: int = 1000, start_price: float = 100.0, mu: float = 0.0002,
                              sigma: float = 0.01) -> pd.Series:
    """Simulate a geometric Brownian motion price series."""
    dt = 1.0
    prices = [start_price]
    for _ in range(n - 1):
        # dS = mu * S * dt + sigma * S * dW
        prices.append(prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.random.randn() * math.sqrt(dt)))
    return pd.Series(prices, name="Price")


def momentum_strategy(prices: pd.Series, lookback: int = 10) -> pd.DataFrame:
    """Momentum strategy: long when price has increased over lookback period, short when decreased."""
    returns = prices.pct_change()
    signal = (prices > prices.shift(lookback)).astype(int) * 2 - 1  # 1 if momentum up, -1 if down
    strategy_returns = signal.shift(1) * returns  # shift signal to avoid lookahead bias
    return pd.DataFrame({"price": prices, "returns": returns, "signal": signal, "strategy_returns": strategy_returns})


def mean_reversion_strategy(prices: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Mean‑reversion strategy: short when price is above moving average, long when below."""
    returns = prices.pct_change()
    ma = prices.rolling(lookback).mean()
    signal = (prices < ma).astype(int) * 2 - 1  # long when below MA (price expected to revert up), short when above
    strategy_returns = signal.shift(1) * returns
    return pd.DataFrame({"price": prices, "returns": returns, "ma": ma, "signal": signal, "strategy_returns": strategy_returns})


def performance_metrics(strategy_returns: pd.Series) -> dict:
    """Compute cumulative return, volatility, and Sharpe ratio for a strategy."""
    cumulative_return = (1 + strategy_returns.fillna(0)).prod() - 1
    volatility = strategy_returns.std() * np.sqrt(len(strategy_returns))
    sharpe = cumulative_return / volatility if volatility != 0 else 0.0
    drawdown = (strategy_returns.fillna(0).cumsum() - strategy_returns.fillna(0).cumsum().cummax()).min()
    return {
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def run_backtest(n=1000):
    # Generate synthetic price data
    prices = generate_synthetic_prices(n=n, start_price=100.0, mu=0.0002, sigma=0.01)

    # Buy‑and‑hold returns
    bh_returns = prices.pct_change().fillna(0)

    # Momentum strategy
    mom_df = momentum_strategy(prices, lookback=10)
    mom_metrics = performance_metrics(mom_df["strategy_returns"].fillna(0))

    # Mean‑reversion strategy
    mr_df = mean_reversion_strategy(prices, lookback=20)
    mr_metrics = performance_metrics(mr_df["strategy_returns"].fillna(0))

    # Benchmark metrics
    benchmark_metrics = performance_metrics(bh_returns)

    # Print summary
    print("Backtest Summary (synthetic data)")
    print("--------------------------------")
    print(f"Benchmark (Buy & Hold): Cumulative Return = {benchmark_metrics['cumulative_return']:.2%}, "
          f"Volatility = {benchmark_metrics['volatility']:.2%}, Sharpe ≈ {benchmark_metrics['sharpe']:.2f}")
    print(f"Momentum Strategy:      Cumulative Return = {mom_metrics['cumulative_return']:.2%}, "
          f"Volatility = {mom_metrics['volatility']:.2%}, Sharpe ≈ {mom_metrics['sharpe']:.2f}, "
          f"Max Drawdown = {mom_metrics['max_drawdown']:.2%}")
    print(f"Mean‑Reversion Strategy: Cumulative Return = {mr_metrics['cumulative_return']:.2%}, "
          f"Volatility = {mr_metrics['volatility']:.2%}, Sharpe ≈ {mr_metrics['sharpe']:.2f}, "
          f"Max Drawdown = {mr_metrics['max_drawdown']:.2%}")

    # Optionally save results to CSV
    results = pd.DataFrame({
        "date": range(len(prices)),
        "price": prices,
        "bh_returns": bh_returns,
        "momentum_returns": mom_df["strategy_returns"],
        "mean_reversion_returns": mr_df["strategy_returns"],
    })
    results.to_csv("strategy_returns.csv", index=False)


if __name__ == "__main__":
    run_backtest(n=1000)