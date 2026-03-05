"""
Backtesting Engine
===================

This script implements a simple yet powerful backtesting engine for evaluating
trading strategies on historical price data. It supports multiple strategies,
computes common performance metrics and produces equity curve plots. The goal
is to provide a modular framework that can be extended with custom
strategies, multiple assets or alternative risk management rules.

Features
--------
* Fetches daily price data for one or more tickers using ``yfinance`` when
  internet access is available. If ``yfinance`` is unavailable or network
  access fails, it falls back to an embedded sample price series derived
  from historical S&P 500 data for reproducibility.
* Implements two example strategies: ``momentum_strategy`` and
  ``mean_reversion_strategy``. Each strategy generates a series of trading
  signals (1 for long, -1 for short, 0 for flat) based on simple technical
  indicators.
* Backtests strategies by computing daily portfolio returns, cumulative
  returns, annualised volatility and a Sharpe ratio. A max drawdown
  function quantifies downside risk.
* Saves a plot of the equity curve for each strategy to
  ``backtesting_equity_<strategy>.png`` in the working directory.
* Prints a summary table of performance metrics to the console.

Usage
-----
Run this script directly to execute the built‑in strategies on SPY daily
prices over the last 5 years. The example uses a 20‑day lookback period for
both momentum and mean reversion strategies. You may edit the ``main``
function to test other tickers or periods or to add new strategies.

This project demonstrates software engineering practices expected in
quantitative roles: modular design, error handling, reproducibility and
performance analysis. It can serve as a foundation for more sophisticated
backtesting engines (e.g. with transaction costs, multiple assets, or
position sizing algorithms).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf  # type: ignore
except ImportError:
    # We'll handle missing yfinance at runtime
    yf = None  # type: ignore


def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetches adjusted close prices for the given tickers between start and end.

    If ``yfinance`` is unavailable or data cannot be downloaded, falls back
    to a sample price series extracted from historical S&P 500 data. The
    fallback ensures the script runs offline.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with one column per ticker.
    """
    if yf is not None:
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)[
                "Adj Close"
            ]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data = data.dropna(how="all")
            if not data.empty:
                return data
        except Exception:
            pass

    # Fallback: synthetic price series from 1927/1928 SP500 dataset
    # The numbers were extracted from the SP500.csv file lines as documented
    # in the citation; they provide realistic price evolution for demonstration.
    fallback_prices = np.array(
        [
            17.65999984741211,
            17.510000228881836,
            16.940000534057617,
            16.389999389648438,
            16.610000610351562,
            16.700000762939453,
            16.610000610351562,
            16.360000610351562,
            16.399999618530273,
            16.219999313354492,
            15.729999542236328,
            15.930000305175781,
            16.260000228881836,
            16.299999237060547,
            16.280000686645508,
            16.260000228881836,
        ]
    )
    dates = pd.date_range(start=start, periods=len(fallback_prices), freq="D")
    df = pd.DataFrame(fallback_prices, index=dates, columns=[tickers[0]])
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes daily percentage returns from price data."""
    return prices.pct_change().dropna()


def momentum_strategy(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Generates trading signals based on a simple momentum rule.

    A long signal (1) is generated when the current price is above its
    ``lookback``‑day moving average; a short signal (-1) is generated when
    below. Flat positions (0) are used when the price is equal to the
    moving average (rare).

    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame of asset prices.
    lookback : int, optional
        Lookback window for the moving average (default 20).

    Returns
    -------
    pandas.DataFrame
        DataFrame of signals with the same index as ``prices``.
    """
    ma = prices.rolling(window=lookback).mean()
    signals = np.sign(prices - ma)
    signals = signals.fillna(0)
    return signals


def mean_reversion_strategy(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Generates trading signals based on a simple mean reversion rule.

    A long signal (1) is generated when the current price is below its
    ``lookback``‑day moving average by more than one standard deviation;
    a short signal (-1) is generated when above by more than one standard
    deviation. Otherwise, the position is flat (0).

    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame of asset prices.
    lookback : int, optional
        Lookback window for the moving average and standard deviation
        (default 20).

    Returns
    -------
    pandas.DataFrame
        DataFrame of signals with the same index as ``prices``.
    """
    ma = prices.rolling(window=lookback).mean()
    std = prices.rolling(window=lookback).std()
    upper_band = ma + std
    lower_band = ma - std

    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    signals[prices > upper_band] = -1
    signals[prices < lower_band] = 1
    signals.fillna(0, inplace=True)
    return signals


def backtest_strategy(
    returns: pd.DataFrame, signals: pd.DataFrame, annualisation_factor: float = 252.0
) -> Tuple[pd.Series, Dict[str, float]]:
    """Backtests a trading strategy given returns and signals.

    Parameters
    ----------
    returns : pandas.DataFrame
        Daily percentage returns of assets.
    signals : pandas.DataFrame
        Trading signals aligned with returns (1 for long, -1 for short, 0 for flat).
    annualisation_factor : float, optional
        Number of trading days per year (default 252).

    Returns
    -------
    pd.Series
        Cumulative equity curve of the strategy.
    dict
        Dictionary of performance metrics: cumulative_return, annualised_vol,
        sharpe_ratio, max_drawdown.
    """
    # Align signals with returns; signals act on next day's returns
    positions = signals.shift(1).reindex_like(returns).fillna(0)
    # Portfolio returns: average across assets (equal weight for each active position)
    # If no positions, return 0 for that day
    portfolio_returns = (
        positions * returns
    )
    # If multiple assets, we can average returns of active positions
    # to avoid leverage doubling when long and short simultaneously
    active_positions = positions.abs()
    portfolio_returns = portfolio_returns.sum(axis=1)
    normaliser = active_positions.sum(axis=1)
    # Avoid division by zero by replacing zeros with NaN and filling later
    normaliser = normaliser.replace(0, np.nan)
    normalised_returns = portfolio_returns / normaliser
    normalised_returns = normalised_returns.fillna(0)

    equity = (1 + normalised_returns).cumprod()
    cumulative_return = equity.iloc[-1] - 1
    daily_vol = normalised_returns.std(ddof=0)
    annualised_vol = daily_vol * np.sqrt(annualisation_factor)
    sharpe_ratio = (normalised_returns.mean() / daily_vol) * np.sqrt(annualisation_factor)
    max_dd = max_drawdown(equity)

    metrics = {
        "cumulative_return": cumulative_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
    }
    return equity, metrics


def max_drawdown(equity: pd.Series) -> float:
    """Calculates maximum drawdown from an equity curve."""
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1
    return drawdown.min()


def plot_equity_curve(
    equity: pd.Series, strategy_name: str, output_file: str | None = None
) -> None:
    """Plots and saves the equity curve of a strategy."""
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values, label=strategy_name)
    plt.title(f"Equity Curve - {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file)
    plt.close()


def print_metrics(strategy_name: str, metrics: Dict[str, float]) -> None:
    """Prints performance metrics for a strategy."""
    print(f"\nStrategy: {strategy_name}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Annualised Volatility: {metrics['annualised_vol']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")


def run_backtest(
    tickers: List[str],
    start: str,
    end: str,
    strategies: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> None:
    """Executes backtests for multiple strategies and prints results.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    start : str
        Start date for historical data in ``YYYY-MM-DD`` format.
    end : str
        End date for historical data in ``YYYY-MM-DD`` format.
    strategies : dict
        Mapping from strategy name to a function that generates signals given
        price data.
    """
    prices = fetch_prices(tickers, start, end)
    returns = compute_returns(prices)
    for name, strategy_func in strategies.items():
        signals = strategy_func(prices)
        equity, metrics = backtest_strategy(returns, signals)
        output_file = f"backtesting_equity_{name.replace(' ', '_').lower()}.png"
        plot_equity_curve(equity, name, output_file)
        print_metrics(name, metrics)


def main() -> None:
    """Example usage demonstrating two strategies on SPY over the last 5 years."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)
    tickers = ["SPY"]

    strategies: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "Momentum Strategy": lambda prices: momentum_strategy(prices, lookback=20),
        "Mean Reversion Strategy": lambda prices: mean_reversion_strategy(prices, lookback=20),
    }

    run_backtest(
        tickers=tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        strategies=strategies,
    )


if __name__ == "__main__":
    main()