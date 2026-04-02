"""
Portfolio Optimisation with Real Data
====================================

This script demonstrates modern portfolio theory on a small set of
assets using real historical price data.  It downloads daily
adjusted closes for a group of popular stocks via the ``yfinance``
package and constructs thousands of random portfolios to explore
the risk–return trade‑off.  For each random portfolio, it computes
expected annualised return, annualised volatility and the Sharpe
ratio.  The portfolio with the highest Sharpe ratio and the portfolio
with the lowest volatility are highlighted in a scatter plot of
volatility versus return.

If ``yfinance`` is unavailable or network access fails, the script
falls back to a small synthetic dataset embedded in the code.  The
resulting scatter plot is saved to ``portfolio_optimisation.png`` and
the portfolio statistics are written to ``portfolio_optimisation.csv``.

Run this script directly to perform the optimisation:

    python3 portfolio_optimization.py

"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False

# Fallback synthetic price data (10 days of prices for three assets).  While
# these values are fictional, they provide a deterministic example if
# live market data cannot be fetched.  Feel free to replace with more
# realistic fallback data if available.
FALLBACK_DATA = {
    "AssetA": [10.0, 10.1, 9.9, 10.2, 10.3, 10.4, 10.1, 10.2, 10.5, 10.6],
    "AssetB": [20.0, 20.2, 19.8, 20.5, 20.4, 20.6, 20.3, 20.4, 20.8, 21.0],
    "AssetC": [30.0, 30.3, 29.9, 30.1, 30.2, 30.5, 30.2, 30.3, 30.7, 30.9],
}


def fetch_prices(tickers: List[str], start: str = "2018-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    """Fetch adjusted close prices for the given tickers.

    Attempts to download data from Yahoo! Finance.  If unsuccessful,
    returns a DataFrame built from the fallback synthetic data.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to download.
    start : str
        Start date for the historical data (YYYY-MM-DD).
    end : str
        End date for the historical data (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices indexed by date with one
        column per ticker.
    """
    if yf is not None:
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            # Drop rows with any missing values
            data = data.dropna()
            if not data.empty:
                return data
        except Exception:
            pass
    # Fall back to synthetic data if downloading fails
    return pd.DataFrame(FALLBACK_DATA)


def generate_random_portfolios(
    returns: pd.DataFrame, n_portfolios: int = 5000, risk_free: float = 0.02
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate random portfolios and compute their statistics.

    Each portfolio is defined by a random weight vector that sums to one.
    For each portfolio we compute annualised return, annualised
    volatility and the Sharpe ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the assets.
    n_portfolios : int, optional
        Number of random portfolios to generate (default is 5000).
    risk_free : float, optional
        Risk‑free rate used for Sharpe ratio calculations (default 2%).

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        A DataFrame of portfolio statistics (columns: return, volatility,
        sharpe) and an array of portfolio weights (shape n_portfolios x n_assets).
    """
    mean_returns = returns.mean() * 252  # annualise
    cov_matrix = returns.cov() * 252  # annualise
    n_assets = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    weights_record = np.zeros((n_portfolios, n_assets))
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record[i] = weights
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free) / portfolio_volatility if portfolio_volatility != 0 else 0.0
        results[i] = [portfolio_return, portfolio_volatility, sharpe_ratio]
    columns = ["return", "volatility", "sharpe"]
    return pd.DataFrame(results, columns=columns), weights_record


def identify_extremes(
    stats: pd.DataFrame, weights: np.ndarray
) -> Tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
    """Identify the maximum Sharpe ratio and minimum volatility portfolios.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame containing portfolio statistics.
    weights : np.ndarray
        Corresponding array of portfolio weights.

    Returns
    -------
    Tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]
        A tuple containing the row of stats and weight vector for the
        maximum‑Sharpe portfolio and the row of stats and weight vector for
        the minimum‑volatility portfolio.
    """
    max_sharpe_idx = stats["sharpe"].idxmax()
    min_vol_idx = stats["volatility"].idxmin()
    return (
        stats.loc[max_sharpe_idx],
        weights[max_sharpe_idx],
        stats.loc[min_vol_idx],
        weights[min_vol_idx],
    )


def plot_efficient_frontier(
    stats: pd.DataFrame,
    max_sharpe: pd.Series,
    min_vol: pd.Series,
    filename: str = "portfolio_optimisation.png",
) -> None:
    """Plot the efficient frontier and highlight key portfolios.

    Parameters
    ----------
    stats : pd.DataFrame
        Portfolio statistics with columns ``return`` and ``volatility``.
    max_sharpe : pd.Series
        Row from ``stats`` corresponding to the maximum Sharpe ratio.
    min_vol : pd.Series
        Row from ``stats`` corresponding to the minimum volatility.
    filename : str, optional
        Name of the PNG file to save (default 'portfolio_optimisation.png').
    """
    if not _HAVE_MATPLOTLIB:
        return
    try:
        plt.figure()
        # Scatter plot of all portfolios
        plt.scatter(
            stats["volatility"], stats["return"], s=5, alpha=0.4
        )
        # Highlight maximum Sharpe and minimum volatility portfolios
        plt.scatter(
            max_sharpe["volatility"], max_sharpe["return"],
            marker="*", s=200, label="Max Sharpe"
        )
        plt.scatter(
            min_vol["volatility"], min_vol["return"],
            marker="X", s=200, label="Min Vol"
        )
        plt.xlabel("Volatility (Annualised)")
        plt.ylabel("Return (Annualised)")
        plt.title("Portfolio Risk–Return Trade‑off")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception:
        pass


def main() -> None:
    # Define a universe of assets for the optimisation.  You can
    # customise this list as desired.  Typical high‑liquidity symbols
    # include technology mega‑caps and broad equity indices.
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    prices = fetch_prices(tickers)
    # Compute daily returns
    returns = prices.pct_change().dropna()
    # Generate random portfolios
    stats, weights = generate_random_portfolios(returns, n_portfolios=5000, risk_free=0.02)
    max_sharpe_stats, max_sharpe_weights, min_vol_stats, min_vol_weights = identify_extremes(stats, weights)
    # Print summary of extremes
    print("Maximum Sharpe Ratio Portfolio:")
    print(max_sharpe_stats)
    print("Weights:")
    print({name: float(weight) for name, weight in zip(prices.columns, max_sharpe_weights.round(3))})
    print()
    print("Minimum Volatility Portfolio:")
    print(min_vol_stats)
    print("Weights:")
    print({name: float(weight) for name, weight in zip(prices.columns, min_vol_weights.round(3))})
    # Save results to CSV
    stats.to_csv("portfolio_optimisation.csv", index=False)
    print("Portfolio statistics saved to portfolio_optimisation.csv")
    # Generate plot
    plot_efficient_frontier(stats, max_sharpe_stats, min_vol_stats, filename="portfolio_optimisation.png")
    if _HAVE_MATPLOTLIB:
        print("Efficient frontier plot saved to portfolio_optimisation.png")


if __name__ == "__main__":
    main()
