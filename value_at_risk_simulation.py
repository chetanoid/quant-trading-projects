"""
Value at Risk and Conditional Value at Risk (CVaR) Estimation
------------------------------------------------------------

This script demonstrates how to compute parametric and historical Value at
Risk (VaR) and Conditional Value at Risk (also known as Expected Shortfall)
for a portfolio of equities. It illustrates:

* Downloading daily adjusted close prices for a set of tickers using ``yfinance``.
* Calculating daily returns and constructing a portfolio return series given
  equal weightings (you may adjust the weights).
* Computing parametric VaR using a normal distribution assumption.
* Computing historical (empirical) VaR from the distribution of observed
  portfolio returns.
* Computing CVaR (Expected Shortfall) as the average of returns that exceed
  the VaR threshold.
* Plotting the distribution of portfolio returns with the VaR and CVaR
  thresholds indicated. The figure is saved to ``portfolio_var_distribution.png``.

If ``yfinance`` is not available or fails to download data, the script falls
back to generating synthetic returns from a simple random process to ensure
that it runs offline. In such cases, the results are illustrative rather
than realistic.

Example usage:

    python3 value_at_risk_simulation.py --symbols AAPL MSFT GOOGL AMZN --start 2018-01-01 --end 2023-12-31 --confidence 0.95

Dependencies: pandas, numpy, matplotlib, scipy (for the normal quantile), yfinance (optional).
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm  # type: ignore
except Exception:
    norm = None  # type: ignore


def download_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers using yfinance."""
    if yf is None:
        raise ImportError(
            "yfinance is not available. Please install yfinance or rely on fallback data."
        )
    data = yf.download(symbols, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    data = data.dropna()
    if data.empty:
        raise RuntimeError("Downloaded data is empty.")
    return data


def generate_fallback_returns(n: int = 500, d: int = 4) -> pd.DataFrame:
    """Generate synthetic daily returns for d assets using Gaussian random variables.

    Parameters
    ----------
    n : int
        Number of observations.
    d : int
        Number of assets.

    Returns
    -------
    DataFrame
        A DataFrame of synthetic returns.
    """
    rng = np.random.default_rng(seed=123)
    # Assume 1% daily volatility and zero mean returns
    mean = np.zeros(d)
    # Generate a random covariance matrix
    A = rng.standard_normal(size=(d, d))
    cov = np.dot(A, A.T)  # positive semi-definite
    cov = cov / np.max(np.abs(cov)) * (0.01 ** 2)
    returns = rng.multivariate_normal(mean, cov, size=n)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    cols = [f'ASSET{i+1}' for i in range(d)]
    return pd.DataFrame(returns, index=dates, columns=cols)


def compute_portfolio_returns(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute daily portfolio returns given price series and weights."""
    returns = prices.pct_change().dropna()
    # Normalise weights
    weights = weights / weights.sum()
    portfolio_returns = returns.dot(weights)
    return portfolio_returns


def parametric_var(return_series: pd.Series, confidence: float) -> Tuple[float, float]:
    """Compute parametric VaR and CVaR assuming normally distributed returns.

    Returns a tuple of (VaR, CVaR) where CVaR is the expected shortfall
    beyond VaR.
    """
    mu = return_series.mean()
    sigma = return_series.std()
    if norm is None:
        raise ImportError(
            "scipy.stats.norm is required for parametric VaR. Install SciPy or set --method historical."
        )
    z = norm.ppf(1 - confidence)
    var = -(mu + sigma * z)
    # Expected shortfall of returns lives further into the loss tail than VaR.
    cvar = -mu + sigma * norm.pdf(z) / (1 - confidence)
    return var, cvar


def historical_var(return_series: pd.Series, confidence: float) -> Tuple[float, float]:
    """Compute historical (empirical) VaR and CVaR."""
    sorted_returns = return_series.sort_values()
    cutoff_index = max(1, int((1 - confidence) * len(sorted_returns)))
    var = -sorted_returns.iloc[:cutoff_index].max()
    cvar = -sorted_returns.iloc[:cutoff_index].mean()
    return var, cvar


def plot_return_distribution(returns: pd.Series, var: float, cvar: float, confidence: float, filename: str) -> None:
    """Plot the distribution of returns with VaR and CVaR markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns, bins=50, density=True, alpha=0.6, color='skyblue')
    ax.axvline(-var, color='red', linestyle='--', label=f'VaR ({confidence*100:.0f}%): {-var:.2%}')
    ax.axvline(-cvar, color='purple', linestyle='--', label=f'CVaR ({confidence*100:.0f}%): {-cvar:.2%}')
    ax.set_title('Portfolio Return Distribution with VaR and CVaR')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Value at Risk and CVaR Calculation')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                        help='Ticker symbols for the portfolio components')
    parser.add_argument('--start', default='2018-01-01', help='Start date for historical data')
    parser.add_argument('--end', default='2023-12-31', help='End date for historical data')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level for VaR and CVaR (e.g., 0.95)')
    parser.add_argument('--method', choices=['parametric', 'historical'], default='parametric',
                        help='Method for VaR calculation')
    args = parser.parse_args()
    try:
        prices = download_prices(args.symbols, args.start, args.end)
        print(f"Downloaded {len(prices)} rows of price data for {args.symbols}.")
    except Exception as e:
        print(f"Warning: failed to download data due to {e}. Using synthetic returns.")
        # Generate synthetic returns (not prices) for fallback
        returns = generate_fallback_returns(n=500, d=len(args.symbols))
        prices = (1 + returns).cumprod()  # convert returns to a price-like series
        prices.columns = args.symbols
    # Use equal weights for simplicity
    weights = np.ones(len(args.symbols)) / len(args.symbols)
    portfolio_returns = compute_portfolio_returns(prices, weights)
    # Compute VaR and CVaR
    if args.method == 'parametric':
        var, cvar = parametric_var(portfolio_returns, args.confidence)
        print(f"Parametric VaR ({args.confidence*100:.0f}%): {var:.4%}\nParametric CVaR: {cvar:.4%}")
    else:
        var, cvar = historical_var(portfolio_returns, args.confidence)
        print(f"Historical VaR ({args.confidence*100:.0f}%): {var:.4%}\nHistorical CVaR: {cvar:.4%}")
    # Plot distribution
    try:
        plot_return_distribution(portfolio_returns, var, cvar, args.confidence,
                                filename='portfolio_var_distribution.png')
        print("Plot saved to 'portfolio_var_distribution.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
