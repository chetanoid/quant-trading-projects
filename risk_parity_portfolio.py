"""
Risk Parity Portfolio Construction
---------------------------------

This script constructs a risk‑parity portfolio from a set of assets. A risk‑parity
portfolio allocates capital such that each asset contributes an equal share of
overall portfolio risk. This is a popular approach in portfolio management
because it avoids concentration in high‑volatility assets while still
diversifying across uncorrelated sources of risk.

Key features of this implementation:

* **Data acquisition:** Fetches daily adjusted close prices for the selected
  tickers using ``yfinance``. If ``yfinance`` is unavailable or the download
  fails, the script falls back to generating synthetic correlated returns
  which are then converted into a price series.
* **Risk contributions:** Computes the covariance matrix of asset returns and
  calculates each asset's marginal and total risk contribution for a given set
  of weights.
* **Equal risk allocation:** Uses a simple multiplicative update algorithm to
  iteratively adjust weights until each asset’s risk contribution is equal.
  The resulting risk‑parity weights are normalised to sum to 1.
* **Performance evaluation:** Computes annualised return, annualised
  volatility and Sharpe ratio of both an equal‑weighted portfolio and the
  risk‑parity portfolio. It also prints the weights and risk contributions.
* **Visualisation:** Saves a bar chart of the risk‑parity weights to
  ``risk_parity_weights.png`` (if ``matplotlib`` is available).

Example usage:

    python3 risk_parity_portfolio.py --symbols AAPL MSFT GOOGL AMZN --start 2018-01-01 --end 2024-01-01

Dependencies: pandas, numpy, matplotlib (optional), yfinance (optional).

"""

import argparse
from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

import matplotlib.pyplot as plt


def download_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers using yfinance."""
    if yf is None:
        raise ImportError(
            "yfinance is not available. Please install yfinance or rely on fallback data."
        )
    data = yf.download(symbols, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    return data.dropna()


def generate_fallback_prices(n: int, d: int) -> pd.DataFrame:
    """Generate synthetic correlated price series as a fallback."""
    rng = np.random.default_rng(seed=2024)
    mean = np.zeros(d)
    A = rng.standard_normal(size=(d, d))
    cov = np.dot(A, A.T)
    cov = cov / np.max(np.abs(cov)) * (0.02 ** 2)
    returns = rng.multivariate_normal(mean, cov, size=n)
    prices = pd.DataFrame(returns, index=pd.date_range('2020-01-01', periods=n, freq='B'))
    prices = (1 + prices).cumprod()
    prices.columns = [f'ASSET{i+1}' for i in range(d)]
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute risk contributions for each asset given weights and covariance matrix."""
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)
    # Marginal risk contribution: derivative of portfolio vol wrt each weight
    mrc = (cov @ weights) / portfolio_vol
    # Total risk contribution of each asset
    rc = weights * mrc
    return rc


def risk_parity_weights(cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Solve for risk‑parity weights via an iterative multiplicative update."""
    n = cov.shape[0]
    # Start with equal weights
    w = np.ones(n) / n
    target_rc = np.ones(n) / n  # equal risk contribution target
    for _ in range(max_iter):
        rc = portfolio_risk_contribution(w, cov)
        # Avoid division by zero
        rc[rc == 0] = 1e-12
        # Update rule: multiply weights by target/actual risk contribution
        w = w * (target_rc / rc)
        # Normalise weights to sum to 1
        w = np.maximum(w, 0)
        w = w / w.sum()
        # Check convergence
        if np.linalg.norm(rc - target_rc) < tol:
            break
    return w


def annualised_return(vol_returns: pd.Series) -> float:
    """Compute annualised return from a series of daily returns."""
    return vol_returns.mean() * 252


def annualised_volatility(vol_returns: pd.Series) -> float:
    """Compute annualised volatility from a series of daily returns."""
    return vol_returns.std() * np.sqrt(252)


def sharpe_ratio(vol_returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute the Sharpe ratio of a return series (risk‑free rate assumed constant)."""
    return (vol_returns.mean() - risk_free / 252) / vol_returns.std() * np.sqrt(252)


def main() -> None:
    parser = argparse.ArgumentParser(description='Risk Parity Portfolio Construction')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                        help='Ticker symbols to include in the portfolio')
    parser.add_argument('--start', default='2018-01-01', help='Start date for historical data')
    parser.add_argument('--end', default='2024-01-01', help='End date for historical data')
    args = parser.parse_args()
    try:
        prices = download_prices(args.symbols, args.start, args.end)
        print(f"Downloaded {len(prices)} rows of price data for {args.symbols}.")
    except Exception as e:
        print(f"Warning: failed to download data due to {e}. Using synthetic data.")
        prices = generate_fallback_prices(n=500, d=len(args.symbols))
        prices.columns = args.symbols
    returns = compute_returns(prices)
    cov = returns.cov().values
    # Compute risk parity weights
    rp_weights = risk_parity_weights(cov)
    # Compute equal weights
    ew_weights = np.ones(len(args.symbols)) / len(args.symbols)
    # Portfolio returns
    rp_returns = returns.dot(rp_weights)
    ew_returns = returns.dot(ew_weights)
    # Risk contributions
    rp_rc = portfolio_risk_contribution(rp_weights, cov)
    # Annualised metrics
    rp_ret = annualised_return(rp_returns)
    rp_vol = annualised_volatility(rp_returns)
    rp_sr = sharpe_ratio(rp_returns)
    ew_ret = annualised_return(ew_returns)
    ew_vol = annualised_volatility(ew_returns)
    ew_sr = sharpe_ratio(ew_returns)
    print("Risk Parity Weights:")
    for sym, w, rc in zip(args.symbols, rp_weights, rp_rc):
        print(f"  {sym}: weight={w:.3f}, risk contribution={rc:.3f}")
    print(f"\nRisk Parity Portfolio: Return={rp_ret:.2%}, Volatility={rp_vol:.2%}, Sharpe={rp_sr:.2f}")
    print(f"Equal Weight Portfolio: Return={ew_ret:.2%}, Volatility={ew_vol:.2%}, Sharpe={ew_sr:.2f}")
    # Plot weight allocation
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(args.symbols, rp_weights)
        ax.set_title('Risk Parity Portfolio Weights')
        ax.set_ylabel('Weight')
        plt.tight_layout()
        fig.savefig('risk_parity_weights.png')
        plt.close(fig)
        print("Plot saved to 'risk_parity_weights.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()