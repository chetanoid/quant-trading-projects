"""
Monte Carlo Option Pricing
==========================

This script prices a European call option using a Monte Carlo simulation
of a geometric Brownian motion (GBM).  It optionally estimates
volatility from real SPY returns via the ``yfinance`` package.  If
``yfinance`` or network access is unavailable, it defaults to a
predefined volatility parameter.  The script outputs the option price
and its standard error and, if ``matplotlib`` is available, saves
example price paths and a histogram of option payoffs as PNG files.

Usage:

    python3 monte_carlo_option_pricing.py

Author: OpenAI assistant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False


def estimate_volatility(symbol: str = "SPY", period: str = "1y") -> float:
    """Estimate annualised volatility from historical returns.

    Fetches adjusted close prices for the specified symbol over the given
    period and computes the standard deviation of daily returns,
    annualising by multiplying by sqrt(252).  Returns a default value
    (20% volatility) if data cannot be downloaded.

    Parameters
    ----------
    symbol : str
        Ticker symbol (default 'SPY').
    period : str
        Period string understood by yfinance (default '1y').

    Returns
    -------
    float
        Annualised volatility.
    """
    if yf is not None:
        try:
            data = yf.download(symbol, period=period, progress=False)["Adj Close"]
            returns = data.pct_change().dropna()
            if len(returns) > 0:
                return returns.std() * np.sqrt(252)
        except Exception:
            pass
    # Fallback volatility (20%)
    return 0.20


def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    steps: int,
) -> np.ndarray:
    """Simulate GBM price paths.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk‑free interest rate (continuously compounded).
    sigma : float
        Annualised volatility of returns.
    T : float
        Time to maturity in years.
    n_paths : int
        Number of simulated paths.
    steps : int
        Number of time steps per path.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, steps+1) containing simulated price paths.
    """
    dt = T / steps
    # Initialise matrix of zeros and set the first column to S0
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    # Generate random increments
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return paths


def price_european_call(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float | None = None,
    n_paths: int = 20000,
    steps: int = 252,
) -> Tuple[float, float]:
    """Price a European call option via Monte Carlo simulation.

    Parameters
    ----------
    S0 : float, optional
        Initial stock price (default 100).
    K : float, optional
        Strike price (default 100).
    T : float, optional
        Time to maturity in years (default 1 year).
    r : float, optional
        Risk‑free interest rate (continuously compounded) (default 5%).
    sigma : float or None, optional
        Annualised volatility.  If ``None``, volatility is estimated
        from SPY returns via ``estimate_volatility`` (default ``None``).
    n_paths : int, optional
        Number of Monte Carlo simulations (default 20 000).
    steps : int, optional
        Number of time steps per simulation (default 252 trading days).

    Returns
    -------
    Tuple[float, float]
        Estimated option price and its standard error.
    """
    if sigma is None:
        sigma = estimate_volatility("SPY", "2y")
    # Simulate price paths
    paths = simulate_paths(S0, r, sigma, T, n_paths, steps)
    # Payoff at maturity
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    discounted_payoff = np.exp(-r * T) * payoff
    price = np.mean(discounted_payoff)
    se = np.std(discounted_payoff) / np.sqrt(n_paths)
    # Plot price paths and payoff distribution if matplotlib is available
    if _HAVE_MATPLOTLIB:
        try:
            # Plot a subset of paths for visualisation
            n_examples = min(25, n_paths)
            plt.figure()
            for i in range(n_examples):
                plt.plot(np.linspace(0, T, steps + 1), paths[i, :], alpha=0.6)
            plt.title("Sample Simulated Price Paths")
            plt.xlabel("Time (years)")
            plt.ylabel("Price")
            plt.tight_layout()
            plt.savefig("option_price_paths.png")
            plt.close()
            # Histogram of discounted payoffs
            plt.figure()
            plt.hist(discounted_payoff, bins=50, alpha=0.7)
            plt.title("Distribution of Discounted Payoffs")
            plt.xlabel("Present Value of Payoff")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("option_payoff_distribution.png")
            plt.close()
            print("Price path and payoff distribution plots saved to option_price_paths.png and option_payoff_distribution.png")
        except Exception:
            pass
    return price, se


def main() -> None:
    price, se = price_european_call()
    print(f"Estimated call option price: {price:.4f} ± {se:.4f} (standard error)")


if __name__ == "__main__":
    main()