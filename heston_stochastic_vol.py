r"""
Heston Stochastic Volatility Model Simulation and Option Pricing
----------------------------------------------------------------

This script implements a simplified version of the Heston stochastic volatility
model, a popular framework for modelling the joint dynamics of an asset price
and its variance. The model assumes that the underlying asset price ``S_t``
and its instantaneous variance ``v_t`` follow coupled stochastic
differential equations:

.. math::

   dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_{1,t},\qquad
   dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_{2,t},

where ``W1`` and ``W2`` are Brownian motions with correlation ``rho``.

The script demonstrates how to:

* Simulate sample paths of the Heston model using the Euler–Maruyama method.
* Price a European call option via Monte Carlo simulation under the Heston
  dynamics. The discounted payoffs are averaged to obtain an estimate of
  the option price.
* Visualise the simulated asset price and variance trajectories.

The parameters used below are illustrative; in practice, the Heston model
parameters are calibrated to market data. Real historical SPY prices are
downloaded via ``yfinance`` to set the initial price and risk‐free rate when
available. If the network is unavailable, a fallback synthetic initial price
and risk‐free rate are used. The results file ``heston_sample_paths.png``
shows sample trajectories of the asset price and variance, and the script
prints the estimated call option price.

This project illustrates advanced derivative pricing and stochastic processes
knowledge, showcasing understanding of coupled SDEs and Monte Carlo methods.
"""

import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None  # ``yfinance`` is optional; synthetic data will be used if unavailable.


def fetch_initial_price_and_rate(ticker: str = "SPY"):
    """Fetch the latest closing price and an estimate of the risk‐free rate.

    Attempts to download daily prices for the given ticker using ``yfinance`` and
    returns the most recent close as the initial asset price ``S0``. An
    approximate risk‐free rate of 1.5% per annum is assumed if live rates are
    unavailable. When the network or ``yfinance`` is not available, default
    values are returned.

    Returns
    -------
    S0 : float
        Initial asset price.
    r : float
        Annualised risk‐free rate.
    """
    if yf is None:
        # fallback values
        return 100.0, 0.015
    try:
        data = yf.download(ticker, period="5d", interval="1d")
        if not data.empty:
            S0 = data["Close"].iloc[-1]
            return float(S0), 0.015
    except Exception:
        pass
    # fallback synthetic values
    return 100.0, 0.015


def simulate_heston(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    seed: int | None = None,
):
    """Simulate sample paths of the Heston model using Euler–Maruyama.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance.
    mu : float
        Drift of the asset price.
    kappa : float
        Mean‐reversion speed of the variance.
    theta : float
        Long‐run average variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation between the two Brownian motions.
    T : float
        Time horizon in years.
    n_steps : int
        Number of discretisation steps.
    n_paths : int, default 1000
        Number of simulated paths.
    seed : int | None, default None
        Random seed for reproducibility.

    Returns
    -------
    S_paths : numpy.ndarray
        Simulated asset price paths of shape (n_paths, n_steps+1).
    v_paths : numpy.ndarray
        Simulated variance paths of shape (n_paths, n_steps+1).
    t_grid : numpy.ndarray
        Time grid of shape (n_steps+1,).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    S_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    # Precompute constants
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        # Draw two correlated standard normal samples
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
        v_prev = v_paths[:, i]
        # Enforce positivity of variance by maxing at zero
        v_paths[:, i + 1] = np.maximum(
            v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.maximum(v_prev, 0)) * dW2,
            0.0,
        )
        S_paths[:, i + 1] = (
            S_paths[:, i]
            + mu * S_paths[:, i] * dt
            + np.sqrt(np.maximum(v_prev, 0)) * S_paths[:, i] * dW1
        )
    return S_paths, v_paths, t_grid


def price_european_call_mc(
    S_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
):
    """Price a European call option via Monte Carlo under Heston paths.

    Parameters
    ----------
    S_paths : numpy.ndarray
        Simulated asset price paths of shape (n_paths, n_steps+1).
    K : float
        Strike price of the call option.
    r : float
        Risk‐free rate (annualised).
    T : float
        Maturity in years.

    Returns
    -------
    call_price : float
        Estimated call option price.
    """
    payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    discount_factor = np.exp(-r * T)
    return float(discount_factor * payoff.mean())


def main():
    """Simulate Heston model and price a European call option."""
    # Fetch initial conditions
    S0, r = fetch_initial_price_and_rate()
    v0 = 0.04  # initial variance (20% vol squared)
    # Heston parameters (illustrative)
    mu = 0.05
    kappa = 1.5
    theta = 0.04  # long-run variance (20% vol squared)
    sigma = 0.6
    rho = -0.5
    T = 1.0  # 1 year horizon
    n_steps = 252  # daily steps
    n_paths = 2000
    # Simulate sample paths
    S_paths, v_paths, t_grid = simulate_heston(
        S0, v0, mu, kappa, theta, sigma, rho, T, n_steps, n_paths, seed=42
    )
    # Price a 1-year at-the-money call
    K = S0
    call_price = price_european_call_mc(S_paths, K, r, T)
    print(f"Initial price S0: {S0:.2f}, risk-free rate: {r:.4f}")
    print(f"Estimated call price (Heston): {call_price:.4f}")
    # Plot sample paths for visual insight
    plt.figure(figsize=(10, 6))
    for i in range(min(5, n_paths)):
        plt.plot(t_grid, S_paths[i], alpha=0.8)
    plt.title("Sample Heston asset price paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("heston_asset_paths.png")
    plt.close()
    plt.figure(figsize=(10, 6))
    for i in range(min(5, n_paths)):
        plt.plot(t_grid, v_paths[i], alpha=0.8)
    plt.title("Sample Heston variance paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.savefig("heston_variance_paths.png")
    plt.close()


if __name__ == "__main__":
    main()
