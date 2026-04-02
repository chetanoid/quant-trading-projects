"""
rough_bergomi_model.py
----------------------

This script simulates paths of the rough Bergomi stochastic volatility model.
The rough Bergomi (rBergomi) model is a non‑Markovian stochastic volatility
model where the variance process exhibits fractional behavior with Hurst
parameter H<0.5. It has been used to fit the steep implied volatility
skews observed in equity markets.  Implementing a rough volatility model
requires careful handling of fractional Gaussian noise and correlated Brownian
motions, demonstrating high‑level quantitative modelling beyond standard
Black–Scholes or Heston models.

Key components:

* **Fractional Brownian motion (fBm)**: We construct increments of fBm
  using the Hosking method via the autocovariance of fractional Gaussian
  noise.  The Hurst parameter `H` controls the roughness (H≈0.1–0.4 for
  equity volatility).
* **Variance process**: The instantaneous variance V_t is expressed as
  `V_t = xi * exp(X_t)`, where `X_t` involves an integral of the fBm.
* **Asset price dynamics**: The log‑price S_t evolves according to
  `d log S_t = -0.5 * V_t dt + sqrt(V_t) dW_t` where W and the fBm are
  correlated with coefficient `rho`.

The script provides functions to simulate a single rBergomi path and to
estimate the Monte Carlo price of a European call option.  It saves a plot
of the simulated variance and price paths to `rough_bergomi_price.png`.
This project demonstrates advanced stochastic calculus, numerical methods,
and the ability to implement models not available in standard libraries.
"""

import math
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib.pyplot as plt

def generate_fgn(n, H, dt):
    """Generate fractional Gaussian noise (fGn) of length n with Hurst H.

    Uses the Hosking method to generate increments of fractional Brownian
    motion with covariance given by
        gamma(k) = 0.5 * ((k + 1)**(2H) - 2*k**(2H) + (k - 1)**(2H)).

    Parameters
    ----------
    n : int
        Number of increments.
    H : float
        Hurst parameter (0 < H < 1).
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Fractional Gaussian noise increments of length n.
    """
    # Autocovariance for fGn
    def gamma(k):
        return 0.5 * ((abs(k + 1)**(2 * H)) - 2 * (abs(k)**(2 * H)) + (abs(k - 1)**(2 * H)))

    # Build covariance matrix
    cov = np.fromfunction(lambda i, j: gamma(i - j), (n, n), dtype=float)
    # Cholesky decomposition
    L = np.linalg.cholesky(cov)
    z = np.random.normal(size=n)
    fgn = L @ z
    # Scale by dt^H
    return fgn * (dt**H)


def simulate_rbergomi(T=1.0, n_steps=250, xi=0.04, eta=1.5, H=0.1, rho=-0.7, S0=100.0):
    """Simulate a single rough Bergomi path.

    Parameters
    ----------
    T : float
        Maturity time.
    n_steps : int
        Number of time steps.
    xi : float
        Forward variance level.
    eta : float
        Volatility of volatility parameter.
    H : float
        Hurst parameter for the fBm (0 < H < 0.5).
    rho : float
        Correlation between the two Brownian motions.
    S0 : float
        Initial spot price.

    Returns
    -------
    t : np.ndarray
        Time grid.
    S : np.ndarray
        Simulated asset price path.
    V : np.ndarray
        Simulated variance path.
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    # Generate fGn for the rough volatility driver
    fgn = generate_fgn(n_steps, H, dt)
    # Cumulative sum to get fBm increments
    B_H = np.cumsum(fgn)
    # Generate independent Brownian motion increments
    dW1 = np.random.normal(scale=np.sqrt(dt), size=n_steps)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(scale=np.sqrt(dt), size=n_steps)
    # Variance process X_t and V_t
    X = eta * B_H - 0.5 * eta**2 * (np.arange(1, n_steps + 1) ** (2 * H)) * dt
    V = xi * np.exp(X)
    # Asset price path
    S = np.empty(n_steps + 1)
    S[0] = S0
    for i in range(n_steps):
        S[i + 1] = S[i] * math.exp(-0.5 * V[i] * dt + math.sqrt(max(V[i], 0)) * dW2[i])
    return t, S, V


def price_call_mc(strike=100.0, maturity=1.0, n_paths=1000, **kwargs):
    """Estimate the price of a European call option under the rough Bergomi model using Monte Carlo.

    Parameters
    ----------
    strike : float
        Strike price.
    maturity : float
        Option maturity.
    n_paths : int
        Number of Monte Carlo sample paths.
    kwargs : dict
        Additional parameters for simulate_rbergomi.

    Returns
    -------
    price : float
        Estimated option price.
    stderr : float
        Standard error of the estimate.
    """
    payoffs = []
    for _ in range(n_paths):
        _, S, _ = simulate_rbergomi(T=maturity, **kwargs)
        payoff = max(S[-1] - strike, 0.0)
        payoffs.append(payoff)
    price = np.mean(payoffs) * math.exp(-0.0 * maturity)  # Assume zero interest rate
    stderr = np.std(payoffs) / math.sqrt(n_paths)
    return price, stderr


def main():
    # Parameters for simulation
    T = 1.0
    n_steps = 300
    xi = 0.04
    eta = 1.5
    H = 0.1
    rho = -0.7
    S0 = 100.0
    strike = 100.0
    # Simulate single path
    t, S, V = simulate_rbergomi(T=T, n_steps=n_steps, xi=xi, eta=eta, H=H, rho=rho, S0=S0)
    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(t, S, color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Variance', color=color)
    ax2.plot(t[:-1], V, color=color, linestyle='--', label='Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Rough Bergomi Price and Variance Paths')
    fig.tight_layout()
    plt.savefig('rough_bergomi_price.png')
    plt.close(fig)
    # Price a call option
    price, stderr = price_call_mc(
        strike=strike,
        maturity=T,
        n_paths=500,
        n_steps=n_steps,
        xi=xi,
        eta=eta,
        H=H,
        rho=rho,
        S0=S0,
    )
    print(f"MC price of call option: {price:.4f} +/- {stderr:.4f}")


if __name__ == '__main__':
    main()
