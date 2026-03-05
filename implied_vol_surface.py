"""
Implied Volatility Surface Construction
======================================

This script demonstrates how to construct an implied volatility surface by
calculating option prices across a range of strikes and maturities and then
inverting the Black–Scholes formula to recover the implied volatility.  It
illustrates numerical root finding and visualisation of a 2‑dimensional
volatility surface.

Overview
--------
* **Synthetic option chain:** Choose an underlying asset price ``S0``, a risk‑free
  rate ``r`` and a base volatility function that varies with moneyness and
  maturity.  Generate call option prices on a grid of strikes (e.g. 50 % to
  150 % of ``S0``) and maturities (e.g. one month to two years) using the
  Black–Scholes formula.
* **Implied volatility calculation:** For each option price on the grid, solve
  for the implied volatility that reproduces the price using a numerical
  root‑finding algorithm.  The ``implied_volatility_call`` function uses
  Brent’s method from ``scipy.optimize``.  If ``scipy`` is unavailable, a
  custom bisection method is implemented as a fallback.
* **Visualisation:** Create a 3‑dimensional surface plot of implied volatility
  versus strike and maturity using ``matplotlib``.  Also save a 2D heatmap
  projection for more compact viewing.  The figures are saved as
  ``implied_vol_surface.png`` and ``implied_vol_heatmap.png`` in the
  working directory.

This project demonstrates numerical methods, option pricing theory and data
visualisation—skills relevant for derivatives trading and risk management
roles.  Real data can be substituted by fetching an option chain from an
exchange or data provider.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

try:
    from scipy.stats import norm  # type: ignore
    from scipy.optimize import brentq  # type: ignore
    have_scipy = True
except Exception:
    # Provide fallbacks if scipy is not available
    have_scipy = False


def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Returns the Black–Scholes price of a European call option."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Use normal cdf; approximate if scipy is missing
    if have_scipy:
        cdf = norm.cdf
    else:
        # Approximate CDF using error function
        def cdf(x: float) -> float:
            return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    return S * cdf(d1) - K * np.exp(-r * T) * cdf(d2)


def implied_volatility_call(
    C_market: float, S: float, K: float, T: float, r: float, tol: float = 1e-6
) -> float:
    """Numerically solves for the implied volatility of a call option.

    Parameters
    ----------
    C_market : float
        Observed market price of the call option.
    S : float
        Current underlying price.
    K : float
        Option strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk‑free interest rate.
    tol : float, optional
        Tolerance for the root‑finding algorithm (default 1e-6).

    Returns
    -------
    float
        Implied volatility.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if C_market < intrinsic:
        # No implied vol if price is below intrinsic value
        return 0.0
    # Use Brent's method if available
    if have_scipy:
        def objective(sigma: float) -> float:
            return black_scholes_call_price(S, K, T, r, sigma) - C_market
        return brentq(objective, 1e-6, 5.0, xtol=tol)
    # Fallback: bisection method
    low, high = 1e-6, 5.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        price = black_scholes_call_price(S, K, T, r, mid)
        if price > C_market:
            high = mid
        else:
            low = mid
        if abs(high - low) < tol:
            break
    return 0.5 * (low + high)


def base_volatility(moneyness: float, T: float) -> float:
    """Defines a synthetic volatility surface used to generate option prices.

    The function increases volatility for deep in‑the‑money or deep
    out‑of‑the‑money options and for longer maturities.  Adjust parameters
    to change the shape of the surface.
    """
    return 0.2 + 0.1 * (abs(moneyness - 1) + 0.5 * np.sqrt(T))


def main() -> None:
    # Parameters
    S0 = 100.0  # Current underlying price
    r = 0.02    # Risk‑free rate
    strike_range = np.linspace(50, 150, 25)  # Strikes from 50% to 150% of S0
    maturities = np.linspace(0.1, 2.0, 20)   # Maturities from 0.1 to 2 years

    # Generate synthetic option prices using a base volatility surface
    call_prices = np.zeros((len(maturities), len(strike_range)))
    implied_vols = np.zeros_like(call_prices)

    for i, T in enumerate(maturities):
        for j, K in enumerate(strike_range):
            mny = K / S0
            sigma = base_volatility(mny, T)
            price = black_scholes_call_price(S0, K, T, r, sigma)
            call_prices[i, j] = price
            implied_vols[i, j] = implied_volatility_call(price, S0, K, T, r)

    # Plot implied volatility surface (3D)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(strike_range, maturities)
    ax.plot_surface(X, Y, implied_vols, cmap='viridis')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface')
    fig.tight_layout()
    plt.savefig('implied_vol_surface.png')
    plt.close(fig)

    # Plot heatmap projection for a 2D view
    plt.figure(figsize=(8, 6))
    plt.imshow(implied_vols, aspect='auto', origin='lower',
               extent=[strike_range[0], strike_range[-1], maturities[0], maturities[-1]],
               cmap='viridis')
    plt.colorbar(label='Implied Volatility')
    plt.xlabel('Strike')
    plt.ylabel('Maturity (years)')
    plt.title('Implied Volatility Heatmap')
    plt.tight_layout()
    plt.savefig('implied_vol_heatmap.png')
    plt.close()

    # Print a summary for a few sample points
    print('Sample implied volatilities:')
    for K in [80, 100, 120]:
        for T in [0.5, 1.0, 1.5]:
            price = black_scholes_call_price(S0, K, T, r, base_volatility(K / S0, T))
            iv = implied_volatility_call(price, S0, K, T, r)
            print(f'K={K:3.0f}, T={T:.1f}: implied vol = {iv:.3f}')


if __name__ == '__main__':
    main()