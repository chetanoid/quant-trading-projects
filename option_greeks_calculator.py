"""
Option Pricing and Greeks Calculator
-----------------------------------

This script computes the Black–Scholes price and option Greeks (Delta,
Gamma, Vega, Theta and Rho) for European call and put options. It
demonstrates how to evaluate the sensitivity of option prices to
changes in underlying variables, which is critical for risk
management and hedging.

Key features of this implementation:

* **Inputs:** Allows the user to specify the option type (call or put),
  current stock price, strike price, risk‑free rate, volatility and
  time to maturity. Defaults are provided for convenience.
* **Black–Scholes valuation:** Computes the fair value of the option
  using the Black–Scholes formula. The cumulative distribution
  function (CDF) and probability density function (PDF) of the
  standard normal distribution are implemented via ``scipy.stats`` if
  available, or approximated via a simple numerical approach as a
  fallback.
* **Greeks computation:** Calculates Delta, Gamma, Vega, Theta and Rho
  analytically using the standard closed‑form expressions.
* **Visualisation:** Generates a plot of each Greek as a function of
  the underlying stock price across a range of values (50–150% of the
  current price). The plot is saved to ``option_greeks.png`` if
  ``matplotlib`` is available.

Example usage:

    python3 option_greeks_calculator.py --type call --spot 100 --strike 100 \
        --rate 0.03 --vol 0.2 --t 1

Dependencies: numpy, scipy (optional), matplotlib (optional).

"""

import argparse
import math
from typing import Tuple

import numpy as np

try:
    from scipy.stats import norm  # type: ignore
except Exception:
    norm = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def std_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Return the cumulative distribution function of the standard normal.

    Uses scipy.stats.norm.cdf if available; otherwise falls back to a
    numerical approximation using the error function.
    """
    if norm is not None:
        return norm.cdf(x)
    # fallback: use erf from math to approximate CDF
    return 0.5 * (1 + np.erf(x / math.sqrt(2)))


def std_norm_pdf(x: np.ndarray) -> np.ndarray:
    """Return the probability density function of the standard normal.

    Uses scipy.stats.norm.pdf if available; otherwise computes via
    formula.
    """
    if norm is not None:
        return norm.pdf(x)
    return (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x ** 2)


def black_scholes(
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    t: float
) -> Tuple[float, float, float, float, float, float]:
    """Compute Black–Scholes price and Greeks for a European option.

    Returns a tuple (price, delta, gamma, vega, theta, rho).

    Parameters
    ----------
    option_type : str
        Either "call" or "put".
    spot : float
        Current stock price.
    strike : float
        Option strike price.
    rate : float
        Continuous risk‑free interest rate (e.g. 0.05 for 5%).
    vol : float
        Volatility (standard deviation) of the underlying stock.
    t : float
        Time to maturity in years.
    """
    if spot <= 0 or strike <= 0 or vol <= 0 or t <= 0:
        raise ValueError("Spot, strike, volatility and time must be positive.")
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    # Compute CDF and PDF
    Nd1 = std_norm_cdf(np.array([d1]))[0]
    Nd2 = std_norm_cdf(np.array([d2]))[0]
    pdf_d1 = std_norm_pdf(np.array([d1]))[0]
    if option_type.lower() == 'call':
        price = spot * Nd1 - strike * math.exp(-rate * t) * Nd2
        delta = Nd1
        rho = strike * t * math.exp(-rate * t) * Nd2
    else:  # put
        price = strike * math.exp(-rate * t) * (1 - Nd2) - spot * (1 - Nd1)
        delta = Nd1 - 1
        rho = -strike * t * math.exp(-rate * t) * (1 - Nd2)
    gamma = pdf_d1 / (spot * vol * math.sqrt(t))
    vega = spot * pdf_d1 * math.sqrt(t)
    theta = (
        - (spot * pdf_d1 * vol) / (2 * math.sqrt(t))
        - rate * strike * math.exp(-rate * t) * Nd2
        + (rate * spot * Nd1 if option_type.lower() == 'put' else -rate * spot * Nd1)
    )
    return price, delta, gamma, vega, theta, rho


def plot_greeks(
    option_type: str,
    strike: float,
    rate: float,
    vol: float,
    t: float,
    spot: float
) -> None:
    """Plot option Greeks as functions of the underlying price.

    Creates a range of underlying prices (50–150% of spot) and computes
    each Greek for those prices. Saves the resulting figure to
    ``option_greeks.png``.
    """
    if plt is None:
        return
    prices = np.linspace(0.5 * spot, 1.5 * spot, 50)
    delta_vals = []
    gamma_vals = []
    vega_vals = []
    theta_vals = []
    rho_vals = []
    for s in prices:
        _, delta, gamma, vega, theta, rho = black_scholes(option_type, s, strike, rate, vol, t)
        delta_vals.append(delta)
        gamma_vals.append(gamma)
        vega_vals.append(vega)
        theta_vals.append(theta)
        rho_vals.append(rho)
    # Plot the Greeks
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    ax_list = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]
    greeks = [delta_vals, gamma_vals, vega_vals, theta_vals, rho_vals]
    titles = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    for ax, greek_vals, title in zip(ax_list, greeks, titles):
        ax.plot(prices, greek_vals)
        ax.set_title(title)
        ax.set_xlabel('Underlying Price')
        ax.set_ylabel(title)
    # Hide the empty subplot
    axs[1, 2].axis('off')
    plt.tight_layout()
    fig.savefig('option_greeks.png')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Black–Scholes Option Pricing and Greeks')
    parser.add_argument('--type', dest='option_type', default='call', choices=['call', 'put'],
                        help='Option type: call or put (default: call)')
    parser.add_argument('--spot', type=float, default=100.0, help='Current stock price')
    parser.add_argument('--strike', type=float, default=100.0, help='Strike price of the option')
    parser.add_argument('--rate', type=float, default=0.02, help='Risk‑free rate (annual)')
    parser.add_argument('--vol', type=float, default=0.2, help='Volatility (annual)')
    parser.add_argument('--t', type=float, default=1.0, help='Time to maturity (in years)')
    args = parser.parse_args()
    # Compute price and Greeks at current spot
    price, delta, gamma, vega, theta, rho = black_scholes(
        args.option_type, args.spot, args.strike, args.rate, args.vol, args.t
    )
    print(f"{args.option_type.capitalize()} Option Price: {price:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.6f}")
    print(f"Vega: {vega:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Rho: {rho:.4f}")
    # Generate plots if possible
    try:
        plot_greeks(args.option_type, args.strike, args.rate, args.vol, args.t, args.spot)
        print("Greek plots saved to 'option_greeks.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()