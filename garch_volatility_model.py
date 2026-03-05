"""
GARCH(1,1) Volatility Modeling and Forecasting
---------------------------------------------

This script estimates a GARCH(1,1) model for the volatility of a single
asset's returns, forecasts future volatility, and visualizes the
conditional variance. GARCH models capture volatility clustering and
mean‑reversion, both of which are prevalent in financial time series.

Key features:

* **Data acquisition:** Fetch daily adjusted close prices for a user‑specified
  ticker (default: SPY) using ``yfinance``. If the download fails, use
  fallback sample prices from the S&P 500 historical dataset to construct
  synthetic returns.
* **Return computation:** Compute log returns from the price series and
  normalise by removing the mean.
* **Parameter estimation:** Fit a GARCH(1,1) model to the returns by
  maximizing the likelihood using ``scipy.optimize.minimize``. The
  parameters (omega, alpha, beta) satisfy the usual constraints (omega > 0,
  alpha >= 0, beta >= 0, alpha + beta < 1).
* **Conditional variance & forecasting:** Generate the conditional variance
  series given the estimated parameters and forecast the volatility for
  the next ``n_forecast`` days. Compute the unconditional variance and
  annualised average volatility.
* **Risk metrics:** Compute the one‑day Value at Risk (VaR) at the 95%
  confidence level using the predicted volatility and assume normally
  distributed returns.
* **Visualisation:** Plot the conditional variance and forecasted variance
  and save the figure to ``garch_volatility_model.png``.

Example usage:

    python3 garch_volatility_model.py --ticker SPY --start 2018-01-01 --end 2024-01-01 --forecast 20

Dependencies: numpy, pandas, scipy, matplotlib (optional), yfinance (optional).
"""

import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize  # type: ignore
except Exception:
    minimize = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices for a ticker using yfinance."""
    if yf is None:
        raise ImportError("yfinance is not available. Please install yfinance.")
    data = yf.download(ticker, start=start, end=end, progress=False)['Adj Close']
    return data.dropna()


def fallback_sample_prices() -> pd.Series:
    """Return a small sample of historical S&P 500 prices as a fallback."""
    # Sample closing prices extracted from early S&P 500 data (1927–1928)
    sample = np.array([
        17.66, 18.19, 17.87, 17.91, 17.83, 17.80, 17.88, 17.80, 17.75,
        17.73, 17.60, 17.60, 17.83, 17.68, 17.66, 17.77, 17.71, 17.75,
        17.72, 17.75, 17.78, 17.89, 18.00, 18.28, 18.33, 18.32, 18.43
    ])
    dates = pd.date_range('2020-01-01', periods=len(sample), freq='B')
    return pd.Series(sample, index=dates)


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    """Compute demeaned log returns from price series."""
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    # demean
    return log_returns - log_returns.mean()


def garch_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    """Return the negative log-likelihood of a GARCH(1,1) model."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
        return np.inf
    T = len(data)
    # Initialize conditional variance with the unconditional variance
    var = np.zeros(T)
    var[0] = np.var(data)
    for t in range(1, T):
        var[t] = omega + alpha * data[t - 1] ** 2 + beta * var[t - 1]
    # Compute log-likelihood assuming normal innovations
    ll = -0.5 * (np.log(2 * math.pi) + np.log(var) + (data ** 2) / var)
    return -np.sum(ll)


def fit_garch(data: np.ndarray) -> Tuple[float, float, float]:
    """Estimate GARCH(1,1) parameters by minimising the negative log-likelihood."""
    if minimize is None:
        # simple fallback parameters
        return 1e-6, 0.05, 0.9
    initial = np.array([1e-6, 0.1, 0.8])
    bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
    result = minimize(
        garch_likelihood, initial, args=(data,), bounds=bounds, method='L-BFGS-B'
    )
    omega, alpha, beta = result.x
    return float(omega), float(alpha), float(beta)


def compute_conditional_variance(data: np.ndarray, params: Tuple[float, float, float]) -> np.ndarray:
    """Compute the conditional variance series given GARCH parameters."""
    omega, alpha, beta = params
    T = len(data)
    var = np.zeros(T)
    var[0] = np.var(data)
    for t in range(1, T):
        var[t] = omega + alpha * data[t - 1] ** 2 + beta * var[t - 1]
    return var


def forecast_variance(last_var: float, params: Tuple[float, float, float], n: int) -> np.ndarray:
    """Forecast future conditional variances for n steps ahead."""
    omega, alpha, beta = params
    forecast = np.zeros(n)
    var = last_var
    for i in range(n):
        var = omega + (alpha + beta) * var
        forecast[i] = var
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description='GARCH(1,1) Volatility Modeling')
    parser.add_argument('--ticker', default='SPY', help='Ticker for data download (default: SPY)')
    parser.add_argument('--start', default='2018-01-01', help='Start date for data')
    parser.add_argument('--end', default='2024-01-01', help='End date for data')
    parser.add_argument('--forecast', type=int, default=20, help='Number of days to forecast')
    args = parser.parse_args()
    try:
        prices = download_prices(args.ticker, args.start, args.end)
        print(f"Downloaded {len(prices)} rows of price data for {args.ticker}.")
    except Exception as e:
        print(f"Warning: failed to download data due to {e}. Using fallback sample prices.")
        prices = fallback_sample_prices()
    # Compute log returns
    returns = compute_log_returns(prices)
    # Estimate GARCH parameters
    omega, alpha, beta = fit_garch(returns)
    print(f"Estimated parameters: omega={omega:.2e}, alpha={alpha:.4f}, beta={beta:.4f}")
    # Conditional variance and annualised volatility
    cond_var = compute_conditional_variance(returns, (omega, alpha, beta))
    cond_vol = np.sqrt(cond_var)
    avg_vol_ann = cond_vol.mean() * math.sqrt(252)
    print(f"Average annualised volatility: {avg_vol_ann:.2%}")
    # Forecast variance
    forecast_var = forecast_variance(cond_var[-1], (omega, alpha, beta), args.forecast)
    forecast_vol = np.sqrt(forecast_var)
    print(f"Forecast next {args.forecast} days volatility:")
    for i, vol in enumerate(forecast_vol, 1):
        print(f"  Day {i}: {vol * math.sqrt(252):.2%} annualised")
    # Compute 95% VaR using last variance
    var_1d = cond_vol[-1] * 1.96  # 1.96 * sigma for 95% quantile
    print(f"One-day 95% VaR (assuming normal returns): {var_1d:.2%}")
    # Plot conditional variance and forecast
    try:
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(prices.index[-len(cond_var):], cond_var, label='Conditional Variance')
            forecast_dates = pd.date_range(prices.index[-1], periods=args.forecast + 1, freq='B')[1:]
            ax.plot(forecast_dates, forecast_var, label='Forecast Variance', linestyle='--')
            ax.set_title('GARCH(1,1) Conditional Variance and Forecast')
            ax.set_ylabel('Variance')
            ax.legend()
            plt.tight_layout()
            fig.savefig('garch_volatility_model.png')
            plt.close(fig)
            print("Plot saved to 'garch_volatility_model.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()