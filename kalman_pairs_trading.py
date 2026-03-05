"""
Kalman Filter Pairs Trading
---------------------------

This script implements a dynamic pairs trading strategy using a
Kalman filter to estimate the time‑varying hedge ratio between two
co‑integrated assets. Unlike a static regression or cointegration
analysis, the Kalman filter updates the hedge ratio (beta) each day
based on new price data, allowing it to adapt to changing market
conditions. The residual between the predicted and actual price of
the dependent asset is treated as the spread; trading signals are
generated when the spread deviates sufficiently from its mean.

Key features:

* **Data acquisition with fallback:** Downloads historical adjusted
  closing prices for two user‑specified tickers using ``yfinance``.
  If the download fails (e.g. no network), synthetic correlated
  prices are generated to illustrate the algorithm.
* **Kalman filter regression:** Implements a simple state‑space model
  where the hedge ratio follows a random walk. The filter computes
  the predicted beta and its covariance on each step.
* **Signal generation:** Calculates the spread (residual) and its
  z‑score using a rolling window. Trades are opened when the z‑score
  exceeds an entry threshold and closed when it reverts within an
  exit threshold. Positions are sized to dollar neutrality based on
  the time‑varying hedge ratio.
* **Performance metrics:** Computes cumulative returns of the
  strategy, counts trades and reports final equity and total return.
* **Visualisation:** Generates plots of the evolving hedge ratio,
  z‑score with thresholds and the cumulative equity curve. All plots
  are saved in a single figure ``kalman_pairs_trading_results.png``.

Example usage:

    python3 kalman_pairs_trading.py --y_ticker SPY --x_ticker QQQ \
        --start 2015-01-01 --end 2024-01-01 --entry 2.0 --exit 0.5

Dependencies: numpy, pandas, matplotlib (optional), yfinance (optional).
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def download_two_assets(y_ticker: str, x_ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for two tickers using yfinance."""
    if yf is None:
        raise ImportError("yfinance is not available.")
    data = yf.download([y_ticker, x_ticker], start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        # If only one ticker returned (unlikely), convert to DataFrame
        data = data.to_frame(name=y_ticker)
    data = data.dropna()
    data.columns = [y_ticker, x_ticker]
    return data


def generate_fallback_prices(n: int) -> Tuple[pd.Series, pd.Series]:
    """Generate synthetic correlated price series for demonstration purposes."""
    rng = np.random.default_rng(seed=2026)
    # Simulate correlated log returns
    mean = np.array([0.0004, 0.0003])
    cov = np.array([[0.0004 ** 2, 0.0004 * 0.0003 * 0.8],
                    [0.0004 * 0.0003 * 0.8, 0.0003 ** 2]])
    returns = rng.multivariate_normal(mean, cov, size=n)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    prices = pd.DataFrame(returns, index=dates, columns=['Y', 'X'])
    prices = (1 + prices).cumprod()
    return prices['Y'], prices['X']


def kalman_filter(y: np.ndarray, x: np.ndarray, q: float, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a one‑dimensional Kalman filter to estimate the time‑varying
    hedge ratio between y and x. The model assumes:

    beta_t = beta_{t-1} + w_t,   w_t ~ N(0, q)
    y_t = beta_t * x_t + v_t,    v_t ~ N(0, r)

    Returns arrays of estimated beta, prediction residuals and
    prediction error variance.
    """
    n = len(y)
    beta = np.zeros(n)
    residuals = np.zeros(n)
    beta_var = np.zeros(n)
    # Initialise beta and variance
    beta[0] = 0.0
    P = 1.0
    for t in range(1, n):
        # Prediction step
        beta_pred = beta[t - 1]
        P_pred = P + q
        # Observation
        x_t = x[t]
        y_t = y[t]
        # Predicted observation and residual
        y_pred = beta_pred * x_t
        e = y_t - y_pred
        # Innovation variance
        S = x_t ** 2 * P_pred + r
        # Kalman gain
        K = P_pred * x_t / S
        # Update step
        beta[t] = beta_pred + K * e
        P = (1 - K * x_t) * P_pred
        # Store residual and variance
        residuals[t] = e
        beta_var[t] = P
    return beta, residuals, beta_var


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z‑score of a series."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=0)
    z = (series - rolling_mean) / rolling_std
    return z


def pairs_trading_strategy(
    y_prices: pd.Series,
    x_prices: pd.Series,
    entry: float,
    exit: float,
    q: float = 1e-5,
    r: float = 1e-3,
    window: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series, float, int]:
    """
    Execute a pairs trading strategy using a Kalman filter for hedge ratio.

    Parameters
    ----------
    y_prices : pd.Series
        Prices of the dependent asset.
    x_prices : pd.Series
        Prices of the independent asset.
    entry : float
        Z‑score threshold at which to open a trade.
    exit : float
        Z‑score threshold at which to close a trade.
    q : float
        Process noise variance for Kalman filter.
    r : float
        Measurement noise variance for Kalman filter.
    window : int
        Rolling window for z‑score calculation.

    Returns
    -------
    equity_curve : pd.Series
        Cumulative returns of the strategy (starting at 1).
    beta_series : pd.Series
        Estimated hedge ratio over time.
    zscore_series : pd.Series
        Z‑score of the spread (residual) over time.
    final_return : float
        Final return (in percent) of the strategy.
    num_trades : int
        Number of completed trades.
    """
    y = y_prices.values
    x = x_prices.values
    n = len(y)
    beta, residuals, _ = kalman_filter(y, x, q, r)
    spread = pd.Series(residuals, index=y_prices.index)
    zscores = compute_zscore(spread, window)
    # Trading variables
    pos = 0  # 0: flat, 1: short spread, -1: long spread
    equity = [1.0]
    num_trades = 0
    prev_beta = beta[0]
    # Compute returns of underlying assets
    ret_y = y_prices.pct_change().fillna(0)
    ret_x = x_prices.pct_change().fillna(0)
    for t in range(1, n):
        if np.isnan(zscores.iloc[t]):
            equity.append(equity[-1])
            continue
        z = zscores.iloc[t]
        b = beta[t]
        # Trading logic
        if pos == 0:
            if z > entry:
                pos = 1  # short spread: short y, long b * x
                num_trades += 1
            elif z < -entry:
                pos = -1  # long spread: long y, short b * x
                num_trades += 1
        elif pos == 1:  # short spread
            if z < exit:
                pos = 0
        elif pos == -1:  # long spread
            if z > -exit:
                pos = 0
        # Compute daily return
        pnl = pos * (-ret_y.iloc[t] + b * ret_x.iloc[t])
        equity.append(equity[-1] * (1 + pnl))
    equity_curve = pd.Series(equity, index=y_prices.index)
    final_return = (equity_curve.iloc[-1] - 1.0) * 100
    beta_series = pd.Series(beta, index=y_prices.index)
    return equity_curve, beta_series, zscores, final_return, num_trades


def plot_results(
    beta_series: pd.Series,
    zscores: pd.Series,
    entry: float,
    exit: float,
    equity_curve: pd.Series
) -> None:
    """Plot hedge ratio, z‑score and equity curve and save to file."""
    if plt is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # Hedge ratio
    axes[0].plot(beta_series.index, beta_series, label='Hedge Ratio (Beta)')
    axes[0].set_title('Kalman Filter Hedge Ratio')
    axes[0].set_ylabel('Beta')
    axes[0].legend()
    # Z‑score and thresholds
    axes[1].plot(zscores.index, zscores, label='Spread Z‑Score')
    axes[1].axhline(entry, color='r', linestyle='--', linewidth=1, label='Entry')
    axes[1].axhline(-entry, color='r', linestyle='--')
    axes[1].axhline(exit, color='g', linestyle=':', linewidth=1, label='Exit')
    axes[1].axhline(-exit, color='g', linestyle=':')
    axes[1].set_title('Spread Z‑Score and Trading Thresholds')
    axes[1].set_ylabel('Z‑Score')
    axes[1].legend()
    # Equity curve
    axes[2].plot(equity_curve.index, equity_curve, label='Equity Curve')
    axes[2].set_title('Cumulative Equity')
    axes[2].set_ylabel('Equity')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    plt.tight_layout()
    fig.savefig('kalman_pairs_trading_results.png')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Kalman Filter Pairs Trading')
    parser.add_argument('--y_ticker', default='SPY', help='Dependent asset ticker (Y)')
    parser.add_argument('--x_ticker', default='QQQ', help='Independent asset ticker (X)')
    parser.add_argument('--start', default='2018-01-01', help='Start date')
    parser.add_argument('--end', default='2024-01-01', help='End date')
    parser.add_argument('--entry', type=float, default=2.0, help='Entry z‑score threshold')
    parser.add_argument('--exit', type=float, default=0.5, help='Exit z‑score threshold')
    parser.add_argument('--q', type=float, default=1e-5, help='Process noise variance for Kalman filter')
    parser.add_argument('--r', type=float, default=1e-3, help='Measurement noise variance for Kalman filter')
    args = parser.parse_args()
    try:
        prices = download_two_assets(args.y_ticker, args.x_ticker, args.start, args.end)
        y_prices = prices[args.y_ticker]
        x_prices = prices[args.x_ticker]
        print(f"Downloaded {len(prices)} rows of price data for {[args.y_ticker, args.x_ticker]}.")
    except Exception as e:
        print(f"Warning: failed to download data due to {e}. Using synthetic data.")
        y_prices, x_prices = generate_fallback_prices(n=500)
    equity_curve, beta_series, zscores, final_return, trades = pairs_trading_strategy(
        y_prices, x_prices, args.entry, args.exit, args.q, args.r
    )
    print(f"Final return: {final_return:.2f}% with {trades} trades.")
    # Save plots if possible
    try:
        plot_results(beta_series, zscores, args.entry, args.exit, equity_curve)
        print("Results plot saved to 'kalman_pairs_trading_results.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()