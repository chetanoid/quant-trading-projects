"""
Cointegration Pairs Trading Strategy
------------------------------------

This script implements a simple cointegration based pairs trading strategy using
real historical price data. It demonstrates how to:

* Download daily adjusted close prices for two equities using ``yfinance``.
* Test whether the two price series are cointegrated using the
  Engle–Granger two‑step test via ``statsmodels.tsa.stattools.coint``.
* Construct a trading spread by regressing one price on the other and
  computing the deviation (spread) from the equilibrium relationship.
* Generate z‑scores of the spread and trigger trades when the spread
  deviates beyond a chosen threshold (opening a long/short position) and
  closing when the spread reverts.
* Evaluate the strategy by computing cumulative returns and plotting
  both the spread (with entry/exit points) and the equity curve. The
  plots are saved to ``pairs_spread.png`` and ``pairs_equity_curve.png``.

The script includes a fallback data generator to ensure it runs even if
``yfinance`` is unavailable or fails to download data (for example, when
offline). The fallback generates two correlated random walk price series.

Example usage:

    python3 cointegration_pairs_trading.py --symbols KO PEP --start 2020-01-01 --end 2024-01-01 --threshold 1.5

Dependencies: pandas, numpy, statsmodels, matplotlib, yfinance (optional).

"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

# Attempt to import optional dependencies. If unavailable, fallback later.
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

from statsmodels.tsa.stattools import coint  # type: ignore

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib.pyplot as plt


def download_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for the given symbols using yfinance.

    Parameters
    ----------
    symbols : List[str]
        List of ticker symbols.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    DataFrame
        DataFrame containing the adjusted close prices indexed by date.
    """
    if yf is None:
        raise ImportError(
            "yfinance is not available. Please install yfinance or rely on fallback data."
        )
    data = yf.download(symbols, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        # If only one symbol, convert to DataFrame
        data = data.to_frame(name=symbols[0])
    data = data.dropna()
    if data.empty:
        raise RuntimeError("Downloaded data is empty")
    return data


def generate_fallback_data(n: int = 300) -> pd.DataFrame:
    """Generate a fallback dataset of two correlated random walk price series.

    Parameters
    ----------
    n : int
        Number of observations.

    Returns
    -------
    DataFrame
        Two-column DataFrame with synthetic correlated price series.
    """
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    # Generate two correlated random walks
    # Start at 100 and 50
    step1 = rng.normal(0, 0.5, size=n)
    step2 = rng.normal(0, 0.5, size=n)
    # Introduce correlation by mixing the innovations
    step2 = 0.8 * step1 + 0.2 * step2
    price1 = 100 + np.cumsum(step1)
    price2 = 50 + np.cumsum(step2)
    df = pd.DataFrame({'SYM1': price1, 'SYM2': price2}, index=dates)
    return df


def compute_spread_params(y: pd.Series, x: pd.Series) -> Tuple[float, pd.Series, pd.Series]:
    """Compute the hedge ratio and spread using OLS regression.

    The hedge ratio is obtained by regressing y on x without an intercept.

    Parameters
    ----------
    y : Series
        Dependent variable price series.
    x : Series
        Independent variable price series.

    Returns
    -------
    Tuple[float, Series, Series]
        hedge_ratio, spread, zscore of spread
    """
    # Fit linear regression y = beta * x + epsilon
    # Compute beta by least squares
    beta = np.polyfit(x, y, 1)[0]
    spread = y - beta * x
    zscore = (spread - spread.mean()) / spread.std()
    return beta, spread, zscore


@dataclass
class TradeSignal:
    index: pd.Timestamp
    signal: str  # 'long' or 'short' or 'exit'


def generate_trade_signals(zscore: pd.Series, threshold: float) -> List[TradeSignal]:
    """Generate trading signals based on z-score thresholds.

    Parameters
    ----------
    zscore : Series
        Standardised spread series.
    threshold : float
        Threshold for opening positions.

    Returns
    -------
    List[TradeSignal]
        List of trading signals with timestamps.
    """
    signals = []
    position = 0  # 0 = no position, 1 = long spread (long y, short x), -1 = short spread (short y, long x)
    # pandas 2.0+ removed iteritems; use .items()
    for ts, z in zscore.items():
        if position == 0:
            if z > threshold:
                signals.append(TradeSignal(ts, 'short'))
                position = -1
            elif z < -threshold:
                signals.append(TradeSignal(ts, 'long'))
                position = 1
        elif position == 1:
            if z >= 0:
                signals.append(TradeSignal(ts, 'exit'))
                position = 0
        elif position == -1:
            if z <= 0:
                signals.append(TradeSignal(ts, 'exit'))
                position = 0
    return signals


def backtest_pairs(
    x: pd.Series,
    y: pd.Series,
    beta: float,
    zscore: pd.Series,
    threshold: float,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Simulate a simple pairs trading strategy based on z-score signals.

    The strategy invests an equal dollar amount in each leg when a signal is triggered.
    Positions are normalised so that the total notional equals 1 at entry. When the
    spread reverts (z-score crosses zero), the position is closed.

    Parameters
    ----------
    x : Series
        Independent variable price series.
    y : Series
        Dependent variable price series.
    beta : float
        Hedge ratio obtained from regression.
    zscore : Series
        Standardised spread series.
    threshold : float
        Entry threshold for opening trades.
    initial_capital : float
        Starting capital of the strategy.

    Returns
    -------
    DataFrame
        Equity curve, positions and daily returns.
    """
    # Determine entry and exit signals
    signals = generate_trade_signals(zscore, threshold)
    # Initialise tracking variables
    equity = initial_capital
    equity_curve = pd.Series(index=zscore.index, data=initial_capital, dtype=float)
    position = 0  # 1 for long spread, -1 for short spread
    entry_price_x = 0.0
    entry_price_y = 0.0
    for ts in zscore.index:
        # Check if a signal occurs at this timestamp
        for signal in [s for s in signals if s.index == ts]:
            if signal.signal == 'long' and position == 0:
                # Enter long spread: long y, short x*beta
                position = 1
                entry_price_x = x.loc[ts]
                entry_price_y = y.loc[ts]
            elif signal.signal == 'short' and position == 0:
                # Enter short spread: short y, long x*beta
                position = -1
                entry_price_x = x.loc[ts]
                entry_price_y = y.loc[ts]
            elif signal.signal == 'exit' and position != 0:
                # Close position
                if position == 1:
                    # Profit from long spread
                    dx = x.loc[ts] - entry_price_x
                    dy = y.loc[ts] - entry_price_y
                    pnl = -beta * dx + dy
                else:
                    dx = x.loc[ts] - entry_price_x
                    dy = y.loc[ts] - entry_price_y
                    pnl = beta * dx - dy
                equity += pnl
                position = 0
        # Update equity curve – mark to market existing position but not closing
        if position != 0:
            if position == 1:
                current_pnl = -beta * (x.loc[ts] - entry_price_x) + (y.loc[ts] - entry_price_y)
            else:
                current_pnl = beta * (x.loc[ts] - entry_price_x) - (y.loc[ts] - entry_price_y)
            equity_curve.loc[ts] = initial_capital + current_pnl
        else:
            equity_curve.loc[ts] = equity
    returns = equity_curve.pct_change().fillna(0.0)
    result = pd.DataFrame(
        {
            'equity_curve': equity_curve,
            'returns': returns,
        }
    )
    return result


def plot_results(
    spread: pd.Series,
    zscore: pd.Series,
    equity_curve: pd.Series,
    threshold: float,
    output_prefix: str = 'pairs'
) -> None:
    """Plot the spread with trading bands and the equity curve.

    Parameters
    ----------
    spread : Series
        Spread series.
    zscore : Series
        Z-score of the spread.
    equity_curve : Series
        Equity curve from the backtest.
    threshold : float
        Threshold used to generate bands on the z-score plot.
    output_prefix : str
        Prefix for output filenames.
    """
    # Plot the z-score with entry bands
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(zscore.index, zscore.values, label='Z-score of Spread')
    ax[0].axhline(threshold, color='red', linestyle='--', label=f'+{threshold}')
    ax[0].axhline(-threshold, color='green', linestyle='--', label=f'-{threshold}')
    ax[0].set_title('Spread Z-score and Trading Bands')
    ax[0].legend()
    # Plot the equity curve
    ax[1].plot(equity_curve.index, equity_curve.values, label='Equity Curve')
    ax[1].set_title('Pairs Trading Equity Curve')
    ax[1].set_ylabel('Equity (base units)')
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_trading_results.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Cointegration Pairs Trading Strategy')
    parser.add_argument('--symbols', nargs=2, metavar=('Y', 'X'), default=['KO', 'PEP'],
                        help='Ticker symbols for the pair (dependent and independent)')
    parser.add_argument('--start', default='2015-01-01', help='Start date for historical data')
    parser.add_argument('--end', default='2024-01-01', help='End date for historical data')
    parser.add_argument('--threshold', type=float, default=1.0, help='Z-score threshold for entry')
    args = parser.parse_args()
    symbols = args.symbols
    try:
        prices = download_prices(symbols, args.start, args.end)
        print(f"Downloaded {len(prices)} rows of price data for {symbols}.")
    except Exception as e:
        print(f"Warning: failed to download data due to {e}. Using fallback synthetic data.")
        prices = generate_fallback_data(n=300)
        # Rename columns to match expected symbols
        prices.columns = symbols
    # Extract series
    y = prices[symbols[0]]
    x = prices[symbols[1]]
    # Test for cointegration
    coint_t, pvalue, _ = coint(y, x)
    print(f"Cointegration test statistic: {coint_t:.4f}, p-value: {pvalue:.4f}")
    # Compute hedge ratio and spread
    beta, spread, zscore = compute_spread_params(y, x)
    print(f"Hedge ratio (beta): {beta:.4f}")
    # Backtest strategy
    result = backtest_pairs(x, y, beta, zscore, args.threshold)
    final_equity = result['equity_curve'].iloc[-1]
    total_return = (final_equity - 1.0) * 100.0
    print(f"Final equity: {final_equity:.4f} (Total return: {total_return:.2f}%)")
    # Plot results
    try:
        plot_results(spread, zscore, result['equity_curve'], args.threshold, output_prefix='pairs')
        print("Plots saved to 'pairs_trading_results.png'.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
