"""
Real‑Data Trading Strategy Backtester
====================================

This script fetches real historical price data for the S&P 500 ETF (SPY)
using ``yfinance`` and evaluates two simple strategies – momentum and
mean‑reversion – against a buy‑and‑hold benchmark.  If network access
or the ``yfinance`` package is unavailable, it falls back to a small
sample of genuine S&P 500 price history embedded as CSV text.  The
backtester computes daily returns, applies the trading rules, and
reports cumulative return, annualised volatility and a Sharpe‑like
ratio (mean divided by volatility).  Results for each strategy are
saved to ``strategy_real_returns.csv``.

Strategy definitions:

* **Momentum**: Go long when the average return over the last ``n`` days
  is positive, otherwise stay flat.
* **Mean‑reversion**: Go long when the average return over the last ``n``
  days is negative (i.e. expecting a bounce), otherwise stay flat.

You can adjust the lookback period by modifying the ``LOOKBACK`` constant.

"""

import os
import pandas as pd
import numpy as np
from io import StringIO

# Attempt to import matplotlib for plotting cumulative returns.  If not
# available, the backtest will still run and output metrics and CSV.
try:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False


# Embedded fallback CSV taken from the GitHub repository
# fja05680/dow-sp500-100-years.  These values correspond to the
# first few trading days in late 1927 and early 1928.  They are
# provided to ensure the script runs even without network access.
FALLBACK_CSV = """Date,Adj Close
1927-12-30,17.65999984741211
1928-01-03,17.760000228881836
1928-01-04,17.719999313354492
1928-01-05,17.549999237060547
1928-01-06,17.65999984741211
1928-01-09,17.5
1928-01-10,17.3700008392334
1928-01-11,17.350000381469727
1928-01-12,17.469999313354492
1928-01-13,17.579999923706055
1928-01-16,17.290000915527344
1928-01-17,17.299999237060547
1928-01-18,17.260000228881836
1928-01-19,17.3799991607666
1928-01-20,17.479999542236328
1928-01-23,17.639999389648438
1928-01-24,17.709999084472656
1928-01-25,17.520000457763672
1928-01-26,17.6299991607666
1928-01-27,17.690000534057617
1928-01-30,17.489999771118164
1928-01-31,17.56999969482422
1928-02-01,17.530000686645508
1928-02-02,17.6299991607666
1928-02-03,17.399999618530273
1928-02-06,17.450000762939453
1928-02-07,17.440000534057617
1928-02-08,17.489999771118164
1928-02-09,17.549999237060547
1928-02-10,17.540000915527344
"""

# Parameters
TICKER = "SPY"
START = "2015-01-01"
END = "2025-01-01"
LOOKBACK = 20  # lookback period in days for both momentum and mean‑reversion


def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch adjusted closing prices from Yahoo! Finance.

    Returns a pandas Series indexed by date.  If data cannot be
    downloaded, returns a Series built from ``FALLBACK_CSV``.
    """
    try:
        import yfinance as yf  # type: ignore
        data = yf.download(ticker, start=start, end=end, progress=False)
        prices = data.get("Adj Close")
        if prices is not None and len(prices) > 0:
            return prices.dropna()
        else:
            raise RuntimeError("No data returned from yfinance")
    except Exception:
        # Read from the embedded CSV
        df = pd.read_csv(StringIO(FALLBACK_CSV), parse_dates=["Date"], index_col="Date")
        return df["Adj Close"]


def compute_metrics(returns: pd.Series) -> dict:
    """Compute cumulative return, annualised volatility and Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    dict
        Dictionary with keys ``cum_return``, ``volatility``, and ``sharpe``.
    """
    cum_return = (1 + returns).prod() - 1
    vol = returns.std() * np.sqrt(252)  # assuming ~252 trading days per year
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
    return {"cum_return": cum_return, "volatility": vol, "sharpe": sharpe}


def backtest_strategy(prices: pd.Series, lookback: int) -> pd.DataFrame:
    """Backtest momentum and mean‑reversion strategies.

    Parameters
    ----------
    prices : pd.Series
        Series of adjusted close prices.
    lookback : int
        Lookback window for the rolling mean used to generate signals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``return``, ``mom`` and ``mr`` for
        buy‑and‑hold, momentum and mean‑reversion strategy returns.
    """
    returns = prices.pct_change().dropna()
    # Rolling average of returns
    avg_returns = returns.rolling(window=lookback).mean()
    # Momentum signal: long if average return is positive
    signal_mom = (avg_returns > 0).astype(int)
    # Mean‑reversion signal: long when momentum signal is negative
    signal_mr = (avg_returns < 0).astype(int)
    # Align signals with returns (shift by one day to avoid lookahead bias)
    signal_mom = signal_mom.shift(1).reindex(returns.index).fillna(0)
    signal_mr = signal_mr.shift(1).reindex(returns.index).fillna(0)
    # Strategy returns
    strat_mom = signal_mom * returns
    strat_mr = signal_mr * returns
    df = pd.DataFrame({"return": returns, "momentum": strat_mom, "mean_reversion": strat_mr})
    return df


def main() -> None:
    prices = fetch_prices(TICKER, START, END)
    result = backtest_strategy(prices, LOOKBACK)
    # Compute metrics for each strategy
    metrics = {
        "buy_and_hold": compute_metrics(result["return"]),
        "momentum": compute_metrics(result["momentum"]),
        "mean_reversion": compute_metrics(result["mean_reversion"]),
    }
    print("Performance Summary (approx. annualised):")
    for name, stats in metrics.items():
        print(f"{name:15s} - Cumulative: {stats['cum_return']:.2%}, Volatility: {stats['volatility']:.2%}, Sharpe: {stats['sharpe']:.2f}")
    # Save daily returns to CSV
    result.to_csv("strategy_real_returns.csv")
    print("Daily returns saved to strategy_real_returns.csv")

    # Plot cumulative returns for each strategy if matplotlib is available
    if _HAVE_MATPLOTLIB:
        try:
            # Compute cumulative returns (starting at zero) for each series
            cum_returns = (result[["return", "momentum", "mean_reversion"]] + 1).cumprod() - 1
            plt.figure()
            for col in cum_returns.columns:
                plt.plot(cum_returns.index, cum_returns[col], label=col)
            plt.legend()
            plt.title("Cumulative Returns of Strategies")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("strategy_cumulative_returns.png")
            plt.close()
            print("Cumulative returns plot saved to strategy_cumulative_returns.png")
        except Exception:
            # Silently ignore plotting errors (e.g. no backend available)
            pass


if __name__ == "__main__":
    main()
