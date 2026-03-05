"""
factor_model_regression.py
--------------------------

This script demonstrates how to run a multi‑factor regression of an asset’s
returns on common risk factors.  In quantitative finance, factor models
such as the Fama–French three‑factor model decompose returns into market,
size and value components.  Here we attempt to download daily prices for a
user‑specified asset via `yfinance` and compute percentage returns.  We
either fetch Fama–French factor returns using `pandas_datareader` or,
if that fails, generate synthetic factor data that loosely resembles
market, SMB (small minus big) and HML (high minus low) factors.  The
regression is performed using `statsmodels` and the results are printed
to the console.  A bar chart of the estimated factor exposures (betas) is
saved to `factor_exposures.png`.

Usage:
    python3 factor_model_regression.py --ticker AAPL --start 2019-01-01 --end 2024-01-01

The script accepts optional arguments for the ticker symbol and start/end
dates.  It automatically aligns the asset returns with the factor returns
by using the intersection of their date indices.
"""

import argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None  # network or module unavailable

try:
    import pandas_datareader.data as web
except ImportError:
    web = None  # datareader may not be installed

import statsmodels.api as sm


def fetch_asset_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily returns for a given ticker using yfinance.  If unable to
    download data, generate synthetic returns as a fallback.

    Returns
    -------
    pandas.Series
        Series of daily percentage returns indexed by date.
    """
    try:
        if yf is not None:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                returns = data['Adj Close'].pct_change().dropna()
                return returns
    except Exception:
        pass
    # fallback: generate synthetic returns (mean 0.05%, std 1%)
    date_range = pd.bdate_range(start=start, end=end)
    np.random.seed(0)
    synthetic = np.random.normal(loc=0.0005, scale=0.01, size=len(date_range))
    return pd.Series(synthetic, index=date_range)


def fetch_factor_returns(start: str, end: str) -> pd.DataFrame:
    """Fetch Fama–French factor returns via pandas_datareader or generate
    synthetic factors if the download fails.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['Mkt-RF', 'SMB', 'HML'] representing factor
        returns (in decimal form) and an index of dates.
    """
    # Attempt to fetch Fama–French factors from the French Data Library via datareader
    if web is not None:
        try:
            ff_data = web.DataReader('F-F_Research_Data_Factors_Daily', 'famafrench', start, end)
            factors = ff_data[0] / 100.0  # Convert from percent to decimal
            factors = factors.rename(columns={'Mkt-RF': 'Mkt_RF'})
            # Add risk free rate to market factor to compute raw market returns
            factors['Mkt'] = factors['Mkt_RF'] + factors['RF']
            return factors[['Mkt', 'SMB', 'HML']]
        except Exception:
            pass
    # fallback: generate synthetic factor returns (market ~1%, SMB ~0.2%, HML ~0.3%)
    date_range = pd.bdate_range(start=start, end=end)
    np.random.seed(1)
    market = np.random.normal(loc=0.0008, scale=0.009, size=len(date_range))
    smb = np.random.normal(loc=0.0002, scale=0.004, size=len(date_range))
    hml = np.random.normal(loc=0.0003, scale=0.005, size=len(date_range))
    factors = pd.DataFrame({'Mkt': market, 'SMB': smb, 'HML': hml}, index=date_range)
    return factors


def run_factor_regression(asset_returns: pd.Series, factors: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Align asset and factor returns and run an OLS regression.

    Returns
    -------
    RegressionResultsWrapper
        The fitted OLS model.
    """
    # Align on intersection of dates
    joined = pd.concat([asset_returns, factors], axis=1, join='inner').dropna()
    y = joined.iloc[:, 0]
    X = joined.iloc[:, 1:]
    X = sm.add_constant(X)  # add intercept
    model = sm.OLS(y, X).fit()
    return model


def save_factor_exposures_plot(model, filename: str = 'factor_exposures.png') -> None:
    """Plot factor exposures and save to file."""
    params = model.params.drop('const', errors='ignore')
    exposures = params.values
    factors = params.index
    plt.figure(figsize=(6, 4))
    plt.bar(factors, exposures)
    plt.title('Estimated Factor Exposures')
    plt.xlabel('Factor')
    plt.ylabel('Beta')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-factor regression on asset returns.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol for the asset')
    parser.add_argument('--start', type=str, default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=dt.date.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    # Fetch asset returns
    asset_returns = fetch_asset_returns(args.ticker, args.start, args.end)
    # Fetch factor returns
    factors = fetch_factor_returns(args.start, args.end)
    # Run regression
    model = run_factor_regression(asset_returns, factors)
    # Print summary
    print(model.summary())
    # Save exposures plot
    save_factor_exposures_plot(model)
    print("Factor exposures plot saved to 'factor_exposures.png'.")


if __name__ == '__main__':
    main()