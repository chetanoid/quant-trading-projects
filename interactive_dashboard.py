"""
Interactive Quantitative Trading Dashboard
========================================

This script creates an interactive dashboard summarising key metrics from
several of the quantitative trading projects in this repository.  It
demonstrates an ability to synthesise results across multiple models and
presents them in a visually engaging format that is easy to explore.

The dashboard is implemented using Plotly, which is installed in this
environment.  It generates a single HTML file (`dashboard.html`) containing
two interactive plots:

1. **Cumulative returns** for three simple strategies (buy‐and‐hold,
   momentum and mean‐reversion) computed from the `strategy_returns.csv`
   file produced by the `real_data_strategy_backtest.py` script.  The
   cumulative returns are calculated by compounding the daily returns.

2. **Efficient frontier** constructed from synthetic asset data.  The
   portfolio optimisation uses a Monte Carlo approach to generate random
   weight allocations across a set of five assets, then computes the
   annualised return, volatility and Sharpe ratio for each portfolio.  The
   script identifies and highlights the portfolios with the maximum
   Sharpe ratio and the minimum volatility.  Generating the asset data
   internally (instead of downloading live prices) ensures the dashboard
   can run without external dependencies.

To run this script and generate the dashboard, execute:

```
python interactive_dashboard.py
```

The resulting `dashboard.html` file will appear in the current
directory.  Open it with a web browser to explore the charts.

Note: if `strategy_returns.csv` is not present, the cumulative return
chart will still display, but all series will be flat at zero.  This
fallback ensures the dashboard does not crash if the CSV is missing.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

RETURN_FILE_CANDIDATES = ("strategy_real_returns.csv", "strategy_returns.csv")


def _build_placeholder_returns(n_points: int = 100) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "plot_index": np.arange(n_points),
            "bh_returns": np.zeros(n_points),
            "momentum_returns": np.zeros(n_points),
            "mean_reversion_returns": np.zeros(n_points),
        }
    )
    df["cum_bh"] = 0.0
    df["cum_momentum"] = 0.0
    df["cum_mean_rev"] = 0.0
    return df


def _detect_plot_index(df: pd.DataFrame) -> pd.Series:
    for candidate in ("date", "Date"):
        if candidate in df.columns:
            series = df[candidate]
            if series.dtype == object:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().all():
                    return parsed
            return series
    return pd.Series(np.arange(len(df)), name="plot_index")


def _normalise_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    alias_map = {
        "return": "bh_returns",
        "buy_and_hold": "bh_returns",
        "bh_returns": "bh_returns",
        "momentum": "momentum_returns",
        "momentum_returns": "momentum_returns",
        "mean_reversion": "mean_reversion_returns",
        "mean_reversion_returns": "mean_reversion_returns",
    }

    for original, canonical in alias_map.items():
        if original in df.columns and canonical not in df.columns:
            df[canonical] = df[original]

    for column in ("bh_returns", "momentum_returns", "mean_reversion_returns"):
        if column not in df.columns:
            df[column] = 0.0

    df["plot_index"] = _detect_plot_index(df)
    df[["bh_returns", "momentum_returns", "mean_reversion_returns"]] = df[
        ["bh_returns", "momentum_returns", "mean_reversion_returns"]
    ].fillna(0.0)
    df["cum_bh"] = (1.0 + df["bh_returns"]).cumprod() - 1.0
    df["cum_momentum"] = (1.0 + df["momentum_returns"]).cumprod() - 1.0
    df["cum_mean_rev"] = (1.0 + df["mean_reversion_returns"]).cumprod() - 1.0
    return df


def load_strategy_returns(base_dir: str) -> tuple[pd.DataFrame, str | None]:
    """Load whichever strategy return CSV is available first."""
    for filename in RETURN_FILE_CANDIDATES:
        candidate = os.path.join(base_dir, filename)
        if os.path.isfile(candidate):
            return _normalise_strategy_returns(pd.read_csv(candidate)), candidate
    return _build_placeholder_returns(), None

def generate_synthetic_prices(n_assets: int = 5, n_days: int = 252*2) -> pd.DataFrame:
    """Generate synthetic asset price data using geometric Brownian motion.

    Parameters
    ----------
    n_assets : int
        Number of assets to simulate.
    n_days : int
        Number of daily observations to simulate.

    Returns
    -------
    DataFrame
        Synthetic daily prices for each asset.
    """
    np.random.seed(42)
    # Simulate daily log returns for each asset
    # Choose random means and volatilities per asset
    mus = np.random.uniform(0.05, 0.15, size=n_assets) / 252
    sigmas = np.random.uniform(0.15, 0.30, size=n_assets) / np.sqrt(252)
    # Generate correlated random walks (correlation via Cholesky)
    corr_matrix = np.eye(n_assets)
    # Introduce mild correlations
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.2, 0.5)
    L = np.linalg.cholesky(corr_matrix)
    # Simulate log returns matrix
    z = np.random.normal(size=(n_days, n_assets))
    correlated_z = z @ L.T
    log_returns = mus + sigmas * correlated_z
    # Convert to price series
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    columns = [f'Asset_{i + 1}' for i in range(n_assets)]
    return pd.DataFrame(prices, columns=columns)

def monte_carlo_portfolios(prices: pd.DataFrame, num_portfolios: int = 5000):
    """Generate random portfolios and compute return, volatility and Sharpe ratio.

    Parameters
    ----------
    prices : DataFrame
        Daily price series for assets.
    num_portfolios : int
        Number of random portfolios to generate.

    Returns
    -------
    DataFrame
        A DataFrame with columns: 'return', 'volatility', 'sharpe', and
        one column per asset representing the weight allocation.
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252  # Annualised
    cov_matrix = returns.cov() * 252
    n_assets = len(mean_returns)
    results = []
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_return = float(np.dot(weights, mean_returns))
        port_volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
        sharpe = port_return / port_volatility if port_volatility != 0 else 0.0
        results.append([port_return, port_volatility, sharpe] + list(weights))
    columns = ['return', 'volatility', 'sharpe'] + list(prices.columns)
    return pd.DataFrame(results, columns=columns)

def build_dashboard():
    """Assemble and save the interactive dashboard.

    This function loads strategy returns, generates synthetic asset prices,
    computes random portfolios and constructs a three‑panel interactive
    dashboard.  The panels display cumulative returns, the efficient
    frontier and an asset correlation heatmap.  The final dashboard is
    written to `dashboard.html` in the current directory.
    """
    base_dir = os.path.dirname(__file__)
    strat_df, source_file = load_strategy_returns(base_dir)

    # Generate synthetic prices and portfolios
    prices = generate_synthetic_prices(n_assets=5, n_days=252 * 2)
    portfolios = monte_carlo_portfolios(prices, num_portfolios=3000)
    # Identify maximum Sharpe ratio and minimum volatility portfolios
    max_sharpe = portfolios.loc[portfolios['sharpe'].idxmax()]
    min_vol = portfolios.loc[portfolios['volatility'].idxmin()]

    # Compute correlation matrix of returns for heatmap
    returns_matrix = prices.pct_change().dropna()
    corr = returns_matrix.corr()

    # Compute summary metrics for each strategy
    # We calculate annualised return, annualised volatility, Sharpe ratio (risk‑free rate = 0)
    # and maximum drawdown for buy‑and‑hold, momentum and mean reversion strategies.
    metrics = {}
    # Map internal column names to user friendly labels
    strategy_map = {
        'bh_returns': 'Buy & Hold',
        'momentum_returns': 'Momentum',
        'mean_reversion_returns': 'Mean Reversion'
    }
    # Compute metrics
    for col, name in strategy_map.items():
        series = strat_df[col]
        mean_daily = series.mean()
        vol_daily = series.std()
        annual_return = float(mean_daily * 252)
        annual_vol = float(vol_daily * np.sqrt(252))
        sharpe_ratio = (annual_return / annual_vol) if annual_vol != 0 else 0.0
        # Compute cumulative returns for drawdown calculation
        cumulative = (1.0 + series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())
        metrics[name] = {
            'Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    # Prepare data for table
    table_header = ['Strategy', 'Ann. Return', 'Ann. Volatility', 'Sharpe', 'Max Drawdown']
    # Strategy order for display
    order = ['Buy & Hold', 'Momentum', 'Mean Reversion']
    table_values = [
        order,
        [round(metrics[s]['Return'], 4) for s in order],
        [round(metrics[s]['Volatility'], 4) for s in order],
        [round(metrics[s]['Sharpe'], 4) for s in order],
        [round(metrics[s]['Max Drawdown'], 4) for s in order]
    ]

    # Create a four‑row subplot: returns, efficient frontier, correlation heatmap, metrics table
    fig = make_subplots(rows=4, cols=1, subplot_titles=(
        'Cumulative Returns of Strategies',
        'Efficient Frontier (Synthetic Data)',
        'Asset Return Correlation Heatmap',
        'Strategy Performance Summary'
    ), vertical_spacing=0.20, specs=[[{}], [{}], [{}], [{'type': 'table'}]])

    # Plot cumulative returns
    fig.add_trace(go.Scatter(x=strat_df['plot_index'],
                             y=strat_df['cum_bh'],
                             name='Buy and Hold',
                             mode='lines'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=strat_df['plot_index'],
                             y=strat_df['cum_momentum'],
                             name='Momentum Strategy',
                             mode='lines'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=strat_df['plot_index'],
                             y=strat_df['cum_mean_rev'],
                             name='Mean Reversion Strategy',
                             mode='lines'),
                  row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Return', row=1, col=1)
    fig.update_xaxes(title_text='Observation', row=1, col=1)

    # Plot efficient frontier scatter
    fig.add_trace(go.Scatter(
        x=portfolios['volatility'],
        y=portfolios['return'],
        mode='markers',
        marker=dict(color=portfolios['sharpe'], colorscale='Viridis', colorbar=dict(title='Sharpe'),
                    size=4, opacity=0.7),
        name='Portfolios'
    ), row=2, col=1)
    # Highlight maximum Sharpe ratio
    fig.add_trace(go.Scatter(
        x=[max_sharpe['volatility']],
        y=[max_sharpe['return']],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Max Sharpe'
    ), row=2, col=1)
    # Highlight minimum volatility
    fig.add_trace(go.Scatter(
        x=[min_vol['volatility']],
        y=[min_vol['return']],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='star'),
        name='Min Volatility'
    ), row=2, col=1)
    fig.update_xaxes(title_text='Volatility (σ)', row=2, col=1)
    fig.update_yaxes(title_text='Return (μ)', row=2, col=1)

    # Plot correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation')
        ), row=3, col=1
    )
    fig.update_xaxes(tickangle=45, row=3, col=1)
    fig.update_yaxes(autorange='reversed', row=3, col=1)

    # Add table for strategy performance metrics
    fig.add_trace(
        go.Table(
            header=dict(values=table_header, fill_color='lightgrey', align='center'),
            cells=dict(values=table_values, align='center')
        ), row=4, col=1
    )

    title_text = 'Interactive Quantitative Trading Dashboard'
    if source_file is not None:
        title_text += f"<br><sup>Strategy data: {os.path.basename(source_file)}</sup>"
    else:
        title_text += "<br><sup>No strategy CSV found; using placeholder data.</sup>"

    fig.update_layout(height=1500, width=1000,
                      title_text=title_text,
                      showlegend=True)
    # Write the HTML file
    output_path = os.path.join(base_dir, 'dashboard.html')
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Dashboard saved to {output_path}")

if __name__ == '__main__':
    build_dashboard()
