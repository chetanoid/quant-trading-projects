# Quantitative Trading Projects

This portfolio contains 24 quantitative trading and finance projects spanning market microstructure, statistical arbitrage, derivatives pricing, portfolio construction, risk management, machine learning, and trading infrastructure. The repo is built to show both range and depth: research code in Python, numerical modeling, real-data workflows with offline fallbacks, and a systems-oriented C++ matching engine.

## Quick Links

- Public portfolio page: `portfolio.html`
- Interactive dashboard: `dashboard.html`
- Project catalog: `PROJECT_CATALOG.md`
- Flagship shortlist: `FLAGSHIP_PROJECTS.md`
- Project bullet library: `RESUME_PROJECT_BULLETS.md`

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python interactive_dashboard.py
```

Most scripts run directly with `python <script_name>.py`. Many projects try live market data first and then fall back to embedded or synthetic data so the repository remains runnable in a fresh clone.

## Best Files To Open First

1. `high_frequency_limit_order_book.cpp`
2. `avellaneda_stoikov_market_maker.py`
3. `kalman_pairs_trading.py`
4. `rough_bergomi_model.py`
5. `optimal_execution_almgren_chriss.py`
6. `portfolio_optimization.py`

## Market Microstructure And Execution

### 1. Limit Order Book And Market Making Simulator (`limit_order_book_simulator.py`)

This script demonstrates a basic event-driven limit order book with a simple inventory-aware market maker. It is a good first file for understanding the repo's market microstructure layer because it keeps the mechanics readable.

- Inputs: synthetic mid-price evolution, random market-order arrivals, spread, lot size, and inventory parameters
- Outputs: console summary plus `limit_order_book_results.txt`
- Run: `python limit_order_book_simulator.py`

### 2. Real-Data Market Making Simulator (`real_data_market_maker.py`)

This simulation uses real or fallback market prices to drive a market-making strategy. It tracks quotes, fills, inventory, and mark-to-market P&L over time, then saves diagnostic plots so the strategy behavior is visible quickly.

- Inputs: SPY prices from `yfinance` or embedded fallback data
- Outputs: console performance summary, `market_maker_pnl.png`, `market_maker_inventory.png`
- Run: `python real_data_market_maker.py`

### 3. Avellaneda-Stoikov Market-Making Model (`avellaneda_stoikov_market_maker.py`)

This project implements the classic inventory-aware market-making framework where quotes adapt to volatility, time horizon, and current inventory. It is one of the strongest trading-model files in the repository because it connects mathematical structure directly to trading behavior.

- Inputs: model parameters such as `gamma`, `k`, `sigma`, horizon `T`, and time step `dt`
- Outputs: simulation summary and `avellaneda_stoikov_results.png`
- Run: `python avellaneda_stoikov_market_maker.py`

### 4. Almgren-Chriss Optimal Execution (`optimal_execution_almgren_chriss.py`)

This model studies the optimal schedule for liquidating a large order while balancing market impact and execution risk. It demonstrates execution thinking rather than only alpha generation.

- Inputs: total shares, time horizon, volatility, permanent impact, temporary impact, and risk aversion
- Outputs: `ac_optimal_schedule.png`, `ac_shortfall_distribution.png`, console shortfall statistics
- Run: `python optimal_execution_almgren_chriss.py`

### 5. High-Frequency Limit Order Book Engine (`high_frequency_limit_order_book.cpp`)

This C++ project is a compact matching engine built around order-book state, priority queues, and concurrent order producers. It is the strongest systems-oriented file in the repo and a good signal for low-latency or trading-infrastructure interest.

- Inputs: synthetic concurrent bid and ask order flow
- Outputs: console inventory and cash summary
- Build and run: `g++ -std=c++17 -O2 high_frequency_limit_order_book.cpp -o hflob && ./hflob`

## Statistical Arbitrage And Backtesting

### 6. Trading Strategies Research (`trading_strategies_research.py`)

This script backtests momentum and mean-reversion strategies on synthetic price series. It provides a clean baseline for comparing simple trading rules and writes strategy returns for later visualization.

- Inputs: synthetic geometric-Brownian-motion style prices
- Outputs: console summary and `strategy_returns.csv`
- Run: `python trading_strategies_research.py`

### 7. Real-Data Trading Strategy Backtester (`real_data_strategy_backtest.py`)

This project evaluates simple momentum and mean-reversion rules on SPY data with an embedded fallback path. It computes cumulative return, volatility, and Sharpe-like metrics and saves the strategy return series for downstream use.

- Inputs: SPY history from `yfinance` or embedded fallback data
- Outputs: console performance summary, `strategy_real_returns.csv`, `strategy_cumulative_returns.png`
- Run: `python real_data_strategy_backtest.py`

### 8. Cointegration Pairs Trading (`cointegration_pairs_trading.py`)

This strategy implements a classical statistical-arbitrage workflow based on cointegration. It estimates a hedge ratio, constructs the spread, generates z-score signals, and simulates the resulting equity curve.

- Inputs: two tickers, date window, and signal threshold
- Outputs: console cointegration summary and `pairs_trading_results.png`
- Run: `python cointegration_pairs_trading.py`

### 9. Kalman Filter Pairs Trading (`kalman_pairs_trading.py`)

This version of pairs trading upgrades the static hedge ratio to a time-varying one estimated through a Kalman filter. It is stronger than a basic stat-arb demo because it handles changing relationships more realistically.

- Inputs: two tickers, date window, entry threshold, exit threshold
- Outputs: console performance summary and `kalman_pairs_trading_results.png`
- Run: `python kalman_pairs_trading.py`

### 10. Modular Backtesting Engine (`backtesting_engine.py`)

This script provides a reusable backtesting scaffold for multiple strategies. It computes portfolio returns, annualized statistics, drawdowns, and equity curves while keeping the strategy logic modular.

- Inputs: strategy definitions and either downloaded or fallback price data
- Outputs: console metrics and backtest equity plots
- Run: `python backtesting_engine.py`

## Derivatives And Volatility

### 11. Monte Carlo Option Pricing (`monte_carlo_option_pricing.py`)

This project prices a European call option with Monte Carlo simulation and reports both the estimate and its uncertainty. It is a clear example of simulation-based derivatives pricing.

- Inputs: spot, strike, maturity, rate, volatility estimate, simulation count
- Outputs: console price estimate, `option_price_paths.png`, `option_payoff_distribution.png`
- Run: `python monte_carlo_option_pricing.py`

### 12. Option Pricing And Greeks Calculator (`option_greeks_calculator.py`)

This script computes Black-Scholes option prices and the standard Greeks. It is a good fundamentals file for options math, sensitivity analysis, and visual presentation.

- Inputs: option type, spot, strike, rate, volatility, and maturity
- Outputs: console Greeks summary and `option_greeks.png`
- Run: `python option_greeks_calculator.py`

### 13. Implied Volatility Surface (`implied_vol_surface.py`)

This file builds a synthetic option surface, inverts it into implied volatilities, and visualizes the result. It shows numerical root-finding and volatility-surface intuition in a compact project.

- Inputs: synthetic option prices across strikes and maturities
- Outputs: console sample implied vols, `implied_vol_surface.png`, `implied_vol_heatmap.png`
- Run: `python implied_vol_surface.py`

### 14. Heston Stochastic Volatility Model (`heston_stochastic_vol.py`)

This project simulates joint price and variance dynamics under a Heston-style model and uses Monte Carlo pricing for a European option. It demonstrates comfort with richer dynamics than constant-volatility models.

- Inputs: model parameters, simulation count, and optional live/fallback price seed
- Outputs: console option estimate, `heston_asset_paths.png`, `heston_variance_paths.png`
- Run: `python heston_stochastic_vol.py`

### 15. Rough Bergomi Stochastic Volatility Model (`rough_bergomi_model.py`)

This is one of the most mathematically advanced files in the repository. It uses rough-volatility ideas, fractional-process simulation, and Monte Carlo pricing to showcase deeper quantitative modeling.

- Inputs: Hurst parameter, volatility-of-volatility parameters, correlation, path count
- Outputs: console option estimate and `rough_bergomi_price.png`
- Run: `python rough_bergomi_model.py`

### 16. GARCH Volatility Modeling And VaR (`garch_volatility_model.py`)

This project estimates conditional volatility through a GARCH(1,1) style model and uses the forecast to compute Value at Risk. It bridges time-series modeling and practical risk forecasting.

- Inputs: ticker, date range, forecast horizon
- Outputs: estimated parameters, volatility forecast, VaR summary, `garch_volatility_model.png`
- Run: `python garch_volatility_model.py`

## Portfolio Construction And Risk

### 17. Portfolio Optimization (`portfolio_optimization.py`)

This script samples many random portfolios, estimates annualized return and risk, and highlights the maximum-Sharpe and minimum-volatility portfolios. It is a clean and readable portfolio-construction project.

- Inputs: basket of tickers, historical or fallback prices, number of portfolios
- Outputs: `portfolio_optimisation.csv`, `portfolio_optimisation.png`, console portfolio summary
- Run: `python portfolio_optimization.py`

### 18. Risk Parity Portfolio (`risk_parity_portfolio.py`)

This project allocates capital so that each asset contributes a similar share of portfolio risk. It compares the resulting allocation against an equal-weight benchmark.

- Inputs: basket of tickers and historical or fallback prices
- Outputs: console risk contribution summary and `risk_parity_weights.png`
- Run: `python risk_parity_portfolio.py`

### 19. Factor Model Regression (`factor_model_regression.py`)

This script estimates factor exposures through regression, linking asset returns to market, size, and value factors. It is useful for showing interpretable return decomposition and regression workflow.

- Inputs: ticker, factor data, date range, with fallback generation if live data is unavailable
- Outputs: regression summary and `factor_exposures.png`
- Run: `python factor_model_regression.py`

### 20. Value At Risk And CVaR Simulation (`value_at_risk_simulation.py`)

This risk-management project computes parametric and historical VaR and CVaR for a multi-asset portfolio. It is a practical supporting file for risk analytics and portfolio stress thinking.

- Inputs: basket of tickers, confidence level, method, historical or fallback returns
- Outputs: console VaR and CVaR summary and `portfolio_var_distribution.png`
- Run: `python value_at_risk_simulation.py`

## Machine Learning And Presentation

### 21. Sentiment Analysis And Market Direction Prediction (`sentiment_analysis_market_prediction.py`)

This is a compact NLP example built on a small synthetic dataset of market-related text. It demonstrates text vectorization, classification, and prediction on unseen examples in a lightweight setting.

- Inputs: synthetic headlines and social posts
- Outputs: console model accuracy and sample predictions
- Run: `python sentiment_analysis_market_prediction.py`

### 22. Real-Data Sentiment Analysis (`real_data_sentiment_analysis.py`)

This project trains classification models on a labeled financial-news dataset and reports detailed performance statistics. It is a stronger data and NLP pipeline than the toy sentiment example above.

- Inputs: labeled financial-news data, with embedded fallback dataset
- Outputs: accuracy summary, classification reports, `logistic_confusion_matrix.png`, `random_forest_confusion_matrix.png`
- Run: `python real_data_sentiment_analysis.py`

### 23. Reinforcement Learning Trading Agent (`rl_trading_agent.py`)

This file implements a simple Q-learning style trading agent that interacts with a price environment. It is useful for showing sequential decision-making, state design, and policy-learning ideas in trading.

- Inputs: price series, episode count, exploration schedule, learning parameters
- Outputs: console training summary and `rl_trading_equity.png`
- Run: `python rl_trading_agent.py`

### 24. Interactive Quantitative Trading Dashboard (`interactive_dashboard.py`)

This project builds the repository dashboard by combining strategy, portfolio, and risk outputs into a single interactive HTML report. It is the presentation layer that helps readers review results quickly without running every script themselves.

- Inputs: strategy return CSVs plus generated or fallback price data
- Outputs: `dashboard.html`
- Run: `python interactive_dashboard.py`

## Notes

- `index.html` redirects to `portfolio.html` for a cleaner public landing page.
- Generated plots and CSV outputs are committed where they help readers review results quickly.
- A focused shortlist of standout projects lives in `FLAGSHIP_PROJECTS.md`.
