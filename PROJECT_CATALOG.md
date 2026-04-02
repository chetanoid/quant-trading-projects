# Project Catalog

This file gives a recruiter-friendly map of the repository without forcing them to open every script.

## Market Microstructure And Execution

- `limit_order_book_simulator.py`: toy limit order book and inventory-based market maker
- `real_data_market_maker.py`: market-making simulation driven by real or fallback price data
- `avellaneda_stoikov_market_maker.py`: classic inventory-aware quoting model
- `optimal_execution_almgren_chriss.py`: optimal execution schedule and shortfall analysis
- `high_frequency_limit_order_book.cpp`: compact C++ matching engine example

## Statistical Arbitrage And Backtesting

- `trading_strategies_research.py`: synthetic-data momentum and mean-reversion benchmark
- `real_data_strategy_backtest.py`: real-data backtester with offline fallback
- `cointegration_pairs_trading.py`: Engle-Granger pairs workflow with signal construction
- `kalman_pairs_trading.py`: dynamic hedge-ratio estimation with a Kalman filter
- `backtesting_engine.py`: reusable backtesting scaffold for multiple strategies

## Derivatives And Volatility

- `monte_carlo_option_pricing.py`: European option pricing with Monte Carlo simulation
- `option_greeks_calculator.py`: Black-Scholes pricing and Greeks
- `implied_vol_surface.py`: surface generation and implied-vol inversion
- `heston_stochastic_vol.py`: stochastic volatility simulation and pricing
- `rough_bergomi_model.py`: rough-volatility style simulation
- `garch_volatility_model.py`: volatility estimation and risk forecasting

## Portfolio Construction And Risk

- `portfolio_optimization.py`: efficient frontier and random portfolio sampling
- `risk_parity_portfolio.py`: equal-risk-contribution allocation
- `factor_model_regression.py`: factor exposure estimation
- `value_at_risk_simulation.py`: VaR and CVaR estimation

## Machine Learning And Presentation

- `sentiment_analysis_market_prediction.py`: toy NLP trading-sentiment classifier
- `real_data_sentiment_analysis.py`: financial-news sentiment classification with fallback data
- `rl_trading_agent.py`: reinforcement-learning style trading example
- `interactive_dashboard.py`: consolidated dashboard of strategy and portfolio outputs
- `index.html`: lightweight portfolio landing page for the repo

## Suggested Interview Flow

- Start with `README.md`.
- Open one execution project, one stat-arb project, one derivatives project, and one risk project.
- Use `interactive_dashboard.py` to give a quick visual overview after generating strategy outputs.

## Best Files To Highlight First

- `cointegration_pairs_trading.py`
- `avellaneda_stoikov_market_maker.py`
- `portfolio_optimization.py`
- `backtesting_engine.py`
- `high_frequency_limit_order_book.cpp`
