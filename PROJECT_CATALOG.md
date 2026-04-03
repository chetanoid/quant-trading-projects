# Project Catalog

This file gives a quick map of the repository without requiring a full project-by-project read.

## Core Documents

- [README.md](README.md): main public overview plus detailed project walkthroughs
- [FLAGSHIP_PROJECTS.md](FLAGSHIP_PROJECTS.md): standout flagship files in the repository
- [RESUME_PROJECT_BULLETS.md](RESUME_PROJECT_BULLETS.md): project bullet library
- [Live portfolio](https://chetanoid.github.io/quant-trading-projects/portfolio.html): polished portfolio landing page
- [Live dashboard](https://chetanoid.github.io/quant-trading-projects/dashboard.html): interactive results dashboard

## Market Microstructure And Execution

- `limit_order_book_simulator.py`: compact limit order book and inventory-aware quoting
- `real_data_market_maker.py`: market-making simulation driven by real or fallback price data
- `avellaneda_stoikov_market_maker.py`: classic inventory-aware market-making model
- `optimal_execution_almgren_chriss.py`: optimal execution schedule and shortfall analysis
- `high_frequency_limit_order_book.cpp`: compact C++ matching engine example

## Statistical Arbitrage And Backtesting

- `trading_strategies_research.py`: synthetic-data momentum and mean-reversion benchmark
- `real_data_strategy_backtest.py`: real-data backtester with offline fallback
- `cointegration_pairs_trading.py`: Engle-Granger pairs workflow with spread signals
- `kalman_pairs_trading.py`: time-varying hedge-ratio estimation with a Kalman filter
- `backtesting_engine.py`: reusable backtesting scaffold for multiple strategies

## Derivatives And Volatility

- `monte_carlo_option_pricing.py`: European option pricing with Monte Carlo simulation
- `option_greeks_calculator.py`: Black-Scholes pricing and Greeks
- `implied_vol_surface.py`: surface generation and implied-vol inversion
- `heston_stochastic_vol.py`: stochastic-volatility simulation and pricing
- `rough_bergomi_model.py`: rough-volatility style simulation
- `garch_volatility_model.py`: volatility estimation and risk forecasting

## Portfolio Construction And Risk

- `portfolio_optimization.py`: efficient frontier and random portfolio sampling
- `risk_parity_portfolio.py`: equal-risk-contribution allocation
- `factor_model_regression.py`: factor exposure estimation
- `value_at_risk_simulation.py`: VaR and CVaR estimation

## Machine Learning And Presentation

- `sentiment_analysis_market_prediction.py`: synthetic-data NLP trading-sentiment classifier
- `real_data_sentiment_analysis.py`: financial-news sentiment classification with fallback data
- `rl_trading_agent.py`: reinforcement-learning style trading example
- `interactive_dashboard.py`: consolidated dashboard of strategy and portfolio outputs
- `index.html`: redirect entry point to `portfolio.html`

## Best Files To Highlight First

- `high_frequency_limit_order_book.cpp`
- `avellaneda_stoikov_market_maker.py`
- `kalman_pairs_trading.py`
- `rough_bergomi_model.py`
- `optimal_execution_almgren_chriss.py`
- `portfolio_optimization.py`
