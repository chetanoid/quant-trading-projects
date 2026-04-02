# Project Walkthroughs

This document is the detailed companion to `README.md`. It keeps the main README focused while still giving a complete, sequential view of the repository.

## Market Microstructure And Execution

## 1. `limit_order_book_simulator.py`

Toy limit-order-book simulator with bid and ask queues, fills, and inventory-aware P&L tracking. Good entry point for understanding the repo's market-microstructure layer.

## 2. `real_data_market_maker.py`

Market-making simulation driven by real or fallback price data. Demonstrates quoting, fills, inventory evolution, and mark-to-market performance under a simple dealer model.

## 3. `avellaneda_stoikov_market_maker.py`

Implements the classic inventory-aware market-making framework. Connects quote placement, volatility, order-arrival intensity, and inventory risk in a more research-grade setup.

## 4. `optimal_execution_almgren_chriss.py`

Models optimal liquidation under temporary impact, permanent impact, and risk aversion. Useful for showing execution thinking rather than only alpha generation.

## 5. `high_frequency_limit_order_book.cpp`

Compact C++ matching engine with concurrent order producers, order-book state, and trade matching logic. Strongest systems-style project in the repo.

## Statistical Arbitrage And Backtesting

## 6. `trading_strategies_research.py`

Synthetic-data benchmark for momentum and mean-reversion ideas. Useful as a clean baseline before moving into real-data backtests.

## 7. `real_data_strategy_backtest.py`

Real-data strategy backtester with offline fallbacks and exported return series. Produces reusable outputs for later analytics and dashboarding.

## 8. `cointegration_pairs_trading.py`

Engle-Granger style pairs-trading workflow with hedge-ratio estimation, spread construction, z-score signals, and backtest results.

## 9. `kalman_pairs_trading.py`

Adaptive pairs-trading model where the hedge ratio evolves through time. Stronger than a static cointegration demo because it handles non-stationary relationships more realistically.

## 10. `backtesting_engine.py`

Reusable backtesting scaffold for multiple strategies with performance metrics and equity-curve outputs. Good signal for engineering discipline inside research code.

## Derivatives And Volatility

## 11. `monte_carlo_option_pricing.py`

Monte Carlo pricer for European options with simulated paths, payoff estimation, and error reporting. Clear demonstration of simulation-based pricing.

## 12. `option_greeks_calculator.py`

Black-Scholes pricing plus Delta, Gamma, Vega, Theta, and Rho. Good fundamentals project for options math and risk sensitivities.

## 13. `implied_vol_surface.py`

Builds a synthetic options surface and inverts prices into implied volatilities. Demonstrates numerical root-finding and volatility-surface intuition.

## 14. `heston_stochastic_vol.py`

Stochastic-volatility simulation and Monte Carlo pricing under a Heston-style model. Shows comfort with richer dynamics than constant-volatility pricing.

## 15. `rough_bergomi_model.py`

Advanced rough-volatility model using fractional-process ideas. One of the mathematically strongest files in the repository.

## 16. `garch_volatility_model.py`

Fits a GARCH(1,1) style model for conditional volatility and Value at Risk estimation. Useful bridge between time-series modeling and risk forecasting.

## Portfolio Construction And Risk

## 17. `portfolio_optimization.py`

Generates random portfolios, estimates return and volatility, and highlights maximum-Sharpe and minimum-volatility portfolios. Fast, readable demonstration of portfolio construction.

## 18. `risk_parity_portfolio.py`

Constructs equal-risk-contribution allocations and compares them with equal-weight benchmarks. Good portfolio-allocation and risk-budgeting example.

## 19. `factor_model_regression.py`

Estimates factor exposures through regression and plots the resulting betas. Useful for linking asset returns to interpretable risk drivers.

## 20. `value_at_risk_simulation.py`

Computes parametric and historical VaR and CVaR for a multi-asset portfolio. Good supporting risk-management project with direct trading relevance.

## Machine Learning And Presentation

## 21. `sentiment_analysis_market_prediction.py`

Toy NLP-style market sentiment classifier. Simpler than the real-data sentiment project, but useful as a lightweight machine-learning example.

## 22. `real_data_sentiment_analysis.py`

Financial-news sentiment classifier trained on a labeled dataset with an embedded fallback. Stronger data pipeline and reporting than the toy sentiment script.

## 23. `rl_trading_agent.py`

Q-learning style trading agent operating on price-series state regimes. Demonstrates sequential decision-making and policy learning in a trading context.

## 24. `interactive_dashboard.py`

Builds the interactive repository dashboard by combining strategy, portfolio, and risk outputs into a single HTML report. This is the presentation layer that helps recruiters review results quickly.

## Supporting Assets

- `portfolio.html` is the polished public portfolio landing page.
- `dashboard.html` is the generated interactive dashboard output.
- `FLAGSHIP_PROJECTS.md` ranks the strongest files for elite quant screening.
- `RESUME_PROJECT_BULLETS.md` contains resume-ready project phrasing.
