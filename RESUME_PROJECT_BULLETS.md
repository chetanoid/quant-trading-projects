# Resume Project Bullets

Use these bullets selectively. Do not list all 6 projects on one resume. For elite quant applications, 3 to 4 well-chosen projects is usually stronger than a long uncurated list.

## 1. High-Frequency Limit Order Book
File: `high_frequency_limit_order_book.cpp`

Resume bullet options:
- Built a C++ limit-order-book and matching-engine simulation with concurrent bid/ask order producers, price-time-priority matching, and live inventory/cash tracking to model exchange-style market microstructure.
- Implemented multithreaded order ingestion and execution logic in C++ using priority queues, mutexes, and condition variables, then analyzed resulting inventory and P&L dynamics under simulated order flow.
- Designed a systems-oriented trading-engine prototype to demonstrate low-level performance thinking, concurrent state management, and order-book mechanics relevant to electronic trading infrastructure.

## 2. Avellaneda-Stoikov Market Maker
File: `avellaneda_stoikov_market_maker.py`

Resume bullet options:
- Implemented an Avellaneda-Stoikov market-making simulator that adapts bid/ask quotes to inventory, volatility, time horizon, and order-arrival intensity, with P&L and inventory diagnostics.
- Modeled inventory-aware quoting using reservation-price and optimal-spread formulas, then simulated stochastic fills and tracked mark-to-market outcomes across a full trading horizon.
- Built a market-making research prototype connecting closed-form microstructure theory to executable simulation code and visual diagnostics for quote, spread, inventory, and P&L behavior.

## 3. Kalman Filter Pairs Trading
File: `kalman_pairs_trading.py`

Resume bullet options:
- Developed a statistical-arbitrage strategy using a Kalman filter to estimate a time-varying hedge ratio, generate spread z-score signals, and backtest dynamic long/short trading rules.
- Built an adaptive pairs-trading pipeline that combines state-space modeling, rolling normalization, threshold-based execution logic, and equity-curve visualization for dynamic relative-value research.
- Implemented a market-neutral trading framework that updates hedge coefficients online and evaluates strategy behavior under both real-data and synthetic fallback regimes.

## 4. Rough Bergomi Volatility Model
File: `rough_bergomi_model.py`

Resume bullet options:
- Implemented a rough Bergomi stochastic-volatility simulator with fractional Gaussian noise generation and Monte Carlo option pricing to study non-Markovian volatility dynamics.
- Built an advanced derivatives-modeling project using rough-volatility methods, correlated Brownian drivers, and simulation-based pricing with error estimation.
- Translated rough-volatility theory into executable numerical code, including path simulation, variance dynamics, and Monte Carlo pricing of European options.

## 5. Almgren-Chriss Optimal Execution
File: `optimal_execution_almgren_chriss.py`

Resume bullet options:
- Implemented an Almgren-Chriss execution model to derive optimal liquidation schedules under market-impact and risk-aversion assumptions, then simulated implementation shortfall across price paths.
- Modeled execution cost as a function of temporary impact, permanent impact, volatility, and trade schedule, and quantified the cost-variance tradeoff through repeated simulation.
- Built an execution-research prototype connecting closed-form schedule derivation to transaction-cost analysis and shortfall-distribution visualization.

## 6. Portfolio Optimization
File: `portfolio_optimization.py`

Resume bullet options:
- Built a portfolio-optimization workflow that samples thousands of portfolios, estimates annualized return/volatility, identifies Sharpe-optimal and minimum-volatility allocations, and visualizes the efficient frontier.
- Implemented a mean-variance research pipeline with real-data ingestion, return/covariance estimation, random weight generation, and exportable portfolio analytics.
- Created a portfolio-construction project that links market data, risk estimation, and position-allocation decisions into a recruiter-friendly research artifact.

## Best 4-Project Resume Mixes

For quant trader:
- `avellaneda_stoikov_market_maker.py`
- `kalman_pairs_trading.py`
- `optimal_execution_almgren_chriss.py`
- `high_frequency_limit_order_book.cpp`

For quant researcher:
- `kalman_pairs_trading.py`
- `rough_bergomi_model.py`
- `avellaneda_stoikov_market_maker.py`
- `portfolio_optimization.py`

For quant developer / research engineer:
- `high_frequency_limit_order_book.cpp`
- `optimal_execution_almgren_chriss.py`
- `avellaneda_stoikov_market_maker.py`
- `kalman_pairs_trading.py`

## One-Line Summary For Resume Header Or Project Section

- Selected projects in market microstructure, statistical arbitrage, stochastic volatility, execution modeling, and portfolio optimization, implemented across Python and C++.
