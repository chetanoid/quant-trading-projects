# Flagship Projects For Elite Quant Screening

This file highlights the strongest 6 projects in the repository for Jane Street / Citadel-style screening.

The selection is based on the types of skills emphasized in official role descriptions: quantitative modeling, statistical analysis, market signals, backtesting, algorithmic trading systems, large-scale data work, risk thinking, and strong Python/C++ implementation. In practice, these firms reward projects that show judgment, mathematical precision, and the ability to turn ideas into clean code.

## Recommended Top 6

## 1. `high_frequency_limit_order_book.cpp`

Why it stands out:
- Shows systems-level thinking instead of only notebook-style research
- Uses C++, concurrency primitives, matching logic, and order-book state management
- Maps well to market microstructure, trading infrastructure, and low-latency engineering

What to emphasize:
- Priority-queue design for bids and asks
- Matching-engine thread plus producer threads
- Inventory and cash tracking
- What you would optimize next for true low-latency use

Best fit:
- Jane Street software / low-latency / production engineering
- Citadel Securities research engineering / trading systems

## 2. `avellaneda_stoikov_market_maker.py`

Why it stands out:
- Uses a canonical market-making model rather than a toy strategy heuristic
- Connects inventory risk, quote placement, volatility, and order-arrival intensity
- Demonstrates that you understand both the math and the trading intuition

What to emphasize:
- Reservation price and spread derivation
- Inventory-aware quoting
- Poisson-style execution arrivals
- How model parameters affect P&L and inventory paths

Best fit:
- Quant trader
- Quant researcher
- Electronic market making / liquidity provision teams

## 3. `kalman_pairs_trading.py`

Why it stands out:
- Stronger than a basic cointegration demo because the hedge ratio updates dynamically
- Combines state-space modeling, signal construction, and backtesting
- Looks like real research work: model, signal, thresholding, and performance

What to emphasize:
- Dynamic beta estimation with a Kalman filter
- Spread construction and rolling z-score logic
- Trade entry / exit rules and dollar-neutral framing
- Why adaptive hedging is better than a static regression in non-stationary markets

Best fit:
- Quant research
- Statistical arbitrage
- Trader / researcher hybrid roles

## 4. `rough_bergomi_model.py`

Why it stands out:
- This is materially more advanced than standard Black-Scholes or a basic Monte Carlo pricer
- Signals comfort with stochastic calculus, fractional processes, and numerical methods
- Uncommon enough to make strong reviewers pause and actually read

What to emphasize:
- Fractional Gaussian noise construction
- Rough-volatility dynamics and parameter interpretation
- Monte Carlo pricing setup and error estimation
- Tradeoff between model realism and computational complexity

Best fit:
- Derivatives research
- Quantitative modeling
- Volatility / options-focused roles

## 5. `optimal_execution_almgren_chriss.py`

Why it stands out:
- Shows execution thinking, not just alpha generation
- Connects risk aversion, temporary impact, permanent impact, and implementation shortfall
- Strong complement to market-making and order-book work

What to emphasize:
- Closed-form liquidation schedule
- Cost-variance tradeoff
- Shortfall simulation across paths
- How execution objectives change with volatility and risk aversion

Best fit:
- Quant trading
- Execution research
- Market microstructure roles

## 6. `portfolio_optimization.py`

Why it stands out:
- Clear, readable example of turning return/covariance estimates into portfolio decisions
- Good “bridge” project between research and risk
- Useful because recruiters can understand it quickly, even outside pure quant teams

What to emphasize:
- Efficient frontier generation
- Sharpe-optimal versus minimum-volatility portfolios
- Random-portfolio sampling and annualization
- How you would extend this to robust optimization, constraints, or transaction costs

Best fit:
- Quant research
- Risk / portfolio analytics
- Generalist quant screens

## Honorable Mentions

- `cointegration_pairs_trading.py`: good backup if you want a more classical stat-arb example
- `value_at_risk_simulation.py`: useful supporting risk project, but less differentiated than the top 6
- `real_data_sentiment_analysis.py`: reasonable ML/NLP side project, but weaker as a flagship than the core quant / microstructure work
- `backtesting_engine.py`: good infrastructure signal, especially if you strengthen the framework angle further

## Best 4 If You Need A Shortlist

If you only want to push 4 projects hard, use:

1. `high_frequency_limit_order_book.cpp`
2. `avellaneda_stoikov_market_maker.py`
3. `kalman_pairs_trading.py`
4. `rough_bergomi_model.py`

That set gives you:
- systems
- market microstructure
- statistical modeling
- advanced derivatives math

## Role-Based Ordering

For quant trader applications:
1. `avellaneda_stoikov_market_maker.py`
2. `kalman_pairs_trading.py`
3. `optimal_execution_almgren_chriss.py`
4. `high_frequency_limit_order_book.cpp`
5. `rough_bergomi_model.py`

For quant researcher applications:
1. `kalman_pairs_trading.py`
2. `rough_bergomi_model.py`
3. `avellaneda_stoikov_market_maker.py`
4. `portfolio_optimization.py`
5. `value_at_risk_simulation.py`

For quant developer / research engineer applications:
1. `high_frequency_limit_order_book.cpp`
2. `optimal_execution_almgren_chriss.py`
3. `avellaneda_stoikov_market_maker.py`
4. `kalman_pairs_trading.py`
5. `backtesting_engine.py`
