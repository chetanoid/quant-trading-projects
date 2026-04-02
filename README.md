# Quant Trading Projects

A recruiter-facing portfolio of 24 quantitative trading and finance projects spanning market microstructure, statistical arbitrage, derivatives pricing, portfolio construction, risk analytics, machine learning, and trading infrastructure.

This repository is designed to show range and depth:
- research code in Python
- numerical modeling and simulation
- offline-safe market-data fallbacks
- interactive presentation assets
- one systems-oriented C++ matching engine

## Start Here

- Public portfolio page: `portfolio.html`
- Interactive dashboard: `dashboard.html`
- Quick repository map: `PROJECT_CATALOG.md`
- Full project walkthroughs: `PROJECT_WALKTHROUGHS.md`
- Best projects for elite quant screens: `FLAGSHIP_PROJECTS.md`
- Resume-ready bullet points: `RESUME_PROJECT_BULLETS.md`

## Best Projects To Open First

If you only review six files, start here:

1. `high_frequency_limit_order_book.cpp` - systems-oriented matching engine with concurrency primitives and order-book logic
2. `avellaneda_stoikov_market_maker.py` - inventory-aware market making model with stochastic fills and P&L tracking
3. `kalman_pairs_trading.py` - adaptive statistical arbitrage with time-varying hedge ratios
4. `rough_bergomi_model.py` - advanced stochastic-volatility simulation and Monte Carlo pricing
5. `optimal_execution_almgren_chriss.py` - execution-cost modeling and implementation-shortfall analysis
6. `portfolio_optimization.py` - efficient-frontier construction and portfolio selection under risk-return tradeoffs

## Repository Coverage

- Market microstructure and execution: 5 projects
- Statistical arbitrage and backtesting: 5 projects
- Derivatives and volatility: 6 projects
- Portfolio construction and risk: 4 projects
- Machine learning and presentation: 4 projects

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python interactive_dashboard.py
```

Most scripts can be run directly with `python <script_name>.py`. Many projects attempt to download market data first and automatically fall back to embedded or synthetic data so the repo remains runnable in a fresh clone.

## What This Repo Demonstrates

- market microstructure intuition
- statistical modeling for trading signals
- derivatives pricing and stochastic-volatility modeling
- portfolio and risk analytics
- backtesting and research workflows
- practical presentation of results through dashboards and portfolio pages

## Notes

- `index.html` is a lightweight redirect entry point to `portfolio.html`.
- Generated plots and CSV outputs are committed where they help the repository present results quickly.
- Detailed project-by-project descriptions live in `PROJECT_WALKTHROUGHS.md` so the main README stays sharp and recruiter-friendly.
