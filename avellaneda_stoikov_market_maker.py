"""
avellaneda_stoikov_market_maker.py
---------------------------------

This script implements a simplified version of the Avellaneda–Stoikov
optimal market–making model.  The model sets dynamic bid and ask
quotes based on a market maker’s inventory, risk aversion and the time
remaining in a trading horizon.  The algorithm derives closed–form
expressions for the reservation price (optimal mid quote) and the half
spread to maximise expected utility of terminal wealth under a
logarithmic utility function.

We simulate a market with a single asset whose mid price evolves
according to a geometric Brownian motion.  At discrete time steps, the
market maker quotes a bid and ask price derived from the
Avellaneda–Stoikov formulas.  Orders arrive according to Poisson
processes with intensities that decay exponentially as the quoted
price moves away from the mid price.  When an order is executed, the
market maker’s inventory and cash position are updated.  At the end
of the simulation the script prints the final profit and loss, the
time series of inventory and the evolution of the reservation price
and spreads.  If `matplotlib` is installed, diagnostic plots are
generated and saved to `avellaneda_stoikov_results.png`.

The implementation defaults to using synthetic price data.  If
`yfinance` is available and the network is accessible, you can pass
`--use_real_data` to download daily SPY prices instead.  The real
prices are linearly interpolated to the simulation grid.

References:
  * Marco Avellaneda and Sasha Stoikov, “High-frequency trading in a
    limit order book”, Quantitative Finance, 2008.

Usage:
    python3 avellaneda_stoikov_market_maker.py --steps 200 --T 1.0 --gamma 0.1 --sigma 0.02 --k 1.5 --use_real_data
"""

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib.pyplot as plt

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # network or module unavailable


def fetch_mid_prices(use_real_data: bool, steps: int, T: float) -> np.ndarray:
    """Return an array of mid prices for the simulation horizon.

    If `use_real_data` is True and yfinance is available, download
    daily SPY prices and interpolate them to the simulation grid.
    Otherwise generate a synthetic geometric Brownian motion.
    """
    if use_real_data and yf is not None:
        try:
            data = yf.download('SPY', period='1y', progress=False)
            if not data.empty:
                closes = data['Adj Close'].values
                # Create time grid and interpolate prices to it
                t_grid = np.linspace(0, 1, steps)
                daily_t = np.linspace(0, 1, len(closes))
                prices = np.interp(t_grid, daily_t, closes)
                return prices
        except Exception:
            pass
    # fallback: simulate synthetic price path using geometric Brownian motion
    dt = T / steps
    mu = 0.0  # drift
    sigma_synth = 0.02
    prices = np.zeros(steps)
    prices[0] = 100.0
    for i in range(1, steps):
        rnd = np.random.normal()
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma_synth ** 2) * dt + sigma_synth * np.sqrt(dt) * rnd)
    return prices


def avellaneda_quotes(S_t: float, q_t: int, t: float, T: float, gamma: float, sigma: float, k: float) -> tuple[float, float, float]:
    """Compute reservation price and half–spread using Avellaneda–Stoikov formulas.

    Args:
        S_t: current mid price
        q_t: current inventory (positive long, negative short)
        t: current time (0–T)
        T: time horizon
        gamma: risk aversion parameter
        sigma: volatility of the mid price process
        k: market depth parameter controlling arrival rate decay

    Returns:
        (reservation_price, bid_price, ask_price)
    """
    tau = T - t
    # Reservation price: adjust mid price by inventory and remaining time
    r_t = S_t - q_t * gamma * sigma ** 2 * tau
    # Half–spread: combination of risk and order arrival cost
    # See Avellaneda–Stoikov (2008) for derivation: a_t = (1/gamma) * ln(1 + gamma / k) + 0.5 * gamma * sigma^2 * tau
    a_t = (1.0 / gamma) * np.log(1.0 + gamma / k) + 0.5 * gamma * sigma ** 2 * tau
    return r_t, r_t - a_t, r_t + a_t


def simulate_avellaneda_stoikov(steps: int, T: float, gamma: float, sigma: float, k: float, use_real_data: bool) -> dict:
    """Run the Avellaneda–Stoikov simulation and return results.

    Returns a dictionary containing time series of mid prices, quotes,
    inventory, cash and P&L.
    """
    prices = fetch_mid_prices(use_real_data, steps, T)
    dt = T / steps
    inventory = 0
    cash = 0.0
    inv_series = []
    cash_series = []
    pnl_series = []
    reservation_prices = []
    bid_prices = []
    ask_prices = []
    times = []
    for i in range(steps):
        t = i * dt
        S_t = prices[i]
        r_t, bid, ask = avellaneda_quotes(S_t, inventory, t, T, gamma, sigma, k)
        reservation_prices.append(r_t)
        bid_prices.append(bid)
        ask_prices.append(ask)
        times.append(t)
        # Order arrival intensities
        lambda_bid = np.exp(-k * (S_t - bid))
        lambda_ask = np.exp(-k * (ask - S_t))
        # Poisson probability of at least one order in dt: p = 1 - exp(-lambda * dt)
        if np.random.rand() < 1 - np.exp(-lambda_bid * dt):
            # buy order arrives -> we buy, inventory increases
            inventory += 1
            cash -= bid
        if np.random.rand() < 1 - np.exp(-lambda_ask * dt):
            # sell order arrives -> we sell
            inventory -= 1
            cash += ask
        inv_series.append(inventory)
        pnl_series.append(cash + inventory * S_t)
        cash_series.append(cash)
    # Final results
    result = {
        'times': np.array(times),
        'prices': prices,
        'reservation_prices': np.array(reservation_prices),
        'bid_prices': np.array(bid_prices),
        'ask_prices': np.array(ask_prices),
        'inventory': np.array(inv_series),
        'cash': np.array(cash_series),
        'pnl': np.array(pnl_series)
    }
    return result


def plot_results(result: dict[str, np.ndarray]) -> None:
    """Generate and save diagnostic plots."""
    times = result['times']
    plt.figure(figsize=(10, 8))
    # First subplot: price and reservation price
    plt.subplot(3, 1, 1)
    plt.plot(times, result['prices'], label='Mid price')
    plt.plot(times, result['reservation_prices'], label='Reservation price')
    plt.legend(loc='best')
    plt.title('Mid Price and Reservation Price')
    # Second subplot: bid/ask quotes
    plt.subplot(3, 1, 2)
    plt.plot(times, result['bid_prices'], label='Bid quote')
    plt.plot(times, result['ask_prices'], label='Ask quote')
    plt.legend(loc='best')
    plt.title('Bid and Ask Quotes')
    # Third subplot: P&L and inventory
    plt.subplot(3, 1, 3)
    plt.plot(times, result['pnl'], label='Mark-to-market P&L')
    plt.plot(times, result['inventory'], label='Inventory')
    plt.legend(loc='best')
    plt.title('P&L and Inventory')
    plt.tight_layout()
    plt.savefig('avellaneda_stoikov_results.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Simulate Avellaneda-Stoikov market making.')
    parser.add_argument('--steps', type=int, default=200, help='Number of time steps')
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon (e.g. 1.0 for one day)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Risk aversion parameter')
    parser.add_argument('--sigma', type=float, default=0.02, help='Volatility of mid price process')
    parser.add_argument('--k', type=float, default=1.5, help='Market depth parameter')
    parser.add_argument('--use_real_data', action='store_true', help='Download real SPY data via yfinance')
    args = parser.parse_args()
    result = simulate_avellaneda_stoikov(args.steps, args.T, args.gamma, args.sigma, args.k, args.use_real_data)
    # Print summary
    final_pnl = result['pnl'][-1]
    final_inv = result['inventory'][-1]
    print(f'Final P&L: {final_pnl:.2f}, Final inventory: {final_inv}')
    # Plot results
    try:
        plot_results(result)
        print("Plots saved to 'avellaneda_stoikov_results.png'.")
    except Exception:
        print("Plotting failed. Try installing matplotlib.")


if __name__ == '__main__':
    main()
