"""
rl_trading_agent.py
-------------------

This script illustrates a simple reinforcement learning (Q‑learning)
approach to trading.  We simulate a price series (either by
downloading a real asset with `yfinance` or generating a synthetic
random walk) and define a discrete state space based on the asset’s
position relative to a moving average.  At each time step the agent
chooses an action: go long (+1), go short (–1) or stay flat (0).
Rewards are proportional to the price change times the current
position.  The Q‑learning algorithm updates its state–action value
table using the Bellman equation.  After training for multiple
episodes, we evaluate the learned policy on the price series and
produce an equity curve.  The resulting plot is saved to
`rl_trading_equity.png`.

This example is deliberately simplified to run without external
libraries such as TensorFlow or PyTorch.  It serves to demonstrate
your understanding of RL concepts (states, actions, rewards,
exploration vs. exploitation, learning rate and discount factor) in a
finance context.

Usage:
    python3 rl_trading_agent.py --ticker SPY --episodes 50 --window 5

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
    yf = None


def fetch_price_series(ticker: str, start: str = '2019-01-01', end: str = None) -> np.ndarray:
    """Fetch daily prices for a ticker or generate synthetic random walk."""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    if yf is not None:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                return data['Adj Close'].values
        except Exception:
            pass
    # fallback: generate random walk
    n = 1000
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + np.random.normal(scale=1.0)
    return prices


def discretize_state(prices: np.ndarray, t: int, window: int) -> int:
    """Discretise state based on the price relative to its moving average.

    Returns an integer state: -1 if price below moving average minus
    threshold, 1 if above, 0 if near.
    """
    if t < window:
        return 0
    ma = np.mean(prices[t - window:t])
    diff = prices[t] - ma
    thresh = 0.5  # threshold for state classification
    if diff > thresh:
        return 1
    elif diff < -thresh:
        return -1
    else:
        return 0


def q_learning(prices: np.ndarray, episodes: int = 50, window: int = 5,
               alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1) -> np.ndarray:
    """Train a Q‑table via Q‑learning and return the learned table.

    States: {-1, 0, 1}; actions: {short (-1), flat (0), long (1)}.
    """
    state_space = [-1, 0, 1]
    action_space = [-1, 0, 1]
    # Q table shape (n_states, n_actions)
    Q = np.zeros((len(state_space), len(action_space)))
    state_to_idx = {s: i for i, s in enumerate(state_space)}
    action_to_idx = {a: i for i, a in enumerate(action_space)}
    n = len(prices)
    for _ in range(episodes):
        position = 0  # start flat
        for t in range(1, n - 1):
            # Determine current state
            s = discretize_state(prices, t, window)
            s_idx = state_to_idx[s]
            # Choose action via epsilon‑greedy
            if np.random.rand() < epsilon:
                a = np.random.choice(action_space)
            else:
                a = action_space[np.argmax(Q[s_idx])]
            a_idx = action_to_idx[a]
            # Reward: next price change times current position
            reward = ((prices[t + 1] - prices[t]) / max(abs(prices[t]), 1e-8)) * position
            # Update position to chosen action
            position = a
            # Next state
            s_next = discretize_state(prices, t + 1, window)
            s_next_idx = state_to_idx[s_next]
            # Temporal difference target
            td_target = reward + gamma * np.max(Q[s_next_idx])
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error
        # After each episode reset position
    return Q


def evaluate_policy(prices: np.ndarray, Q: np.ndarray, window: int = 5) -> np.ndarray:
    """Evaluate the learned policy on the price series and return equity curve."""
    state_space = [-1, 0, 1]
    action_space = [-1, 0, 1]
    state_to_idx = {s: i for i, s in enumerate(state_space)}
    action_space_array = np.array(action_space)
    position = 0
    cash = 1.0
    equity_curve = [cash]
    for t in range(1, len(prices)):
        # Realise P&L from the position held over the previous step.
        reward = ((prices[t] - prices[t - 1]) / max(abs(prices[t - 1]), 1e-8)) * position
        cash += reward
        state = discretize_state(prices, t, window)
        s_idx = state_to_idx[state]
        # Choose best action
        a_idx = np.argmax(Q[s_idx])
        position = action_space_array[a_idx]
        equity_curve.append(cash)
    return np.array(equity_curve)


def plot_equity(equity: np.ndarray) -> None:
    """Save equity curve to a PNG file."""
    plt.figure(figsize=(8, 4))
    plt.plot(equity)
    plt.title('Equity Curve using Q‑Learning Policy')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()
    plt.savefig('rl_trading_equity.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train a simple Q‑learning trading agent.')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to download')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--window', type=int, default=5, help='Moving average window for state definition')
    args = parser.parse_args()
    prices = fetch_price_series(args.ticker)
    Q = q_learning(prices, episodes=args.episodes, window=args.window)
    equity = evaluate_policy(prices, Q, window=args.window)
    # Print simple performance metrics
    total_return = (equity[-1] - equity[0]) / equity[0] if len(equity) > 1 else 0.0
    print(f'Final equity value: {equity[-1]:.2f}, Total return over evaluation: {total_return * 100:.2f}%')
    try:
        plot_equity(equity)
        print("Equity curve saved to 'rl_trading_equity.png'.")
    except Exception:
        print("Plotting failed. Install matplotlib if you wish to view charts.")


if __name__ == '__main__':
    main()
