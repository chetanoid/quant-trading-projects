r"""
Almgren–Chriss Optimal Execution Model
--------------------------------------

The Almgren–Chriss (AC) model provides a framework for determining the optimal
trading schedule when liquidating a large position over a fixed time horizon.
It balances the trade‑off between market impact (cost) and risk (variance of
execution cost) by solving for a sequence of trades that minimises a linear
combination of expected cost and variance. This project demonstrates how to
compute an optimal schedule and evaluate its performance using a synthetic
price process.

Key features of this script include:

* **Closed‑form optimal schedule:** Given permanent and temporary impact
  parameters (``gamma`` and ``eta``), risk aversion ``lambda``, initial
  position ``Q``, and trading horizon ``T``, the AC model yields optimal
  shares to trade at each interval. The script derives these quantities
  directly without solving differential equations.
* **Synthetic price simulation:** We simulate a simple Brownian motion
  (geometric random walk) to represent the execution price path. The
  execution prices incorporate temporary impact proportional to the
  instantaneous trade size.
* **Cost and variance estimation:** The realised execution cost and its
  variance over many simulated price paths are computed, illustrating the
  classic cost–variance trade‑off at the heart of the AC model.
* **Visualisation:** The optimal trading schedule and distribution of costs
  across simulations are saved as PNG files. These plots help convey the
  execution dynamics to reviewers and interviewers.

The model parameters used here are illustrative; they can be adapted to
reflect market conditions. For further reading, see: Robert Almgren and Neil
Chriss, "Optimal execution of portfolio transactions," *Journal of Risk* (2000).
"""

import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib
# Use a non-interactive backend to avoid issues in headless environments
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt


def compute_optimal_schedule(
    Q: float,
    T: float,
    N: int,
    sigma: float,
    gamma: float,
    eta: float,
    lamb: float,
):
    """Compute the optimal AC trading schedule.

    The AC solution has the form:

    .. math::

       x_i = A\sinh(\kappa (T - t_i)) + B\cosh(\kappa (T - t_i)),

    where \kappa = \sqrt{\lambda \sigma^2 / \eta} and constants A,B are
    determined by the boundary conditions x(0)=Q and x(T)=0. Discretising
    evenly spaced time points ``t_i`` yields the shares remaining at each
    interval. The trade size ``q_i`` executed at time ``t_i`` is the
    difference between successive ``x`` values.

    Parameters
    ----------
    Q : float
        Initial shares to liquidate.
    T : float
        Trading horizon (hours or days).
    N : int
        Number of trades (time intervals).
    sigma : float
        Volatility of the asset price (per sqrt(time unit)).
    gamma : float
        Permanent market impact coefficient.
    eta : float
        Temporary market impact coefficient.
    lamb : float
        Risk aversion parameter.

    Returns
    -------
    t_grid : ndarray
        Time points at which trades occur.
    x : ndarray
        Shares remaining at each time point (length N+1).
    q : ndarray
        Shares traded in each interval (length N).
    """
    dt = T / N
    t_grid = np.linspace(0, T, N + 1)
    # Compute kappa and normalising constants
    if lamb <= 0:
        # Risk-neutral solution: trade equally in each interval
        q = np.full(N, Q / N)
        x = Q - np.cumsum(np.hstack([[0.0], q]))
        return t_grid, x, q
    kappa = np.sqrt(lamb * sigma**2 / eta)
    # Compute numerator and denominator for A,B constants
    denom = np.sinh(kappa * T)
    # x(t) = Q * (sinh(kappa*(T - t)) / sinh(kappa*T))
    x = Q * np.sinh(kappa * (T - t_grid)) / denom
    q = -np.diff(x)
    return t_grid, x, q


def simulate_price_path(
    S0: float,
    sigma: float,
    T: float,
    N: int,
    seed: int | None = None,
):
    """Simulate a simple geometric Brownian motion price path."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(N):
        S[i + 1] = S[i] * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * np.random.randn())
    return S


def evaluate_execution_cost(
    S: np.ndarray,
    q: np.ndarray,
    gamma: float,
    eta: float,
    permanent_impact: bool = True,
):
    """Compute execution cost for a single price path and schedule.

    Permanent impact shifts the price linearly with cumulative trades (``gamma``),
    while temporary impact applies to each trade (``eta`` * q_i).
    """
    cost = 0.0
    S_exec = S.copy()
    cumulative = 0.0
    for i in range(len(q)):
        if permanent_impact:
            S_exec[i] -= gamma * cumulative
        execution_price = S_exec[i] - eta * q[i]
        cost += execution_price * q[i]
        cumulative += q[i]
    # Market value of shares sold: approximate as initial price * total quantity
    proceeds = S[0] * np.sum(q)
    implementation_shortfall = proceeds - cost
    return implementation_shortfall


def main():
    """Run the Almgren–Chriss model and evaluate the optimal schedule."""
    # Model parameters (illustrative)
    Q = 1_000_000  # shares to sell
    T = 1.0  # trading horizon in days
    N = 20  # number of trades
    sigma = 0.02  # daily volatility (2%)
    gamma = 2e-7  # permanent impact coefficient
    eta = 5e-5  # temporary impact coefficient
    lamb = 1e-6  # risk aversion
    # Compute optimal schedule
    t_grid, x, q = compute_optimal_schedule(Q, T, N, sigma, gamma, eta, lamb)
    # Simulate many price paths to estimate cost distribution
    n_sims = 500
    S0 = 100.0
    costs = []
    for i in range(n_sims):
        S_path = simulate_price_path(S0, sigma, T, N)
        cost = evaluate_execution_cost(S_path, q, gamma, eta, permanent_impact=True)
        costs.append(cost)
    costs = np.array(costs)
    mean_cost = costs.mean()
    cost_std = costs.std()
    print(f"Optimal trade sizes (first five): {q[:5].astype(int)} ... total {q.sum():.0f}")
    print(f"Mean implementation shortfall: ${mean_cost:,.2f}, Std: ${cost_std:,.2f}")
    # Visualise schedule and cost distribution
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1, N + 1), q / 1000)
    plt.title("Optimal AC trading schedule (k shares per interval)")
    plt.xlabel("Trade number")
    plt.ylabel("Shares traded (k)")
    plt.tight_layout()
    plt.savefig("ac_optimal_schedule.png")
    plt.close()
    # Plot cost distribution using numpy.histogram to avoid interactive `plt.hist`
    hist_counts, bin_edges = np.histogram(costs / 1e6, bins=30)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.figure(figsize=(8, 4))
    plt.bar(bin_centres, hist_counts, width=(bin_edges[1] - bin_edges[0]))
    plt.title("Distribution of implementation shortfall (millions)")
    plt.xlabel("Shortfall ($MM)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("ac_shortfall_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
