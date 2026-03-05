"""
Real‑Data Market Making Simulator
================================

This script implements a simple limit order book and market‑making strategy
using a series of *real* mid‑prices drawn from actual financial markets.
It attempts to fetch recent daily prices for the SPY ETF from Yahoo! Finance
via the ``yfinance`` package.  If the network or library is unavailable,
it falls back to a hard‑coded sample of historical S&P 500 levels (circa
1928) drawn from an open data set on GitHub.  Although the sample data
dates back many decades, it is genuine market data and illustrates how
to integrate real observations into a microstructure simulation.

Key components:

* **Price ingestion**: ``get_price_series`` tries to download SPY prices
  with ``yfinance`` and returns a list of floats.  On failure it falls
  back to ``SAMPLE_PRICES`` defined below.
* **Order book and matching engine**: identical to the toy engine in
  ``limit_order_book_simulator.py``.  Orders are stored in heaps for
  efficient best bid/ask retrieval.
* **Market maker**: quotes around each mid price and manages inventory.
* **Event loop**: iterates over the real price series, posts quotes,
  injects random market orders, matches, and tracks marked‑to‑market P&L.

Run this script directly (``python3 real_data_market_maker.py``) to
execute the simulation.  It will print final P&L and basic summary
statistics.  Feel free to modify the sample price series or the market
maker parameters to explore different behaviours.

Author: OpenAI assistant
"""

import heapq
import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Hard‑coded fallback price series taken from the first month of the S&P 500
# dataset hosted at https://github.com/fja05680/dow-sp500-100-years (1927/1928).
# These values are real closing levels (to four decimal places) and provide
# a deterministic series if network download is unavailable.  Feel free
# to replace with more recent figures if you have access to newer data.
SAMPLE_PRICES = [
    17.6599, 17.7600, 17.7199, 17.5500, 17.6600, 17.5000, 17.3700,
    17.3500, 17.4700, 17.5800, 17.2900, 17.3000, 17.2600, 17.3800,
    17.4800, 17.6400, 17.7100, 17.5200, 17.6300, 17.6900, 17.4900,
    17.5700, 17.5300, 17.6300, 17.4000, 17.4500, 17.5400, 17.4400,
    17.4900, 17.5500, 17.5400, 17.4400, 17.5000, 17.5500, 17.6000,
    17.6500, 17.6200, 17.6800
]


def get_price_series() -> List[float]:
    """Retrieve a list of real mid prices to drive the simulation.

    Attempts to download SPY (S&P 500 ETF) daily adjusted close prices
    from Yahoo! Finance via ``yfinance``.  If unsuccessful (e.g. network
    access blocked or library unavailable), returns a hard‑coded sample
    defined in ``SAMPLE_PRICES``.

    Returns
    -------
    List[float]
        Sequence of mid prices for each timestep.
    """
    try:
        # ``yfinance`` is not part of the standard library; import it lazily.
        import yfinance as yf  # type: ignore
        # Download a single year of SPY prices as a demonstration.  Adjust
        # the start/end dates as desired.  progress=False suppresses
        # progress bars.
        data = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
        close_prices = data.get("Adj Close")
        if close_prices is not None and len(close_prices) > 0:
            # Convert the pandas Series to a Python list of floats
            return close_prices.dropna().tolist()
        else:
            raise RuntimeError("No data returned from yfinance")
    except Exception:
        # Fall back to the sample data if anything goes wrong
        return SAMPLE_PRICES


_order_id_counter = itertools.count()


@dataclass(order=True)
class Order:
    """Represents an order in the order book."""

    sort_index: float = field(init=False, repr=False)
    price: float
    qty: int
    side: str  # 'bid' or 'ask'
    trader: str  # identifier for the trader
    id: int = field(default_factory=lambda: next(_order_id_counter))

    def __post_init__(self) -> None:
        # Negative price for bids yields a max‑heap; positive price for asks yields a min‑heap
        self.sort_index = -self.price if self.side == "bid" else self.price


class OrderBook:
    """A minimal limit order book using heaps for best bid/ask retrieval."""

    def __init__(self) -> None:
        self.bids: List[Order] = []
        self.asks: List[Order] = []

    def add_order(self, order: Order) -> None:
        if order.side == "bid":
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

    def best_bid(self) -> Optional[Order]:
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Optional[Order]:
        return self.asks[0] if self.asks else None

    def match(self) -> List[Tuple[Order, Order, int]]:
        """Match crossing orders and return the executed trades.

        This method repeatedly matches the highest bid with the lowest ask
        whenever the bid price is greater than or equal to the ask price.
        Partially filled orders are re‑queued with their remaining quantity.
        """
        trades: List[Tuple[Order, Order, int]] = []
        while self.bids and self.asks and self.best_bid().price >= self.best_ask().price:
            bid_order = heapq.heappop(self.bids)
            ask_order = heapq.heappop(self.asks)
            traded_qty = min(bid_order.qty, ask_order.qty)
            trades.append((bid_order, ask_order, traded_qty))
            if bid_order.qty > traded_qty:
                remaining_bid = Order(price=bid_order.price, qty=bid_order.qty - traded_qty,
                                     side=bid_order.side, trader=bid_order.trader)
                heapq.heappush(self.bids, remaining_bid)
            if ask_order.qty > traded_qty:
                remaining_ask = Order(price=ask_order.price, qty=ask_order.qty - traded_qty,
                                     side=ask_order.side, trader=ask_order.trader)
                heapq.heappush(self.asks, remaining_ask)
        return trades


class MarketMaker:
    """A simple market maker with basic inventory control.

    The market maker posts a bid and ask around each mid price.  If its
    inventory drifts above the ``target_inventory``, it slightly lowers
    its bid and raises its ask to discourage buying.  If its inventory
    drifts below the target, it does the opposite.  This dynamic helps
    stabilize inventory around the target.
    """

    def __init__(self, name: str, book: OrderBook, spread: float = 0.01,
                 target_inventory: int = 0, max_inventory: int = 50, lot_size: int = 10) -> None:
        self.name = name
        self.book = book
        self.spread = spread
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.lot_size = lot_size
        self.inventory = 0
        self.cash = 0.0

    def quote(self, mid_price: float) -> None:
        """Submit bid and ask quotes based on current mid price and inventory."""
        # Adjust prices depending on how far inventory deviates from the target
        inventory_offset = 0.0
        if self.inventory > self.target_inventory:
            inventory_offset = 0.01
        elif self.inventory < self.target_inventory:
            inventory_offset = -0.01
        bid_price = max(0.01, mid_price * (1 - self.spread) - inventory_offset)
        ask_price = max(0.01, mid_price * (1 + self.spread) + inventory_offset)

        self.book.add_order(Order(price=bid_price, qty=self.lot_size, side="bid", trader=self.name))
        self.book.add_order(Order(price=ask_price, qty=self.lot_size, side="ask", trader=self.name))

    def update_inventory(self, trades: List[Tuple[Order, Order, int]]) -> None:
        """Update inventory and cash based on executed trades."""
        for bid_order, ask_order, qty in trades:
            if bid_order.trader == self.name:
                self.inventory += qty
                self.cash -= bid_order.price * qty
            if ask_order.trader == self.name:
                self.inventory -= qty
                self.cash += ask_order.price * qty

    @property
    def mark_to_market(self) -> float:
        """Mark the inventory to market using the current mid price."""
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        if best_bid and best_ask:
            last_price = (best_bid.price + best_ask.price) / 2
        elif best_bid:
            last_price = best_bid.price
        elif best_ask:
            last_price = best_ask.price
        else:
            last_price = 1.0
        return self.cash + self.inventory * last_price


def simulate_market_with_real_data(price_series: List[float], spread: float = 0.01,
                                   target_inventory: int = 0) -> None:
    """Run the market making simulation over a real price series.

    Parameters
    ----------
    price_series : List[float]
        Sequence of mid prices (e.g. daily adjusted closes).
    spread : float, optional
        Half‑spread for quoting (default is 0.01 = 1%).
    target_inventory : int, optional
        Desired inventory level around which to mean‑revert.
    """
    book = OrderBook()
    maker = MarketMaker(name="MarketMaker", book=book, spread=spread,
                        target_inventory=target_inventory)
    pl_history: List[float] = []

    for mid_price in price_series:
        # Market maker quotes around the real mid price
        maker.quote(mid_price)

        # Introduce a few random market orders to drive trading.  The arrival
        # probability is kept low to reflect thin liquidity on daily prices.
        for _ in range(random.randint(0, 2)):
            if random.random() < 0.5:
                best_ask = book.best_ask()
                if best_ask:
                    order = Order(price=best_ask.price, qty=random.randint(1, maker.lot_size), side="bid",
                                  trader="Trader")
                    book.add_order(order)
            else:
                best_bid = book.best_bid()
                if best_bid:
                    order = Order(price=best_bid.price, qty=random.randint(1, maker.lot_size), side="ask",
                                  trader="Trader")
                    book.add_order(order)

        trades = book.match()
        maker.update_inventory(trades)
        pl_history.append(maker.mark_to_market)

    # Final statistics
    final_pl = maker.mark_to_market
    print("Simulation complete using {} price points.".format(len(price_series)))
    print(f"Final mark‑to‑market P&L: {final_pl:.2f}")
    print(f"Final inventory: {maker.inventory}")


if __name__ == "__main__":
    prices = get_price_series()
    simulate_market_with_real_data(prices)