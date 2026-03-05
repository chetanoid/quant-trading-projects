"""
Limit Order Book & Market Making Simulator
=========================================

This script implements a simple limit order book and a market‑making strategy for educational purposes.
It simulates an order book over a fixed number of timesteps, uses a very basic market maker
to quote bid/ask prices around a reference price, handles market orders from other traders,
matches orders, and calculates a P&L for the market maker.  The goal is to demonstrate
key concepts in market microstructure rather than to serve as production‑ready code.

Key features:
  * Event‑driven matching engine for limit and market orders.
  * Simple inventory‑based quoting strategy for the market maker.
  * Synthetic price process for the underlying asset.
  * Calculation of P&L, inventory and trade count for the market maker.

The code is heavily commented to help the reader follow along.  Feel free to experiment
with the parameters (number of timesteps, volatility, order size, etc.) to see how the
strategy behaves under different conditions.

Author: OpenAI assistant
"""

import heapq
import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Unique ID generator for orders
_order_id_counter = itertools.count()


@dataclass(order=True)
class Order:
    """Represents an order in the order book."""

    sort_index: float = field(init=False, repr=False)
    price: float
    qty: int
    side: str  # 'bid' or 'ask'
    trader: str  # market maker or external trader
    id: int = field(default_factory=lambda: next(_order_id_counter))

    def __post_init__(self):
        # For bids, use negative price to create a max-heap; for asks use positive price for min-heap
        # The sort_index ensures proper ordering in the heap
        self.sort_index = -self.price if self.side == "bid" else self.price


class OrderBook:
    """A simple limit order book supporting bid/ask matching."""

    def __init__(self):
        # Use heaps to represent the best bid and best ask queues
        self.bids: List[Order] = []  # max‑heap of bids
        self.asks: List[Order] = []  # min‑heap of asks

    def add_order(self, order: Order):
        """Add an order to the book."""
        if order.side == "bid":
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

    def best_bid(self) -> Optional[Order]:
        """Return the best bid order without removing it."""
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Optional[Order]:
        """Return the best ask order without removing it."""
        return self.asks[0] if self.asks else None

    def match(self) -> List[Tuple[Order, Order]]:
        """Match crossing orders and return a list of executed trades."""
        trades = []
        while self.bids and self.asks and self.best_bid().price >= self.best_ask().price:
            bid_order = heapq.heappop(self.bids)
            ask_order = heapq.heappop(self.asks)
            traded_qty = min(bid_order.qty, ask_order.qty)
            trades.append((bid_order, ask_order, traded_qty))
            # Reduce quantities if partially matched
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
    """A simple market maker quoting around a reference price with basic inventory control."""

    def __init__(self, name: str, book: OrderBook, spread: float = 0.02, target_inventory: int = 0,
                 max_inventory: int = 50, lot_size: int = 10):
        self.name = name
        self.book = book
        self.spread = spread  # half‑spread around mid price
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.inventory = 0
        self.lot_size = lot_size
        self.cash = 0.0  # cash position for P&L calculation

    def quote(self, mid_price: float):
        """Generate quotes based on current inventory and mid price."""
        # Adjust quote to nudge inventory towards target
        inventory_offset = 0.0
        if self.inventory > self.target_inventory:
            inventory_offset = 0.01  # quote lower to encourage selling
        elif self.inventory < self.target_inventory:
            inventory_offset = -0.01  # quote higher to encourage buying
        bid_price = max(0.01, mid_price * (1 - self.spread) - inventory_offset)
        ask_price = max(0.01, mid_price * (1 + self.spread) + inventory_offset)

        bid_order = Order(price=bid_price, qty=self.lot_size, side="bid", trader=self.name)
        ask_order = Order(price=ask_price, qty=self.lot_size, side="ask", trader=self.name)
        self.book.add_order(bid_order)
        self.book.add_order(ask_order)

    def update_inventory(self, trades: List[Tuple[Order, Order, int]]):
        """Update inventory and cash based on executed trades."""
        for bid_order, ask_order, qty in trades:
            # If self is involved as bid or ask side, update inventory and cash accordingly
            if bid_order.trader == self.name:
                # We bought qty at bid_order.price
                self.inventory += qty
                self.cash -= bid_order.price * qty
            if ask_order.trader == self.name:
                # We sold qty at ask_order.price
                self.inventory -= qty
                self.cash += ask_order.price * qty

    @property
    def mark_to_market(self) -> float:
        """Return the marked‑to‑market value of current holdings at the last traded price."""
        # If there is a last traded price, use best bid/ask midpoint; fallback to 1.0
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


def simulate_market(timesteps: int = 200, base_price: float = 100.0, volatility: float = 0.2) -> None:
    """Run the limit order book simulation for a fixed number of timesteps."""
    book = OrderBook()
    maker = MarketMaker(name="MarketMaker", book=book, spread=0.01, target_inventory=0, max_inventory=50, lot_size=10)

    mid_price = base_price
    price_history = []
    p_and_l = []

    for t in range(timesteps):
        # Update mid price via geometric Brownian motion
        drift = 0.0
        shock = random.gauss(0, 1) * volatility
        mid_price *= (1 + drift + shock)
        price_history.append(mid_price)

        # Market maker posts quotes around mid price
        maker.quote(mid_price)

        # Simulate random market order arrivals
        num_market_orders = random.randint(0, 3)
        for _ in range(num_market_orders):
            # Choose side opposite to current mid price movement to create crossings
            if random.random() < 0.5:
                # Market buy order crosses the ask
                best_ask = book.best_ask()
                if best_ask:
                    order = Order(price=best_ask.price, qty=random.randint(1, 10), side="bid", trader="Trader")
                    book.add_order(order)
            else:
                # Market sell order crosses the bid
                best_bid = book.best_bid()
                if best_bid:
                    order = Order(price=best_bid.price, qty=random.randint(1, 10), side="ask", trader="Trader")
                    book.add_order(order)

        # Match orders and update market maker inventory and cash
        trades = book.match()
        maker.update_inventory(trades)
        p_and_l.append(maker.mark_to_market)

    # Summary statistics
    final_pl = maker.mark_to_market
    total_trades = len([t for t in book.match()])
    print("Simulation complete.")
    print(f"Final P&L (mark‑to‑market): {final_pl:.2f}")
    print(f"Inventory at end: {maker.inventory}")
    print(f"Cash at end: {maker.cash:.2f}")
    print(f"Approx. number of trades executed by market maker: {total_trades}")

    # Optionally write results to a file
    try:
        with open("limit_order_book_results.txt", "w") as f:
            f.write("Final P&L: {:.2f}\n".format(final_pl))
            f.write("Final inventory: {}\n".format(maker.inventory))
            f.write("Final cash: {:.2f}\n".format(maker.cash))
            f.write("Price history: {}\n".format(price_history))
            f.write("P&L history: {}\n".format(p_and_l))
    except Exception as e:
        print(f"Could not write results file: {e}")


if __name__ == "__main__":
    # Run the simulation when executed as a script
    simulate_market(timesteps=200, base_price=100.0, volatility=0.005)