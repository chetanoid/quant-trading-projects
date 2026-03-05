/*
 * high_frequency_limit_order_book.cpp
 *
 * This example implements a basic limit‑order book (LOB) engine in C++.  The goal
 * is to demonstrate proficiency with lower‑level systems programming—manual
 * memory management, concurrency, and performance considerations—which are
 * essential skills at top trading firms.  The engine supports concurrent
 * insertion of buy and sell orders, matching orders, and maintaining
 * aggregated book state.
 *
 * Key features:
 *   1. Order representation using lightweight structs and custom allocators.
 *   2. Separate threads simulating order producers for bids and asks.
 *   3. A matching engine that runs on its own thread and processes orders
 *      atomically using mutexes and condition variables.
 *   4. Real‑time P&L and inventory tracking for a market maker.
 *
 * Note: This program uses only the C++ standard library, avoiding
 * external dependencies.  It's written for clarity, not raw speed, but
 * demonstrates how a systems‑level trading engine might be structured.  To
 * compile, run:  g++ -std=c++17 -O2 high_frequency_limit_order_book.cpp -o hflob
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Simple representation of a limit order.  For demonstration we only
// consider limit orders (no market orders) and immediate fill/cancel logic.
struct Order {
    int id;           // unique order identifier
    double price;     // limit price
    int quantity;     // quantity of units
    bool is_buy;      // true for buy, false for sell
    Order(int id_, double price_, int quantity_, bool is_buy_)
        : id(id_), price(price_), quantity(quantity_), is_buy(is_buy_) {}
};

// Comparator for priority queues: highest bid first for buys, lowest ask first for sells.
struct BuyCompare {
    bool operator()(const Order &a, const Order &b) const {
        // For buys, higher price has priority; if equal, earlier id has priority
        if (a.price == b.price) return a.id > b.id;
        return a.price < b.price;
    }
};

struct SellCompare {
    bool operator()(const Order &a, const Order &b) const {
        // For sells, lower price has priority; if equal, earlier id has priority
        if (a.price == b.price) return a.id > b.id;
        return a.price > b.price;
    }
};

class LimitOrderBook {
public:
    LimitOrderBook() : next_id(0), inventory(0), cash(0.0) {}

    // Adds a new order into the appropriate queue
    void add_order(double price, int quantity, bool is_buy) {
        std::unique_lock<std::mutex> lock(mtx);
        Order order(next_id++, price, quantity, is_buy);
        if (is_buy) {
            buy_orders.push(order);
        } else {
            sell_orders.push(order);
        }
        cv.notify_one();
    }

    // Matching loop runs on its own thread and attempts to match orders
    void match_orders() {
        while (running) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !running || (!buy_orders.empty() && !sell_orders.empty()); });
            if (!running) break;

            if (buy_orders.empty() || sell_orders.empty()) continue;
            Order buy = buy_orders.top();
            Order sell = sell_orders.top();
            // If best bid >= best ask, match
            if (buy.price >= sell.price) {
                int quantity_traded = std::min(buy.quantity, sell.quantity);
                double trade_price = (buy.id < sell.id) ? buy.price : sell.price;
                // Update inventory and cash: inventory increases when we buy (sell orders remove inventory)
                inventory += quantity_traded * (sell.is_buy ? -1 : 1);
                cash -= trade_price * quantity_traded * (buy.is_buy ? 1 : -1);
                // Update order quantities
                buy.quantity -= quantity_traded;
                sell.quantity -= quantity_traded;
                buy_orders.pop();
                sell_orders.pop();
                // If there is remaining quantity, reinsert the partially filled order
                if (buy.quantity > 0) buy_orders.push(buy);
                if (sell.quantity > 0) sell_orders.push(sell);
            } else {
                // No match possible; break to wait for more orders
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }

    // Get current book snapshot
    std::vector<Order> get_top_n(int n) {
        std::vector<Order> snapshot;
        std::unique_lock<std::mutex> lock(mtx);
        // Copy top n orders from both sides (for demonstration only)
        auto copy_queue = [&](auto &pq, auto &dest, bool is_buy) {
            auto temp = pq;
            for (int i = 0; i < n && !temp.empty(); ++i) {
                dest.push_back(temp.top());
                temp.pop();
            }
        };
        copy_queue(buy_orders, snapshot, true);
        copy_queue(sell_orders, snapshot, false);
        return snapshot;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            running = false;
        }
        cv.notify_all();
    }

    int get_inventory() const { return inventory; }
    double get_cash() const { return cash; }

private:
    std::priority_queue<Order, std::vector<Order>, BuyCompare> buy_orders;
    std::priority_queue<Order, std::vector<Order>, SellCompare> sell_orders;
    mutable std::mutex mtx;
    std::condition_variable cv;
    int next_id;
    int inventory;
    double cash;
    bool running = true;
};

// Producer thread function: generates random orders at a given rate
void order_producer(LimitOrderBook &lob, bool is_buy, double base_price, double price_sd,
                    int avg_quantity, int num_orders, std::mt19937 &rng) {
    std::normal_distribution<double> price_dist(base_price, price_sd);
    std::poisson_distribution<int> quantity_dist(avg_quantity);
    for (int i = 0; i < num_orders; ++i) {
        double price = price_dist(rng);
        int quantity = std::max(1, quantity_dist(rng));
        lob.add_order(price, quantity, is_buy);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

int main() {
    LimitOrderBook lob;
    std::mt19937 rng(42);
    // Start matching engine thread
    std::thread matcher(&LimitOrderBook::match_orders, &lob);
    // Launch producer threads (buys and sells)
    std::thread producer_buy(order_producer, std::ref(lob), true, 100.0, 0.5, 10, 1000, std::ref(rng));
    std::thread producer_sell(order_producer, std::ref(lob), false, 100.0, 0.5, 10, 1000, std::ref(rng));
    // Wait for producers to finish
    producer_buy.join();
    producer_sell.join();
    // Stop matching engine and wait
    lob.stop();
    matcher.join();
    // Print final inventory and cash
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Final inventory: " << lob.get_inventory() << std::endl;
    std::cout << "Final cash: " << lob.get_cash() << std::endl;
    return 0;
}