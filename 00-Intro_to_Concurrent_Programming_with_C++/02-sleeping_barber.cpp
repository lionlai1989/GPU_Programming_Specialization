#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int rand_int(int lo, int hi) {
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<> dist(lo, hi);
    return dist(gen);
}

// Forward declaration
class BarberShop;

class Customer {
  public:
    Customer(int id, BarberShop &shop);
    void operator()(); // thread entry

    int get_id() const { return id; }

  private:
    int id;
    BarberShop &shop;
};

class Barber {
  public:
    Barber(int id, BarberShop &shop);
    void operator()(); // thread entry

  private:
    int id;
    BarberShop &shop;
};

class BarberShop {
  public:
    BarberShop(int num_chairs) : num_chairs_(num_chairs), shop_open_(true), waiting_count_(0) {}

    // Called by Customer threads
    bool enter_shop(int customer_id) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (waiting_count_ >= num_chairs_) {
            // No free chair
            return false;
        }
        // Sit in waiting room
        waiting_.push(customer_id);
        ++waiting_count_;
        cv_barber_.notify_one(); // wake a barber
        cv_customer_.wait(lock, [&] {
            // wait until barber signals it's this customer's turn
            return serving_id_ == customer_id;
        });
        return true;
    }

    // Called by Barber threads
    bool next_customer(int &out_customer_id) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_barber_.wait(lock, [&] { return !waiting_.empty() || !shop_open_.load(); });
        if (!shop_open_.load() && waiting_.empty())
            return false; // no more work, time to go home

        out_customer_id = waiting_.front();
        waiting_.pop();
        --waiting_count_;
        serving_id_ = out_customer_id;
        cv_customer_.notify_one(); // tell that customer to proceed
        return true;
    }

    void close_shop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shop_open_ = false;
        }
        cv_barber_.notify_all();
    }

  private:
    friend class Customer;
    friend class Barber;

    const int num_chairs_;
    std::atomic<bool> shop_open_;
    std::mutex mtx_;
    std::condition_variable cv_barber_;
    std::condition_variable cv_customer_;

    std::queue<int> waiting_;
    int waiting_count_;
    int serving_id_; // ID of customer currently in chair
};

// Customer Implementation
Customer::Customer(int id, BarberShop &shop) : id(id), shop(shop) {}

void Customer::operator()() {
    // simulate random arrival delay
    std::this_thread::sleep_for(std::chrono::milliseconds(rand_int(200, 800)));

    if (!shop.enter_shop(id)) {
        std::cout << "Customer " << id << " leaves (no free chairs).\n";
        return;
    }

    // Now being served
    int haircut_ms = rand_int(500, 2000);
    std::cout << "Customer " << id << " is getting a haircut (" << haircut_ms << "ms).\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(haircut_ms));
    std::cout << "Customer " << id << " done.\n";
}

// Barber Implementation
Barber::Barber(int id, BarberShop &shop) : id(id), shop(shop) {}

void Barber::operator()() {
    while (true) {
        int cust_id;
        if (!shop.next_customer(cust_id)) {
            // shop closed and no customers
            break;
        }
        std::cout << "[Barber " << id << "] starts haircut for Customer " << cust_id
                  << ". Waiting remain: " << (shop.waiting_count_) << "\n";
        // actual haircut simulated in Customer thread
    }
    std::cout << "[Barber " << id << "] is going home.\n";
}

// g++ -std=c++17 -pthread -Wall -Wextra -pedantic 02-sleeping_barber.cpp && ./a.out
int main() {
    constexpr int NUM_BARBERS = 3;
    constexpr int NUM_CHAIRS = 5;
    constexpr int NUM_CUSTOMERS = 20;

    BarberShop shop(NUM_CHAIRS);

    // Start barber threads
    std::vector<std::thread> barber_threads;
    for (int i = 1; i <= NUM_BARBERS; ++i)
        barber_threads.emplace_back(Barber(i, shop));

    // Start customer threads
    std::vector<std::thread> customer_threads;
    for (int i = 1; i <= NUM_CUSTOMERS; ++i)
        customer_threads.emplace_back(Customer(i, shop));

    // Wait for all customers to finish (either served or left)
    for (auto &t : customer_threads)
        t.join();

    // Close shop and wake barbers so they can exit
    shop.close_shop();

    for (auto &t : barber_threads)
        t.join();

    std::cout << "Shop is now closed.\n";
    return 0;
}
