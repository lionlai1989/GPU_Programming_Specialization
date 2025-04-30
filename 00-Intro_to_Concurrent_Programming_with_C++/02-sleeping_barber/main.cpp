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

constexpr int CUSTOMERS_SEATS = 15; // Number of seats in BarberShop
constexpr int BARBERS = 3;          // Number of Barbers working

std::mutex cout_mutex;
std::mutex queue_mutex;
std::condition_variable barber_cv;
std::atomic<bool> SHOP_OPEN{true};
std::atomic<int> Earnings{0};

class Customer {
  public:
    Customer() {
        static const std::vector<std::pair<std::string, int>> customer_types = {
            {"adult", 16}, {"senior", 7}, {"student", 10}, {"child", 7}};

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, customer_types.size() - 1);

        auto type = customer_types[dis(gen)];
        customer_type = type.first;
        rate = type.second;

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << customer_type << " rate." << std::endl;
        }
    }

    void trim() {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Customer haircut started." << std::endl;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 3.0);
        double haircut_time = dis(gen);

        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(haircut_time * 1000)));

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Haircut finished. Haircut took " << haircut_time << " seconds" << std::endl;
        }

        Earnings += rate;
    }

  private:
    std::string customer_type;
    int rate;
};

class Barber {
  public:
    Barber(std::queue<Customer> &queue, int id) : queue_(queue), id_(id), sleeping_(true) {}

    void run() {
        while (SHOP_OPEN) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            barber_cv.wait(lock, [this] { return !queue_.empty() || !SHOP_OPEN; });

            if (!SHOP_OPEN)
                break;

            if (queue_.empty()) {
                sleeping_ = true;
                {
                    std::lock_guard<std::mutex> cout_lock(cout_mutex);
                    std::cout << "------------------\nBarber " << id_ << " is sleeping\n------------------"
                              << std::endl;
                }
                continue;
            }

            sleeping_ = false;
            {
                std::lock_guard<std::mutex> cout_lock(cout_mutex);
                std::cout << "Barber " << id_ << " is awake." << std::endl;
            }

            Customer customer = queue_.front();
            queue_.pop();
            lock.unlock();

            customer.trim();
        }
    }

  private:
    std::queue<Customer> &queue_;
    int id_;
    bool sleeping_;
};

void wait() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dis(gen) * 1000)));
}

int main() {
    std::queue<Customer> customer_queue;
    std::vector<std::thread> barber_threads;
    std::vector<std::thread> customer_threads;

    // Create barbers
    for (int i = 0; i < BARBERS; ++i) {
        barber_threads.emplace_back([&customer_queue, i]() {
            Barber barber(customer_queue, i);
            barber.run();
        });
    }

    // Create customers
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "----" << std::endl;
            std::cout << "Queue size: " << customer_queue.size() << std::endl;
        }

        wait();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (customer_queue.size() < CUSTOMERS_SEATS) {
                customer_queue.emplace();
                barber_cv.notify_one();
            } else {
                std::lock_guard<std::mutex> cout_lock(cout_mutex);
                std::cout << "Queue full, customer has left." << std::endl;
            }
        }
    }

    // Wait for all customers to be served
    while (!customer_queue.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Close the shop
    SHOP_OPEN = false;
    barber_cv.notify_all();

    // Join all threads
    for (auto &thread : barber_threads) {
        thread.join();
    }

    std::cout << "Total earnings: €" << Earnings << std::endl;

    return 0;
}