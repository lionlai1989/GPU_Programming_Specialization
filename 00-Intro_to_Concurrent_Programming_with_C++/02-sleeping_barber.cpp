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

constexpr int CUSTOMERS_SEATS = 10; // Number of seats in BarberShop
constexpr int BARBERS = 3;          // Number of Barbers working

std::condition_variable barber_cv;
std::atomic<bool> SHOP_OPEN{true};

template <typename T> class BlockingQueue {
  private:
    const int capacity;
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv_not_full, cv_not_empty;
    std::condition_variable cv_empty;

  public:
    BlockingQueue(int cap) : capacity(cap) {}

    // Enqueue value, blocking if queue is full
    void enqueue(T value) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [&]() { return q.size() < capacity; });
        q.push(value);
        cv_not_empty.notify_one();
    }

    // Dequeue value, blocking if queue is empty
    T dequeue() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [&]() { return !q.empty(); });
        T value = q.front();
        q.pop();
        cv_not_full.notify_one();
        if (q.empty()) {
            cv_empty.notify_one();
        }
        return value;
    }

    void wait_until_empty() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_empty.wait(lock, [&]() { return q.empty(); });
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return q.empty();
    }
    bool full() {
        std::lock_guard<std::mutex> lock(mtx);
        return q.size() == capacity;
    }
};

class Customer {
  public:
    std::string name;
    int haircut_time;

    Customer(std::string name, int haircut_time) : name(name), haircut_time(haircut_time) {}

    void trim() {
        std::cout << name << " haircut started." << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(haircut_time)));

        std::cout << name << " haircut finished. Took " << haircut_time << " seconds" << std::endl;
    }
};

class Barber {
  private:
    BlockingQueue<Customer> &bq;
    int id;
    bool is_sleeping;

  public:
    Barber(BlockingQueue<Customer> &bq, int id) : bq(bq), id(id), is_sleeping(true) {}

    void run() {
        while (SHOP_OPEN) {
            Customer c = bq.dequeue();
        }
    }
};

void wait() {
    // Wait for 5 to 10 seconds.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(5, 10);
    std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(dis(gen))));
}

std::string random_name() {
    static constexpr char CHARACTERS[] = "abcdefghijklmnopqrstuvwxyz";
    static constexpr std::size_t CHAR_COUNT = sizeof(CHARACTERS) - 1;

    // Seed with a real random value, if available
    static thread_local std::random_device rd;
    static thread_local std::mt19937 engine(rd());
    std::uniform_int_distribution<std::size_t> dist(0, CHAR_COUNT - 1);

    std::string result;
    result.reserve(5);
    for (std::size_t i = 0; i < 5; ++i) {
        result += CHARACTERS[dist(engine)];
    }
    return result;
}

int random_seconds() {
    // Returns a random integer in [10, 30]
    static constexpr int MIN = 10;
    static constexpr int MAX = 30;

    static thread_local std::random_device rd;
    static thread_local std::mt19937 engine(rd());
    std::uniform_int_distribution<int> dist(MIN, MAX);
    return dist(engine);
}

// g++ 02-sleeping_barber.cpp -std=c++17 -pthread -Wall -Wextra && ./a.out
int main() {
    BlockingQueue<Customer> customer_queue(CUSTOMERS_SEATS);
    std::vector<std::thread> barber_threads;
    std::vector<std::thread> customer_threads;

    // Create barbers
    for (int i = 0; i < BARBERS; ++i) {
        barber_threads.emplace_back([&customer_queue, i]() {
            Barber barber(customer_queue, i);
            barber.run();
        });
    }

    // Create customers. There are only 10 seats for customers, but 20 customers
    // will be spawned unpredictably.
    for (int i = 0; i < 20; ++i) {
        Customer c(random_name(), random_seconds());

        // From every 5 to 10 seconds, there will be a customer entering.
        wait();

        if (!customer_queue.full()) {
            customer_queue.enqueue(std::move(c));
        } else {
            std::cout << "Queue full, customer " << c.name << "  has left." << std::endl;
        }
    }

    // Wait for all customers to be served
    customer_queue.wait_until_empty();

    // Close the shop
    SHOP_OPEN = false;
    barber_cv.notify_all();

    for (auto &thread : barber_threads) {
        thread.join();
    }

    return 0;
}