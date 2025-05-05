#include <atomic>             // atomic
#include <chrono>             // chrono
#include <condition_variable> // condition_variable
#include <functional>         // ref
#include <iostream>           // cout
#include <mutex>              // mutex
#include <queue>              // queue
#include <random>             // random
#include <string>             // string
#include <thread>             // thread
#include <vector>             // vector

/**
 * From the beginning to the end of the program, there will be only one
 * BlockingQueue instance.
 */
class BlockingQueue {
  private:
    const int capacity;
    std::queue<int> q;
    std::mutex mtx;
    std::condition_variable cv_not_full, cv_not_empty;
    std::condition_variable cv_empty;

  public:
    BlockingQueue(int cap) : capacity(cap) {}

    // Enqueue value, blocking if queue is full
    void enqueue(int value) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [&]() { return q.size() < capacity; });
        q.push(value);
        cv_not_empty.notify_one();
    }

    // Dequeue value, blocking if queue is empty
    int dequeue() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [&]() { return !q.empty(); });
        int value = q.front();
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
};

class Producer {
    std::string name;
    std::vector<int> items;
    BlockingQueue &bq;

  public:
    Producer(std::string n, std::vector<int> xs, BlockingQueue &queue)
        : name(std::move(n)), items(std::move(xs)), bq(queue) {}

    void operator()() {
        for (int item : items) {
            // Simulate work
            std::this_thread::sleep_for(std::chrono::seconds(2));
            bq.enqueue(item);
            std::cout << "Producer " << name << " enqueued " << item << "\n";
        }
    }
};

class Consumer {
    std::string name;
    BlockingQueue &bq;

  public:
    Consumer(std::string n, BlockingQueue &queue) : name(std::move(n)), bq(queue) {}

    void operator()() {
        while (true) {
            int val = bq.dequeue();
            if (val < 0) {
                std::cout << "Consumer " << name << " received shutdown signal\n";
                break;
            }
            std::cout << "Consumer " << name << " processing " << val << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(val));
        }
    }
};

// g++ 01-producer_consumer.cpp -std=c++17 -pthread -Wall -Wextra && ./a.out
int main() {
    // Initialize random number generation
    std::mt19937 random_gen{std::random_device{}()};
    std::uniform_int_distribution<> dist(1, 5);

    BlockingQueue one_and_only_block_queue(10);
    int num_producers = 5;
    int num_consumers = 3;

    std::cout << "Launching Producers\n";
    std::vector<std::thread> producers;
    for (int i = 0; i < num_producers; ++i) {
        // Each producer produces 5 items
        std::vector<int> items;
        for (int j = 0; j < 5; ++j) {
            items.push_back(dist(random_gen));
        }

        /**
         * producers.push_back(Producer(std::to_string(i), items, one_and_only_block_queue));
         * no matching member function for call to 'push_back'
         *
         * Here, producers.push_back only accepts a std::thread (lvalue or
         * rvalue). But it's passing a Producer functor instead. There is no
         * implicit conversion from Producer to std::thread, so overload
         * resolution finds no matching.
         * std::thread is not copy-constructible (its copy ctor is deleted) but
         * is move-constructible.
         */
        std::thread prod_thread{Producer(std::to_string(i), items, one_and_only_block_queue)};
        producers.push_back(std::move(prod_thread)); // moves into the vector
    }
    std::cout << "Producers started\n";

    std::cout << "Launching Consumers\n";
    std::vector<std::thread> consumers;
    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back(Consumer(std::to_string(i), one_and_only_block_queue));
    }
    std::cout << "Consumers started\n";

    // Wait for all producers to finish
    for (auto &t : producers) {
        t.join();
    }
    std::cout << "All producers finished\n";

    // Explain why "poison pill" cannot be enqueued here.
    // It's possible that when the last consumer ends, the last cv_empty signal
    // is not captured by "wait_until_empty". Then, we have deadlock.

    // Wait until queue is drained
    one_and_only_block_queue.wait_until_empty();

    // Send one "poison pill" per consumer
    std::cout << "Send one poison pill per consumer\n";
    for (int i = 0; i < num_consumers; ++i) {
        one_and_only_block_queue.enqueue(-1);
    }

    // Wait for all consumers to finish
    for (auto &t : consumers) {
        t.join();
    }
    std::cout << "All consumers finished\n";
    return 0;
}
