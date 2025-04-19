#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

class Philosopher {
  private:
    std::string name;
    std::mutex &left_fork;
    std::mutex &right_fork;
    static bool running;

  public:
    Philosopher(const std::string &name, std::mutex &left, std::mutex &right)
        : name(name), left_fork(left), right_fork(right) {}

    static void setRunning(bool value) { running = value; }

    void operator()() {
        while (running) {
            // Thinking
            std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 10000 + 3000));
            std::cout << name << " is hungry." << std::endl;
            dine();
        }
    }

    void dine() {
        std::mutex *fork1 = &left_fork;
        std::mutex *fork2 = &right_fork;

        while (running) {
            fork1->lock();
            bool locked = fork2->try_lock();
            if (locked) {
                break;
            }
            fork1->unlock();
            std::cout << name << " swaps forks" << std::endl;
            std::swap(fork1, fork2);
        }

        if (!running) {
            fork1->unlock();
            return;
        }

        dining();
        fork2->unlock();
        fork1->unlock();
    }

    void dining() {
        std::cout << name << " starts eating" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 9000 + 1000));
        std::cout << name << " finishes eating and leaves to think." << std::endl;
    }
};

bool Philosopher::running = true;

int main() {
    std::vector<std::mutex> forks(5);
    std::vector<std::string> philosopher_names = {"Aristotle", "Kant", "Buddha", "Marx", "Russel"};

    std::vector<std::thread> philosophers;
    for (int i = 0; i < 5; ++i) {
        philosophers.emplace_back(Philosopher(philosopher_names[i], forks[i], forks[(i + 1) % 5]));
    }

    // Run for 100 seconds
    std::this_thread::sleep_for(std::chrono::seconds(100));
    Philosopher::setRunning(false);

    for (auto &philosopher : philosophers) {
        philosopher.join();
    }

    std::cout << "Now we're finishing." << std::endl;
    return 0;
}