#include <atomic>     // atomic
#include <chrono>     // chrono
#include <functional> // ref
#include <iostream>   // cout
#include <mutex>      // mutex, scoped_lock
#include <random>     // random_device, mt19937, uniform_int_distribution
#include <string>     // string
#include <thread>     // thread, this_thread
#include <vector>     // vector

/**
 * Declaring a variable static at namespace (file) scope gives it static storage
 * duration: it is initialized exactly once before main() begins and lives until
 * the program ends.
 * NOTE: calling dist(random_gen) concurrently from multiple threads is not
 * thread-safe. Both std::mt19937::operator() and distribution operator() mutate
 * internal state, so simultaneous access without a lock leads to data races and
 * undefined behavior.
 * thread_local storage ensures each thread gets its own copy of random_gen and
 * dist, avoiding any data races on their internal state.
 * Initialization of a thread_local variable happens once per thread, on first
 * encounter, with safe seeding via std::random_device.
 */
// static std::random_device rd;
// static std::mt19937 random_gen(rd());
// static std::uniform_int_distribution<> dist(1, 8);
thread_local std::mt19937 random_gen{std::random_device{}()};
thread_local std::uniform_int_distribution<> dist(1, 8);

class Philosopher {
  private:
    std::string name;

    /**
     * Reference members alias existing mutex objects rather than copy them.
     * References must be initialized in the constructor’s initializer list and
     * cannot be reseated later.
     * https://stackoverflow.com/questions/16888740/initializing-a-reference-variable-in-a-class-constructor?utm_source=chatgpt.com
     */
    std::mutex &left_fork;
    std::mutex &right_fork;

    int max_eating_times;
    int eat_count = 0;

    /**
     * A static member variable is shared across all instances (Philosophers) of the class.
     * A static member variable in C++ has static storage duration, meaning it
     * is allocated and initialized once before main() begins, remains alive for
     * the entire execution of the program, and is destroyed only when the
     * program terminates.
     * Here, it's just a declaration. No storage is reserved yet.
     * read by spawned threads and written by main thread.
     */
    static std::atomic<bool> running; // shared flag across all Philosophers instances.

  public:
    Philosopher(const std::string &n, std::mutex &left, std::mutex &right, int max_times)
        : name(n), left_fork(left), right_fork(right), max_eating_times(max_times) {}

    /**
     * A static member function does not have a `this` pointer, so it can be
     * called without an object: `Philosopher::setRunning(false);`
     */
    static void setRunning(bool value) {
        running.store(value); // static function can access static member
    }

    /**
     * When constructing a std::thread with a callable object f, the thread
     * begins execution immediately and invokes f.operator()() as its entry point.
     */
    void operator()() {
        while (running.load() && eat_count < max_eating_times) {
            // THINK
            std::cout << name << " is thinking.\n";
            // uses thread-local RNG
            std::this_thread::sleep_for(std::chrono::seconds(dist(random_gen)));

            std::cout << name << " is hungry.\n";
            dine();
        }
    }

    void dine() {
        /**
         * scoped_lock acquires both locks without risk of deadlock.
         */
        std::scoped_lock lock(left_fork, right_fork);

        // EAT
        eat_count += 1;
        std::cout << name << " starts eating (meal " << eat_count << ").\n";
        std::this_thread::sleep_for(std::chrono::seconds(dist(random_gen)));
        std::cout << name << " finishes eating and leaves to think.\n";
    }

    const std::string &getName() const { return name; }
    int getEatCount() const { return eat_count; }
};

/**
 * This out-of-class definition allocates the one shared bool object for
 * Philosopher::running in our program’s memory.
 */
std::atomic<bool> Philosopher::running{false};

int main() {
    const int N = 5;
    std::vector<std::mutex> forks(N);
    std::vector<std::string> names = {"Aristotle", "Kant", "Buddha", "Marx", "Russell"};

    Philosopher::setRunning(true);

    std::vector<Philosopher> philosophers;
    philosophers.reserve(N);
    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i) {
        philosophers.emplace_back(names[i], forks[i], forks[(i + 1) % N], 4);

        /**
         * Launch thread that calls the same Philosopher instance by reference.
         * The newly spawned thread calls `philosophers.back().operator()();`
         * Without std::ref, std::thread would copy the functor, and it ends up
         * inspecting dead counters in vector.
         */
        threads.emplace_back(std::ref(philosophers.back()));
    }

    // Let them eat 30 seconds.
    std::this_thread::sleep_for(std::chrono::seconds(30));
    Philosopher::setRunning(false);

    for (auto &t : threads) {
        t.join();
    }

    std::cout << "\nEating counts:\n";
    for (auto &ph : philosophers) {
        std::cout << ph.getName() << " ate " << ph.getEatCount() << " times.\n";
    }

    std::cout << "Simulation finished.\n";
    return 0;
}
