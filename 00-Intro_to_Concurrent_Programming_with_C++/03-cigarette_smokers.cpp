/**
 * In the Cigarette Smokers Problem, is it possible that all three smokers are
 * smoking at the same time?
 *
 * Yes.
 */

#include <array>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <semaphore> // std::counting_semaphore
#include <thread>

enum Ingredient { TOBACCO = 0, PAPER = 1, MATCHES = 2 };

class Agent {
  private:
    std::tuple<int, int> pick_two_ingredients(std::uniform_int_distribution<int> &dist, std::mt19937_64 &rng) {
        // Pick two distinct ingredients in {0,1,2}
        int first = dist(rng);
        int second;
        do {
            second = dist(rng);
        } while (second == first);

        return std::make_tuple(first, second);
    }

  public:
    // “Ingredient X is on the table” semaphores (max count = 1, initial = 0)
    // This information is shared between the agent and the smokers.
    std::counting_semaphore<1> tobacco;
    std::counting_semaphore<1> paper;
    std::counting_semaphore<1> matches;

    Agent() : tobacco(0), paper(0), matches(0) {}

    /**
     * The agent's main loop. Repeat:
     *     1. Randomly pick two ingredients and place them on the table
     *     2. Wait until the two ingredients are taken by a smoker
     */
    void operator()() {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 2); // 0, 1, 2

        while (true) {
            // Pick two ingredients and place them on the table
            auto [first, second] = pick_two_ingredients(dist, rng);

            // Release the corresponding two semaphores
            if ((first == TOBACCO && second == PAPER) || (first == PAPER && second == TOBACCO)) {
                std::cout << "[Agent] Placing TOBACCO + PAPER on the table.\n";
                tobacco.release();
                paper.release();
            } else if ((first == TOBACCO && second == MATCHES) || (first == MATCHES && second == TOBACCO)) {
                std::cout << "[Agent] Placing TOBACCO + MATCHES on the table.\n";
                tobacco.release();
                matches.release();
            } else {
                std::cout << "[Agent] Placing PAPER + MATCHES on the table.\n";
                paper.release();
                matches.release();
            }

            // Wait until the two ingredients are taken by a smoker. The agent does not need to know
            // which smoker took the ingredients. Once the two ingredients are taken, the agent can
            // place the next two ingredients on the table.
        }
    }
};

/**
 * Each smoker owns one ingredient (0=tobacco,1=paper,2=matches).
 * Its operator()() blocks until both “other ingredients” appear. When the other two ingredients
 * appear, the smoker takes the two ingredients from the table and smokes.
 */
class Smoker {
  private:
    int ingredient;
    Agent &agent_ref; // Explain why an lvalue reference to an Agent is needed here.

  public:
    Smoker(int ingredient_, Agent &agent_) : ingredient(ingredient_), agent_ref(agent_) {}

    void operator()() {
        const char *have_str = (ingredient == TOBACCO ? "tobacco" : ingredient == PAPER ? "paper" : "matches");

        while (true) {
            // Wait until the other two ingredients are on the table

            // Take the other two ingredients from the table

            // Simulate smoking
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Finish smoking
        }
    }
};

// g++ 03-cigarette_smokers.cpp -std=c++20 -pthread -Wall -Wextra && ./a.out
int main() {
    std::cout << "=== Cigarette Smokers Problem ===\n";

    // Create the shared Agent
    Agent agent;

    // Create three Smoker objects, each holding one ingredient
    Smoker smoker0(TOBACCO, agent);
    Smoker smoker1(PAPER, agent);
    Smoker smoker2(MATCHES, agent);

    // Launch the agent thread
    std::thread agent_thread(agent);

    // Launch three smoker threads
    std::thread smoker_tob_thread(smoker0);
    std::thread smoker_pap_thread(smoker1);
    std::thread smoker_mat_thread(smoker2);

    // Join everything (they run forever)
    agent_thread.join();
    smoker_tob_thread.join();
    smoker_pap_thread.join();
    smoker_mat_thread.join();

    return 0;
}
