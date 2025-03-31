#include "assignment.h"

// Global variable definitions (initialize with parentheses)
std::atomic<int> currentTicketNumber(0);
std::string currentPartId;
std::string currentUser;
int currentNumThreads;

// Synchronization primitives
std::mutex mtx;
std::condition_variable cv;

void executeTicketingSystemParticipation(int ticketNumber)
{
    std::cout << "Current thread ticket number: " << ticketNumber << "\n";

    std::string outputFileName = "output-" + currentPartId + ".txt";
    std::ofstream outputFile(outputFileName, std::ofstream::app);
    if (!outputFile.is_open())
    {
        std::cerr << "Error opening output file: " << outputFileName << "\n";
        return;
    }
    outputFile << "C++11: Thread retrieved ticket number: " << ticketNumber << " started.\n";

    // Wait until it's this thread's turn using a lambda predicate.
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]()
                { return currentTicketNumber.load() == ticketNumber; });
    }

    // Once it's this thread's turn, print the completion message.
    outputFile << "C++11: Thread with ticket number: " << ticketNumber << " completed.\n";
    outputFile.close();

    // Increment currentTicketNumber and notify waiting threads.
    {
        std::lock_guard<std::mutex> lock(mtx);
        ++currentTicketNumber;
    }
    cv.notify_all();
}

int runSimulation()
{
    int result = 0;
    std::string userFromFile = getUsernameFromUserFile();
    if (USERNAME == currentUser && USERNAME == userFromFile)
    {
        std::cout << "Simple user verification completed successfully.\n";
        std::vector<std::thread> threads;

        // Create threads; each thread gets a unique ticket number.
        for (int threadIndex = 0; threadIndex < currentNumThreads; ++threadIndex)
        {
            threads.push_back(std::thread([threadIndex]()
                                          { executeTicketingSystemParticipation(threadIndex); }));
        }

        // Wait for all threads to complete.
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }

        std::cout << "All threads completed.\n";
    }
    else
    {
        std::cout << "Simple user verification failed, code will not be executed.\n";
    }
    return result;
}

std::string getUsernameFromUserFile()
{
    std::string line;
    std::ifstream userFile(".user");
    if (userFile.is_open())
    {
        std::getline(userFile, line);
        userFile.close();
    }
    else
    {
        std::cerr << "Error opening .user file.\n";
    }
    std::cout << "User from .user file: " << line << "\n";
    return line;
}

int manageTicketingSystem(std::vector<std::thread> &threads)
{
    // In this implementation, thread joining is handled in runSimulation().
    // This function is provided for compatibility.
    return 0;
}

int main(int argc, char *argv[])
{
    int numThreads = 1;
    std::string user = "Lion";
    std::string partId = "test";
    std::cout << "Starting assignment main function\n";

    if (argc > 3)
    {
        std::cout << "Parsing command line arguments\n";
        numThreads = std::atoi(argv[1]);
        user = argv[2];
        partId = argv[3];
    }

    currentNumThreads = numThreads;
    currentUser = user;
    currentPartId = partId;

    return runSimulation();
}
