#ifndef ASSIGNMENT_H
#define ASSIGNMENT_H

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

const std::string USERNAME = "Lion";

// Global variable declarations
extern std::atomic<int> currentTicketNumber;
extern std::string currentPartId;
extern std::string currentUser;
extern int currentNumThreads;

// Synchronization primitives
extern std::mutex mtx;
extern std::condition_variable cv;

// Function declarations
void executeTicketingSystemParticipation(int ticketNumber);
int runSimulation();
std::string getUsernameFromUserFile();
int manageTicketingSystem(std::vector<std::thread> &threads);

#endif // ASSIGNMENT_H
