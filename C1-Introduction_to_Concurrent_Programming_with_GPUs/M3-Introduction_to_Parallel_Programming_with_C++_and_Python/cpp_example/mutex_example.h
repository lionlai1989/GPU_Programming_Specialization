//
// Created by Chancellor Pascale on 1/29/21.
//
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <string>

#ifndef CPP_EXAMPLES_MUTEX_H
#define CPP_EXAMPLES_MUTEX_H

std::mutex sharedMutex;

void doWorkWithMutexLock(int threadIndex);
void executeThreadsWithMutexLock(int numThreads);
void doWorkWithMutexTryLock(int threadIndex);
void executeAndDetachThreadsWithMutexTryLock(int numThreads);

#endif //CPP_EXAMPLES_MUTEX_H
