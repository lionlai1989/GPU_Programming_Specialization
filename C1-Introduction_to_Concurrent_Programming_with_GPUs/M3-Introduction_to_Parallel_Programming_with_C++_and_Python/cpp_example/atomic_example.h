//
// Created by Chancellor Pascale on 1/29/21.
//
#include <iostream>
#include <atomic>
#include <thread>

#ifndef CPP_EXAMPLES_ATOMIC_EXAMPLE_H
#define CPP_EXAMPLES_ATOMIC_EXAMPLE_H

std::atomic<bool> ready;

void doWorkWithAtomicBoolean(int threadIndex);
void executeThreadsWithAtomicBoolean(int numThreads);
void doWorkWithAtomicThreadFence(int threadIndex);
void executeWithAtomicThreadFence(int numThreads);

#endif //CPP_EXAMPLES_ATOMIC_EXAMPLE_H
