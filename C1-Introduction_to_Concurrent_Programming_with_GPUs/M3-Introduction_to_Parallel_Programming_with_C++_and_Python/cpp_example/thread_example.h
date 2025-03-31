//
// Created by Chancellor Pascale on 1/29/21.
//
#include <iostream>
#include <thread>
#include <chrono>

#ifndef CPP_EXAMPLES_THREAD_EXAMPLE_H
#define CPP_EXAMPLES_THREAD_EXAMPLE_H

void doWork(int threadIndex);
void executeThreads();
void executeAndDetachThread();

#endif //CPP_EXAMPLES_THREAD_EXAMPLE_H
