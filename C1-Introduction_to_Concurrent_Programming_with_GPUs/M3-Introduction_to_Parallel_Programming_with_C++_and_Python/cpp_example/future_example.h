//
// Created by Chancellor Pascale on 1/30/21.
//
#include <iostream>
#include <future>
#include <thread>

#ifndef CPP_EXAMPLES_FUTURE_EXAMPLE_H
#define CPP_EXAMPLES_FUTURE_EXAMPLE_H

void doWorkWithFutures(std::future<int>& fut);
void executeThreadsWithFutures();
int doWorkWithAsync(int x);
void executeWithAsync(int numFutures);

#endif //CPP_EXAMPLES_FUTURE_EXAMPLE_H