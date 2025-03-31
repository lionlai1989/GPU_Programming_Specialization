//
// Created by Chancellor Pascale on 1/30/21.
// Code based on http://www.cplusplus.com/reference/future/promise/
//
#include "future_example.h"

void doWorkWithFutures(std::future<int>& fut){
	int x = fut.get();
	std::cout << "Future index: " << x << " executing\n";
}

void executeThreadsWithFutures(){
	std::cout << "Starting a threads to use futures\n";
	
	std::promise<int> prom;                      // create promise
	std::future<int> fut = prom.get_future();    // engagement with future
	
	// spawn a threads
	std::thread th1(doWorkWithFutures, std::ref(fut));
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	
	prom.set_value(1);
	th1.join();
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	std::cout << "Completed all threads\n";
}

int doWorkWithAsync(int x){
	std::cout << "Squaring " << x << ". Please, wait...\n";
	return x*x;
}

void executeWithAsync(int numFutures){
	std::cout << "Using future with async to asynchronously execute squaring of values" << "\n";
	
	std::future<int> first = std::async (doWorkWithAsync, 1);
	std::future<int> second = std::async (doWorkWithAsync, 2);
	std::future<int> third = std::async (doWorkWithAsync, 3);
	
	int firstResult = first.get();
	std::cout << "Future first squared value: " << firstResult << "\n";
	int secondResult = second.get();
	std::cout << "Future second squared value: " << secondResult << "\n";
	int thirdResult = third.get();
	std::cout << "Future third squared value: " << thirdResult << "\n";
	
}

int main(int argc, char *argv[]){
	int numFutures = 3;
	if(argc > 1){
		numFutures = atoi(argv[1]);
	}
	std::cout << "Pay attention to the fact that newlines will not always be added at the end of lines if multiple <<'s\n";
	executeThreadsWithFutures();
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	
	executeWithAsync(numFutures);
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	
	return 0;
}