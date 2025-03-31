//
// Created by Chancellor Pascale on 1/29/21.
//

#include "mutex_example.h"

void doWorkWithMutexLock(int threadIndex){
	sharedMutex.lock();
	std::cout << "Performing work for thread: " << threadIndex << "\n";
	std::cout << "Sleeping for 1 second\n";
	std::this_thread::sleep_for (std::chrono::seconds(1));
	sharedMutex.unlock();
}

void executeThreadsWithMutexLock(int numThreads){
	std::thread threads[numThreads];
	std::cout << "Starting " << numThreads << " threads to use lock with shared mutex\n";
	int counter = 0;
	// spawn n threads
	for (int i=0; i<numThreads; ++i)
		threads[i] = std::thread(doWorkWithMutexLock, i);
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	
	std::cout << "Joining threads\n";
	for (int i=0; i<numThreads; ++i)
		threads[i].join();
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	std::cout << "Completed all threads\n";
}

// Based on http://www.cplusplus.com/reference/mutex/mutex/try_lock/
void doWorkWithMutexTryLock(int threadIndex){
	int counter = 0;
	while(!sharedMutex.try_lock()){
		// unlike cplusplus.com this will count how many times lock couldn't be acquired across multiple threads
		counter++;
		std::this_thread::sleep_for (std::chrono::seconds(1));
	}
	std::cout << "Counter for work for thread: " << threadIndex << " is " << counter << "\n";
	sharedMutex.unlock();
}

// Based on http://www.cplusplus.com/reference/mutex/mutex/try_lock/
void executeAndDetachThreadsWithMutexTryLock(int numThreads){
	std::thread threads[numThreads];
	std::cout << "Starting " << numThreads << " threads with try_lock on shared mutex\n";
	int counter = 0;
	// spawn n threads
	for (int i=0; i<numThreads; ++i)
		threads[i] = std::thread(doWorkWithMutexTryLock, i);
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	
	std::cout << "Detaching threads with try_lock on shared mutex\n";
	for (int i=0; i<numThreads; ++i)
		threads[i].detach();
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
}

int main(int argc, char *argv[]){
	int numThreads = 3;
	if(argc > 1){
		numThreads = atoi(argv[1]);
	}
	std::cout << "Pay attention to the fact that newlines will not always be added at the end of lines if multiple <<'s\n";
	executeThreadsWithMutexLock(numThreads);
	
	executeAndDetachThreadsWithMutexTryLock(numThreads);

	std::cout << "Sleeping for 1 second\n";
	std::this_thread::sleep_for (std::chrono::seconds(1));
	
	std::cout << "Press a key to let program proceed\n";
	getchar();
	return 0;
}
