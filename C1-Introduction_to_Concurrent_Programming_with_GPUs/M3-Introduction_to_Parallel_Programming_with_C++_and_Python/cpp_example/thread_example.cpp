#include "thread_example.h"

void doWork(int threadIndex){
	std::cout << "Performing work for thread: " << threadIndex << "\n";
}

void executeThreads(){
	std::cout << "Starting threads\n";

    // Create threads
	std::thread zeroth(doWork, 0);
	std::thread first(doWork, 1);
	std::thread second(doWork, 2);

	std::cout << "Press a key to let program proceed\n";
    getchar();
	
	std::cout << "Joining threads\n";
	
    // Wait for threads to complete
	second.join();
	zeroth.join();
	first.join();
	
	std::cout << "Completed all threads\n";
}

void executeAndDetachThread() {
    std::thread thread(doWork, 0);
    thread.detach();
}

int main(){
	std::cout << "Pay attention to the fact that newlines will not always be added at the end of lines if multiple <<'s\n";
	executeThreads();

    executeAndDetachThread();

	std::cout << "Sleeping for 1 second\n";
    std::this_thread::sleep_for (std::chrono::seconds(1));
    
	std::cout << "Press a key to let program proceed\n";
    getchar();
    return 0;
}