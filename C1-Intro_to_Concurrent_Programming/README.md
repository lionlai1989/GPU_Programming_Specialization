## Introduction to Concurrent Programming with GPUs

### M2: Core Principles of Parallel Programming on CPUs and GPUs

#### Concurrent Programming Pitfalls

- **Race Conditions:** Occur when two or more threads access shared data
  simultaneously, causing unpredictable results if not synchronized. 

  ```pseudocode
  shared counter = 0

  // Thread A
  counter = counter + 1

  // Thread B
  counter = counter + 1

  // Expected final value: 2
  // Possible final value due to race condition: 1
  ```

- **Resource Contention:** Multiple threads compete for the same resources,
   leading to degraded performance.

  ```  
  shared resource R is available

  // Thread A
  acquire(R)         // Wait until R is free
  use(R)             // Perform operation using the resource
  release(R)

  // Thread B
  acquire(R)         // Wait until R is free
  use(R)             // Perform operation using the resource
  release(R)
  ```

- **Deadlock:** A state where threads wait indefinitely for resources held by
  each other, halting progress.

  ```
  // Thread A
  lock(Lock1)
  wait()              // Simulate processing
  lock(Lock2)
  // Critical section
  unlock(Lock2)
  unlock(Lock1)

  // Thread B
  lock(Lock2)
  wait()              // Simulate processing
  lock(Lock1)
  // Critical section
  unlock(Lock1)
  unlock(Lock2)
  ```

- **Livelock:** Threads continuously change state in response to others without
  making actual progress.

  ```
  shared side = "A"   // Indicates the side each thread prefers

  // Thread A
  while (not passed) {
      if (side == "A") {
          yield();      // Give way
          side = "B";
      } else {
          pass();       // Try to proceed
          break;
      }
  }

  // Thread B
  while (not passed) {
      if (side == "B") {
          yield();      // Give way
          side = "A";
      } else {
          pass();       // Try to proceed
          break;
      }
  }
  ```

- **Under- and Over-utilization:** Underutilization means resources are not
  fully used, while over-utilization overloads resources, hurting performance.

  ```
  # Under-Utilization Example
  for each task in taskQueue:
      process(task)
      sleep(long_duration)  // Excess delay causing idle resources

  # Over-Utilization Example
  for each task in taskQueue:
      spawn new thread to process(task)
  ```

#### Concurrent Programming Challenges

- **Dining Philosophers:** A model problem illustrating resource sharing and
  potential deadlock among competing processes.

- **Producer-Consumer:** Describes the challenge of balancing the rate at which
  producers generate data and consumers process it.

- **Sleeping Barber:** Models resource management with a service process
  (barber) that sleeps until work (customers) arrives.
  
#### Concurrent Programming Patterns

- **Divide and Conquer:** Breaks a problem into smaller sub-problems, solves
  them in parallel, and combines the results.

- **MapReduce:** Splits data processing into two steps—mapping (processing data
  segments) and reducing (aggregating results).

- **Repository:** A pattern managing shared resources in a central location to
  coordinate access.

- **Pipelines/Workflows (DAG):** Arranges tasks in sequential stages connected
  by data dependencies, often represented as a directed acyclic graph.

- **Recursion:** A function calls itself to solve smaller instances of a
  problem.

#### Serial vs. Parallel Code and Flynn’s Taxonomy

- **Serial Search:** A sequential method for searching through data.

- **Parallel Search:** Distributes the search task across multiple threads to
  reduce search time.

- **Flynn’s Taxonomy:** A classification of computer architectures based on the
  number of concurrent instruction and data streams (SISD, SIMD, MISD, MIMD).

### M3: Introduction to Parallel Programming with C++ and Python

#### Python Parallel Programming Concepts

- **`threading`:** Executes multiple threads within a single process, ideal for I/O-bound tasks.

- **`asyncio`:** Uses an event loop for asynchronous I/O tasks, enabling cooperative multitasking.

- **`multiprocessing`:** Launches separate processes to utilize multiple CPUs, including:
  
  - Interprocess Communication: Mechanisms like queues and pipes to exchange data.

  - Shared Memory (Value and Array): Allows multiple processes to access common data.

  - Pool: Manages a group of worker processes for executing tasks concurrently.


#### C++ Parallel Programming Concepts

- **`std::thread`:** Provides an interface to create and manage threads.

- **`std::mutex`:** Implements mutual exclusion to protect shared data from
  concurrent access.

- **`std::atomic`:** Ensures operations on shared data are atomic, avoiding the
  need for locks in certain cases.

- **`std::future`:** Represents the result of an asynchronous operation,
  allowing retrieval once it’s available.

### M4: NVIDIA GPU Hardware and Software Concepts

- **NVCC Workflows:** Processes for compiling CUDA applications using NVIDIA’s NVCC compiler driver.

- **CUDA Runtime API:** A high-level interface to manage GPU operations, memory allocation, and kernel launches.

- **CUDA Driver API:** A lower-level interface that provides finer control over GPU resources and execution.


### M5: Introduction to GPU Programming

- **`__host__`:** Indicates that a function is executed on the CPU.

- **`__global__`:** Declares a kernel function callable from the host but executed on the GPU.

- **`__device__`:** Specifies that a function runs on the GPU and is callable only from device code.

- **Memory Management:** Understands various GPU memory types (global, shared, constant, texture).

- **Execution Configuration:** Defines the grid, block, and thread structure to launch parallel GPU tasks.

### References:

- [The Dining Philosophers Problem With Ron Swanson](https://www.adit.io/posts/2013-05-11-The-Dining-Philosophers-Problem-With-Ron-Swanson.html)

- [Parallel Programming Patterns by Eun-Gyu Kim](https://snir.cs.illinois.edu/patterns/patterns.pdf)

- [OCaml Programming](https://cs3110.github.io/textbook/cover.html)

- [Parallel Processing in Python – A Practical Guide with Examples](https://www.machinelearningplus.com/python/parallel-processing-python/)

- [CMU's An Introduction to Parallel Computing in C++](https://www.cs.cmu.edu/afs/cs/academic/class/15210-f18/www/pasl.html)

- [CUDA Books archive](https://developer.nvidia.com/cuda-books-archive)

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
