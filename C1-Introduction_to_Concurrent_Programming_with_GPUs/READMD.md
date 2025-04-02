## Introduction to Concurrent Programming with GPUs

### M2: Core Principles of Parallel Programming on CPUs and GPUs

1. concurrent programming pitfalls:
- race conditions
- resource contention
- dead lock
- live lock
- resource under- and over-utilization

2. concurrent programming challenges:
- dining philosophers
- producer-consumer
- sleeping barber
- data and code synchronization
  
3. concurrent programming patterns:
- divide and conquer
- map-reduce
- repository
- pipelines and workflows: directed acyclic graph
- recursion:
- 

4. Serial Versus Parallel Code and Flynn's Taxonomy
- serial search pseudocode
- parallel search pseudocode
- Flynn's Taxonomy

### M3: Introduction to Parallel Programming with C++ and Python

1. Syntax and Parallel Programming Patterns of Python3

explain what are the following three. differences and commons.
- `threading`
- `asyncio`
- `multiprocessing`
  - interprocess communication, queue and pipe
  - shared memory objects, Value and Array
  - Pool: a llow for managing multiple workers

2. Syntax and Parallel Programming Patterns of C++

- `std::thread`
- `std::mutex`
- `std::atomic`
- `std::future`



### M4: NVidia GPU Hardware and Software

1. understand nvcc workflows
2. CUDA Runtime API
3. CUDA Driver API


### M5: Introduction to GPU Programming

1. understand cuda keywords
__host__
__global__
__device__
2. 

### References:

- [The Dining Philosophers Problem With Ron Swanson](https://www.adit.io/posts/2013-05-11-The-Dining-Philosophers-Problem-With-Ron-Swanson.html)

- [Parallel Programming Patterns by Eun-Gyu Kim](https://snir.cs.illinois.edu/patterns/patterns.pdf)

- [OCaml Programming](https://cs3110.github.io/textbook/cover.html)

- [Parallel Processing in Python – A Practical Guide with Examples](https://www.machinelearningplus.com/python/parallel-processing-python/)

- [CMU's An Introduction to Parallel Computing in C++](https://www.cs.cmu.edu/afs/cs/academic/class/15210-f18/www/pasl.html)

- [CUDA Books archive](https://developer.nvidia.com/cuda-books-archive)

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
