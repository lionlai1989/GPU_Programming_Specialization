## CUDA at Scale for the Enterprise


### M2: Multiple CPU and GPU Systems

CUDA unified addressing and peer addressing

### M3: CUDA Events and Streams

blocking and non-blocking streams.

kernel can create an event and wake up another kernel.

use cases:
- create complex data workflows
- asynchronous data control
- process data in batches
- handle partial data
- resource management across one or more GPUs/CPUs

stream syntax: explain each function

- `__host__ ​cudaError_t cudaStreamCreate ( cudaStream_t* pStream )`
- `__host__ ​ __device__ ​cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags )`
- `__host__ ​cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )`
- `__host__ ​cudaError_t cudaStreamAddCallback ( cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags )`

stream memory management: need to use pinned memory or unified addressing memory

What are CUDA Events?

    They have a simple state, occurred or not, meaning has an event been fully reached.

    They can be used in synchronization between one or more streams and the host.

    Also, they can be used to synchronize streams to one another.

    This means that you can determine the order that kernels will execute in a fully or semi-sequential order.

How could events be used?

    Events can be used for multiple step and heterogenous computing workflows to have certain processing done on a device with the host waiting to do more complex processing upon completion and vice versa.

    Events can be used along with asynchronous and continuous memory copies to take input from users or external processes and change how and when GPU/host computation occurs.

    Enforce data coherency between GPUs using the same host memory.

    Ensure that GPU kernels that require other processing to be completed happens before they start.
    
### M4: Sorting Using GPUs

### M5


### References:

- [Nvidia Streams and Concurrency by Steve Rennich](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
- [CUDA Streams: Best Practices and Common Pitfalls](https://www.hpcadmintech.com/wp-content/uploads/2016/03/Carlo_Nardone_presentation.pdf)
- 
