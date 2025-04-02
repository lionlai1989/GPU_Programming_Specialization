#include "calculation.h"

/*
 * CUDA Kernel Device code
 *
 * Perform calculation on d_d using constant d_v and place results into d_i (examples are multiplying by d_v, or dividing by d_v, etc.)
 */
__global__ void calculation(int *d_d, int *d_i, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d_i[i] = d_d[i] + d_v;
    }
}

__host__ int *allocateRandomHostMemory(int numElements)
{
    size_t size = numElements * sizeof(int);

    // Allocate the host input vectors h_d
    int *h_d = (int *)malloc(size);

    // Verify that allocations succeeded
    if (h_d == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_d[i] = rand() % 255;
    }

    return h_d;
}

// Based heavily on https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
// Presumes that there is no header in the csv file
__host__ std::tuple<int * , int>readCsv(std::string filename)
{
    std::vector<int> tempResult;
    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Extract each integer
        while(ss >> val){
            tempResult.push_back(val);
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
        }
    }

    // Close file
    myFile.close();
    int numElements = tempResult.size();
    int result[numElements];
    // Copy all elements of vector to array
    std::copy(tempResult.begin(), tempResult.end(), result);

    return {result, numElements};
}

__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements)
{
    // Allocate the device input vector A
    int *d_d = NULL;
    size_t size = numElements * sizeof(int);
    cudaError_t err = cudaMalloc(&d_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_d (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_i;
    err = cudaMalloc(&d_i, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_i (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return {d_d, d_i};
}

__host__ void copyFromHostToDevice(int h_v, int *h_d, int *d_d, int numElements)
{
    size_t size = numElements * sizeof(int);

    cudaError_t err = cudaMemcpy(d_d, h_d, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyToSymbol(d_v, &h_v, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(int *d_d, int *d_i, int numElements, int threadsPerBlock)
{
    // Launch the search CUDA Kernel
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    calculation<<<blocksPerGrid, threadsPerBlock>>>(d_d, d_i, numElements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(int *d_i, int *h_i, int numElements)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = numElements * sizeof(int);

    cudaError_t err = cudaMemcpy(h_i, d_i, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_i from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(int *d_d, int *d_i)
{

    cudaError_t err = cudaFree(d_d);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_d (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_i);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_i (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
 * Host main routine
 */
int main(int argc, char *argv[])
{
    srand(time(0));
    int numElements = 10;
    int h_v = -1;
    int *h_d;
    int threadsPerBlock = 256;

    bool sortInputData = true;

    // calculation.exe true|false threadsPerBlock numElements searchValue inputFilename

    if(argc > 1)
    {
        std::string sortInputDataStr(argv[1]);
        if(sortInputDataStr == "false")
        {
            sortInputData = false;
        }
    }

    if(argc > 2)
    {
        threadsPerBlock = atoi(argv[2]);
        if(argc > 3)
        {
            numElements = atoi(argv[3]);
        }
    }
    if(argc > 4)
    {
        h_v = atoi(argv[4]);
        std::string inputFilename(argv[5]);
        tuple<int *, int>csvData = readCsv(inputFilename);
        h_d = get<0>(csvData);
        numElements = get<1>(csvData);
    }
    else 
    {
        h_d = allocateRandomHostMemory(numElements);
        // This is a bit hard coded, could consider coming up with another randomization scheme
        h_v = rand() % 255;
    }

    if(sortInputData)
    {
        sort(h_d, h_d + numElements);
    }

    int *h_i = (int *)malloc(numElements * sizeof(int));
    cout << "Data: ";
    for (int i = 0; i < numElements; ++i)
    {
        cout << h_d[i] << " ";
        h_i[i]=0;
    }
    cout << "\n";

    printf("Searching for value: %d \n", h_v);
    auto[d_d, d_i] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_v, h_d, d_d, numElements);

    executeKernel(d_d, d_i, numElements, threadsPerBlock);

    copyFromDeviceToHost(d_i, h_i, numElements);

    cout << "Calculation results: ";
    for (int i = 0; i < numElements; ++i)
    {
        cout << h_i[i] << "\n";
    }

    deallocateMemory(d_d, d_i);

    cleanUpDevice();
    return 0;
}