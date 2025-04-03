/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "search.h"

/*
 * CUDA Kernel Device code
 *
 * Search passed data set for a float value and if the value is at the thread index set the foundIndex value
 */
__global__ void search(int *d_d, int *d_i, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        int inputVal = d_d[i];
        if (inputVal == d_v)
        {
            // d_i[0] = i;
            atomicCAS(&d_i[0], -1, i);
        }
    }
}

__host__ int *allocateRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(int);

    // Allocate the host input vector A
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
        h_d[i] = rand();
    }

    return h_d;
}

// Based heavily on https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
// Presumes that there is no header in the csv file
__host__ std::tuple<int *, int> readCsv(std::string filename)
{
    std::vector<int> tempResult;
    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

    // Read data, line by line
    while (std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Extract each integer
        while (ss >> val)
        {
            tempResult.push_back(val);
            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',')
                ss.ignore();
        }
    }

    // Close file
    myFile.close();
    int numElements = tempResult.size();
    // int result[numElements];
    int *result = new int[numElements];
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
    cudaMalloc((void **)&d_i, sizeof(int));

    return {d_d, d_i};
}

__host__ void copyFromHostToDevice(int h_v, int *h_d, int h_i, int *d_d, int *d_i, int numElements)
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

    err = cudaMemcpy(d_i, &h_i, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy int d_i from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(int *d_d, int *d_i, int numElements, int threadsPerBlock)
{
    // Launch the search CUDA Kernel
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    search<<<blocksPerGrid, threadsPerBlock>>>(d_d, d_i, numElements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(int *d_i, int &h_i)
{
    // Copy the device result int (found index) in device memory to the host result int
    // in host memory.
    cudaError_t err = cudaMemcpy(&h_i, d_i, sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy int d_i from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(int *h_d, int *d_d, int *d_i)
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

__host__ void outputToFile(std::string currentPartId, int *data, int numElements, int searchValue, int foundIndex)
{
    std::string outputFileName = "output-" + currentPartId + ".txt";
    // NOTE: Do not remove this output to file statement as it is used to grade assignment,
    // so it should be called by each thread
    std::ofstream outputFile;
    outputFile.open(outputFileName, std::ofstream::app);

    outputFile << "Data: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << data[i] << " ";
    outputFile << "\n";
    outputFile << "Searching for value: " << searchValue << "\n";
    outputFile << "Found Index: " << foundIndex << "\n";

    outputFile.close();
}

__host__ std::tuple<int, int, std::string, int, std::string, bool> parseCommandLineArguments(int argc, char *argv[])
{
    int numElements = 10;
    int h_v = -1;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    bool sortInputData = true;
    std::string inputFilename = "NULL";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-s") == 0)
        {
            if (value == "false")
            {
                sortInputData = false;
            }
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-n") == 0)
        {
            numElements = atoi(value.c_str());
        }
        else if (option.compare("-v") == 0)
        {
            h_v = atoi(value.c_str());
        }
        else if (option.compare("-f") == 0)
        {
            inputFilename = value;
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }

    return {numElements, h_v, currentPartId, threadsPerBlock, inputFilename, sortInputData};
}

__host__ std::tuple<int *, int, int> setUpSearchInput(std::string inputFilename, int numElements, int h_v, bool sortInputData)
{
    srand(time(0));
    int *h_d;

    if (inputFilename.compare("NULL") != 0)
    {
        tuple<int *, int> csvData = readCsv(inputFilename);
        h_d = get<0>(csvData);
        numElements = get<1>(csvData);
    }
    else
    {
        h_d = allocateRandomHostMemory(numElements);
    }

    if (sortInputData)
    {
        sort(h_d, h_d + numElements);
    }

    if (h_v == -1)
    {
        // Roll a 6-sided die if not a 6 generate from a random value in the input data otherwise pick a random value
        int diceRoll = rand() % 6;
        h_v = diceRoll < 5 ? h_d[rand() % numElements] : rand();
    }

    return {h_d, numElements, h_v};
}

/*
 * Host main routine
 * -s true|false - sort data prior to search
 * -n numElements - the number of elements of random data to create
 * -v searchValue - the value to search for in the data
 * -f inputFile - the file for non-random input data
 * -p currentPartId - the Coursera Part ID
 * -t threadsPerBlock - the number of threads to schedule for concurrent processing
 */
int main(int argc, char *argv[])
{
    int h_i = -1; // host input
    int *h_d;     // host data

    auto [numElements, h_v, currentPartId, threadsPerBlock, inputFilename, sortInputData] = parseCommandLineArguments(argc, argv);
    std::tuple<int *, int, int> searchInputTuple = setUpSearchInput(inputFilename, numElements, h_v, sortInputData);

    h_d = get<0>(searchInputTuple);
    numElements = get<1>(searchInputTuple);
    h_v = get<2>(searchInputTuple); // search value

    auto [d_d, d_i] = allocateDeviceMemory(numElements); // d_d: device_data, d_i: device_input
    copyFromHostToDevice(h_v, h_d, h_i, d_d, d_i, numElements);

    executeKernel(d_d, d_i, numElements, threadsPerBlock);

    copyFromDeviceToHost(d_i, h_i);
    outputToFile(currentPartId, h_d, numElements, h_v, h_i);

    deallocateMemory(h_d, d_d, d_i);

    cleanUpDevice();
    return 0;
}