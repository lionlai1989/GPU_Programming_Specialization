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
#include "memory_copy.h"

__global__ void div(float *d_a, float *d_b, float *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        h_c[i] = d_a[i] / d_b[i];
    }
}

__host__ std::tuple<float *, float *> allocateRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(float);

    float *h_a;
    h_a = (float *)malloc(size);

    float *h_b;
    h_b = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        // Based on float random number generation code at https://stackoverflow.com/questions/686353/random-float-number-generation
        h_a[i] = LO_RAND + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI_RAND-LO_RAND)));
        h_b[i] = LO_RAND + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI_RAND-LO_RAND)));
    }

    return {h_a, h_b};
}

// Based heavily on https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
// Presumes that there is no header in the csv file
__host__ std::tuple<float *, float *, int>readCsv(std::string filename)
{
    vector<int> tempResult;
    // Create an input filestream
    ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw runtime_error("Could not open file");

    // Helper vars
    string line, colname;
    float val;

    // Read 1st line of data
    getline(myFile, line);
    // Create a stringstream of the current line
    stringstream ss0(line);
    
    // Extract each integer
    while(ss0 >> val){
        tempResult.push_back(val);
        // If the next token is a comma, ignore it and move on
        if(ss0.peek() == ',') ss0.ignore();
    }

    int numElements = tempResult.size();
    size_t size = numElements * sizeof(float);
    
    float *h_a;
    h_a = (float *)malloc(size);

    // Copy all elements of vector to h_a
    copy(tempResult.begin(), tempResult.end(), h_a);
    tempResult.clear();

    // Read 2nd line of data
    getline(myFile, line);
    // Create a stringstream of the current line
    stringstream ss1(line);
    
    // Extract each integer
    while(ss1 >> val){
        tempResult.push_back(val);
        // If the next token is a comma, ignore it and move on
        if(ss1.peek() == ',') ss1.ignore();
    }

    // Allocate the host pinned memory input pointer B
    float *h_b;
    cudaMallocHost((float **)&h_b, size);

    // Copy all elements of vector to input_a
    copy(tempResult.begin(), tempResult.end(), h_b);

    // Close file
    myFile.close();
    return {h_a, h_b, numElements};
}

__host__ std::tuple<float *, float *> allocateDeviceMemory(int numElements)
{
    // Allocate the device input vector A
    float *d_a = NULL;
    size_t size = numElements * sizeof(float);

    cudaError_t err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_b;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return {d_a, d_b};
}

__host__ void copyFromHostToDevice(float *h_a, float *h_b, float *d_a, float *d_b, int numElements)
{
    size_t size = numElements * sizeof(int);

    cudaError_t err;

    // FILL IN HOST AND DEVICE MEMORY COPY CODE - SPECIFICALLY copy h_a to d_a
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    // FILL IN HOST AND DEVICE MEMORY COPY CODE - SPECIFICALLY copy h_a to d_a

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // FILL IN HOST AND DEVICE MEMORY COPY CODE - SPECIFICALLY copy h_b to d_b
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // FILL IN HOST AND DEVICE MEMORY COPY CODE - SPECIFICALLY copy h_b to d_b

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(float *d_a, float *d_b, float *h_c, int numElements, int threadsPerBlock)
{
    // Launch the search CUDA Kernel
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    div<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(float *d_a, float *d_b)
{

    cudaError_t err = cudaFree(d_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
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

__host__ void outputToFile(std::string currentPartId, float *h_a, float *h_b, float *h_c, int numElements)
{
	string outputFileName = "output-" + currentPartId + ".txt";
	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread
	ofstream outputFile;
	outputFile.open (outputFileName, ofstream::app);

    outputFile << "PartID: " << currentPartId << "\n";
    outputFile << "Input A: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_a[i] << " ";
    outputFile << "\n";
    outputFile << "Input B: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_b[i] << " ";
    outputFile << "\n";
    outputFile << "Result: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_c[i] << " ";
    outputFile << "\n";

	outputFile.close();
}

__host__ std::tuple<int, std::string, int, std::string, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int numElements = 10;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    std::string mathematicalOperation = "add";
    std::string inputFilename = "NULL";

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-t") == 0) 
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if(option.compare("-n") == 0) 
        {
            numElements = atoi(value.c_str());
        }
        else if(option.compare("-f") == 0) 
        {
            inputFilename = value;
        }
        else if(option.compare("-p") == 0) 
        {
            currentPartId = value;
        }
        else if(option.compare("-o") == 0) 
        {
            mathematicalOperation = value;
        }
    }

    return {numElements, currentPartId, threadsPerBlock, inputFilename, mathematicalOperation};
}

__host__ std::tuple<float *, float *, int> setUpInput(std::string inputFilename, int numElements)
{
    srand(time(0));
    float *h_a;
    float *h_b;

    if(inputFilename.compare("NULL") != 0)
    {
        tuple<float *, float*, int>csvData = readCsv(inputFilename);
        h_a = get<0>(csvData);
        h_b = get<1>(csvData);
        numElements = get<2>(csvData);
    }
    else 
    {
        tuple<float *, float*> randomData = allocateRandomHostMemory(numElements);
        h_a = get<0>(randomData);
        h_b = get<1>(randomData);
    }

    return {h_a, h_b, numElements};
}

/*
 * Host main routine
 * -n numElements - the number of elements of random data to create
 * -f inputFile - the file for non-random input data
 * -o mathematicalOperation - this will decide which math operation kernel will be executed
 * -p currentPartId - the Coursera Part ID
 * -t threadsPerBlock - the number of threads to schedule for concurrent processing
 */
int main(int argc, char *argv[])
{
    auto[numElements, currentPartId, threadsPerBlock, inputFilename, mathematicalOperation] = parseCommandLineArguments(argc, argv);
    tuple<float *, float*, int> searchInputTuple = setUpInput(inputFilename, numElements);
    float *h_a;
    float *h_b;

    h_a = get<0>(searchInputTuple);
    h_b = get<1>(searchInputTuple);
    numElements = get<2>(searchInputTuple);

    float *h_c;
    cudaMallocManaged((float **)&h_c, numElements*sizeof(float));

    auto[d_a, d_b] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_a, h_b, d_a, d_b, numElements);

    executeKernel(d_a, d_b, h_c, numElements, threadsPerBlock);

    outputToFile(currentPartId, h_a, h_b, h_c, numElements);

    deallocateMemory(d_a, d_b);

    cleanUpDevice();
    return 0;
}