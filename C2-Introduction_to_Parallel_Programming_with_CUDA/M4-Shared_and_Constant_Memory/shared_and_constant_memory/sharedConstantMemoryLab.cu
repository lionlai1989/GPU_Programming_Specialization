#include "sharedConstantMemoryLab.hpp"

void checkCudaError(cudaError_t err, const char *functionName)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error in " << functionName << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel(uchar *d_r, uchar *d_g, uchar *d_b)
{
    // Calculate data size and thread index
    int num_image_pixels = d_rows * d_columns;
    // int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    // int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < num_image_pixels)
    {
        // Initialize variables or arrays to hold RGB values from device
        d_r[threadId] = 255 - d_r[threadId];
        d_g[threadId] = 255 - d_g[threadId];
        d_b[threadId] = 255 - d_b[threadId];
        // sync threads so that you can alter RGB values without causing race condition

        // Perform calculations on per thread basis
    }
}

__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, int rows, int columns, int threadsPerBlock)
{
    std::cout << "Executing kernel\n";
    // Launch the convert CUDA Kernel
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_g, d_b);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Launch kernel");
}

__host__ std::tuple<uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int columns)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    // Allocate the device input vector d_r
    uchar *d_r = NULL;
    cudaError_t err = cudaMalloc(&d_r, size);
    checkCudaError(err, ("cudaMalloc d_r with size " + std::to_string(size)).c_str());

    // Allocate the device input vector d_g
    uchar *d_g = NULL;
    err = cudaMalloc(&d_g, size);
    checkCudaError(err, ("cudaMalloc d_g with size " + std::to_string(size)).c_str());

    // Allocate the device input vector d_b
    uchar *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    checkCudaError(err, ("cudaMalloc d_b with size " + std::to_string(size)).c_str());

    // Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b};
}

__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int columns)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    cudaError_t err;
    err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy from h_r to d_r");

    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy from h_g to d_g");

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy from h_b to d_b");
}

__host__ void copyFromDeviceToHost(uchar *d_r, uchar *d_g, uchar *d_b, uchar *h_r, uchar *h_g, uchar *h_b, int rows, int columns)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = rows * columns * sizeof(uchar);

    cudaError_t err;
    err = cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy from d_r to h_r");

    err = cudaMemcpy(h_g, d_g, size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy from d_g to h_g");

    err = cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy from d_b to h_b");
}

// Free device global memory
__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b)
{
    std::cout << "Deallocating GPU device memory\n";
    checkCudaError(cudaFree(d_r), "cudaFree d_r");
    checkCudaError(cudaFree(d_g), "cudaFree d_g");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    checkCudaError(cudaDeviceReset(), "cudaDeviceReset");
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);

    const int rows = img.rows;
    const int columns = img.cols;
    const int channels = img.channels();

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *h_r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < columns; ++c)
        {
            Vec3b intensity = img.at<Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            h_r[r * columns + c] = red;
            h_g[r * columns + c] = green;
            h_b[r * columns + c] = blue;
        }
    }

    return {rows, columns, h_r, h_g, h_b};
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try
    {
        auto [rows, columns, h_r, h_g, h_b] = readImageFromFile(inputImage);
        uchar *gray = (uchar *)malloc(sizeof(uchar) * rows * columns);
        std::tuple<uchar *, uchar *, uchar *> memoryTuple = allocateDeviceMemory(rows, columns);
        uchar *d_r = get<0>(memoryTuple);
        uchar *d_g = get<1>(memoryTuple);
        uchar *d_b = get<2>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, rows, columns);

        executeKernel(d_r, d_g, d_b, rows, columns, threadsPerBlock);

        copyFromDeviceToHost(d_r, d_g, d_b, h_r, h_g, h_b, rows, columns);
        deallocateMemory(d_r, d_g, d_b);
        cleanUpDevice();

        Mat outputImageMat(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < columns; ++c)
            {
                int idx = r * columns + c;
                outputImageMat.at<Vec3b>(r, c) = Vec3b(h_b[idx], h_g[idx], h_r[idx]);
                // outputImageMat.at<uchar>(r, c) = gray[r * columns + c];
            }
        }

        imwrite(outputImage, outputImageMat, compression_params);
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}