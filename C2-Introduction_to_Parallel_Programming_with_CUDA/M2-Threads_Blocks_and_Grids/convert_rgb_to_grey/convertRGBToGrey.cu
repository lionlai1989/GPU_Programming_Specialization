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
#include "convertRGBToGrey.hpp"

/*
 * CUDA Kernel Device code
 *
 */
__global__ void convert(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
    //To convert from RGB to grayscale, use the average of the values in d_r, d_g, d_b and place in d_gray

}

__host__ float compareGrayImages(uchar *gray, uchar *test_gray, int rows, int columns)
{
    cout << "Comparing actual and test grayscale pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0.0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            uchar image0Pixel = gray[r*rows+c];
            uchar image1Pixel = test_gray[r*rows+c];
            imagePixelDifference += abs(image0Pixel - image1Pixel);
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ std::tuple<uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int columns)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    // Allocate the device input vector d_r
    uchar *d_r = NULL;
    cudaError_t err = cudaMalloc(&d_r, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_g
    uchar *d_g = NULL;
    err = cudaMalloc(&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_b
    uchar *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_gray
    uchar *d_gray = NULL;
    err = cudaMalloc(&d_gray, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b, d_gray};
}


__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int columns)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    cudaError_t err;
    err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector b from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";
    //Launch the convert CUDA Kernel
    int blockZSize = 4; // Could consider making the block/grid and memory layout 3d mapped but for now just breaking up computation
    int gridCols = min(columns/(threadsPerBlock*4),1);
    dim3 grid(rows, gridCols, 1);
    dim3 block(1, threadsPerBlock, blockZSize);

    convert<<<grid, block>>>(d_r, d_g, d_b, d_gray);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(uchar *d_gray, uchar *gray, int rows, int columns)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = rows * columns * sizeof(uchar);

    cudaError_t err = cudaMemcpy(gray, d_gray, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_gray from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
    cout << "Deallocating GPU device memory\n";
    cudaError_t err = cudaFree(d_r);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_gray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_image_num_pixels (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    
    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            Vec3b intensity = img.at<Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            h_r[r*rows+c] = red;
            h_g[r*rows+c] = green;
            h_b[r*rows+c] = blue;
        }
    }

    return {rows, columns, h_r, h_g, h_b};
}

__host__ uchar *cpuConvertToGray(std::string inputFile)
{
    cout << "CPU converting image file to grayscale\n";
    Mat grayImage = imread(inputFile, IMREAD_GRAYSCALE);
    const int rows = grayImage.rows;
    const int columns = grayImage.cols;

    uchar *gray = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            gray[r*rows+c] = min(grayImage.at<uchar>(r, c), 254);
        }
    }

    return gray;
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
        auto[rows, columns, h_r, h_g, h_b] = readImageFromFile(inputImage);
        uchar *gray = (uchar *)malloc(sizeof(uchar) * rows * columns);
        std::tuple<uchar *, uchar *, uchar *, uchar *> memoryTuple = allocateDeviceMemory(rows, columns);
        uchar *d_r = get<0>(memoryTuple);
        uchar *d_g = get<1>(memoryTuple);
        uchar *d_b = get<2>(memoryTuple);
        uchar *d_gray = get<3>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, rows, columns);

        executeKernel(d_r, d_g, d_b, d_gray, rows, columns, threadsPerBlock);

        copyFromDeviceToHost(d_gray, gray, rows, columns);
        deallocateMemory(d_r, d_g, d_b, d_gray);
        cleanUpDevice();

        Mat grayImageMat(rows, columns, CV_8UC1);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        //cout << "Output gray intensities: ";
        for(int r = 0; r < rows; ++r)
        {
            for(int c = 0; c < columns; ++c)
            {
                grayImageMat.at<uchar>(r,c) = gray[r*rows+c];
            }
        }
        //cout << "\n";

        imwrite(outputImage, grayImageMat, compression_params);

        uchar *test_gray = cpuConvertToGray(inputImage);
        
        float scaledMeanDifferencePercentage = compareGrayImages(gray, test_gray, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}