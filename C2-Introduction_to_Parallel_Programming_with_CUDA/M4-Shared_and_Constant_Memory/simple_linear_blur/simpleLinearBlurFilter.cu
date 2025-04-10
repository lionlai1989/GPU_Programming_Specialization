#include "simpleLinearBlurFilter.hpp"

void checkCudaError(cudaError_t err, const char *functionName)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error in " << functionName << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void applySimpleLinearBlurFilter(uchar *r, uchar *g, uchar *b)
{
    // Total number of pixels from constant memory.
    int num_image_pixels = d_rows * d_columns;

    // Compute global 1D thread index.
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < num_image_pixels)
    {
        // Compute row and column based on d_columns.
        int row = threadId / d_columns;
        int col = threadId % d_columns;

        // Process only pixels that have a left and right neighbor.
        if (col > 0 && col < d_columns - 1)
        {
            int idx_left = threadId - 1;
            int idx_right = threadId + 1;

            // For each channel, compute the average.
            uchar new_r = (r[idx_left] + r[threadId] + r[idx_right]) / 3;
            uchar new_g = (g[idx_left] + g[threadId] + g[idx_right]) / 3;
            uchar new_b = (b[idx_left] + b[threadId] + b[idx_right]) / 3;

            r[threadId] = new_r;
            g[threadId] = new_g;
            b[threadId] = new_b;
        }
        // For edge pixels, you may opt to leave them unchanged.
    }
}

__host__ float compareColorImages(uchar *r0, uchar *g0, uchar *b0, uchar *r1, uchar *g1, uchar *b1, int rows, int columns)
{
    cout << "Comparing actual and test pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0.0;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < columns; ++c)
        {
            uchar image0R = r0[r * rows + c];
            uchar image0G = g0[r * rows + c];
            uchar image0B = b0[r * rows + c];
            uchar image1R = r1[r * rows + c];
            uchar image1G = g1[r * rows + c];
            uchar image1B = b1[r * rows + c];
            imagePixelDifference += ((abs(image0R - image1R) + abs(image0G - image1G) + abs(image0B - image1B)) / 3);
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ void allocateDeviceMemory(int rows, int columns)
{

    // Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__host__ void executeKernel(uchar *r, uchar *g, uchar *b, int rows, int columns, int threadsPerBlock)
{
    std::cout << "Executing kernel" << std::endl;
    int totalPixels = rows * columns;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel.
    applySimpleLinearBlurFilter<<<blocksPerGrid, threadsPerBlock>>>(r, g, b);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Launch kernel");
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    std::cout << "Cleaning CUDA device" << std::endl;
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
    size_t size = sizeof(uchar) * rows * columns;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *r, *g, *b;
    cudaMallocManaged(&r, size);
    cudaMallocManaged(&g, size);
    cudaMallocManaged(&b, size);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < columns; ++x)
        {
            Vec3b rgb = img.at<Vec3b>(y, x);
            r[y * rows + x] = rgb.val[0];
            g[y * rows + x] = rgb.val[1];
            b[y * rows + x] = rgb.val[2];
        }
    }

    return {rows, columns, r, g, b};
}

__host__ std::tuple<uchar *, uchar *, uchar *> applyBlurKernel(std::string inputImage)
{
    cout << "CPU applying kernel\n";
    Mat img = imread(inputImage, IMREAD_COLOR);
    const int rows = img.rows;
    const int columns = img.cols;

    uchar *r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 1; x < columns - 1; ++x)
        {
            Vec3b rgb0 = img.at<Vec3b>(y, x - 1);
            Vec3b rgb1 = img.at<Vec3b>(y, x);
            Vec3b rgb2 = img.at<Vec3b>(y, x + 1);
            b[y * rows + x] = (rgb0[0] + rgb1[0] + rgb2[0]) / 3;
            g[y * rows + x] = (rgb0[1] + rgb1[1] + rgb2[1]) / 3;
            r[y * rows + x] = (rgb0[2] + rgb1[2] + rgb2[2]) / 3;
        }
    }

    return {r, g, b};
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
        auto [rows, columns, r, g, b] = readImageFromFile(inputImage);

        executeKernel(r, g, b, rows, columns, threadsPerBlock);

        Mat colorImage(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < columns; ++x)
            {
                colorImage.at<Vec3b>(y, x) = Vec3b(b[y * rows + x], g[y * rows + x], r[y * rows + x]);
            }
        }

        imwrite(outputImage, colorImage, compression_params);

        auto [test_r, test_g, test_b] = applyBlurKernel(inputImage);

        float scaledMeanDifferencePercentage = compareColorImages(r, g, b, test_r, test_g, test_b, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

        cleanUpDevice();
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}