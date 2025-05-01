#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    std::string input_file{"data/me.mp4"};
    std::string output_file{"canny_opencv.mp4"};

    if (input_file.empty() || output_file.empty()) {
        exit(EXIT_FAILURE);
    }

    cv::VideoCapture cap(input_file, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << input_file << "\n";
        exit(EXIT_FAILURE);
    }

    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Create video writer with H.264 codec
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 codec
                           fps, cv::Size(width, height),
                           false); // false for grayscale output

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create video writer for: " << output_file << "\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Processing video: " << width << "x" << height << " @ " << fps << " fps" << " with " << total_frames
              << " frames" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Process each frame
    cv::Mat frame;
    cv::Mat gray(height, width, CV_8UC1);
    cv::Mat edge_map(height, width, CV_8UC1);
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edge_map, 100, 200);
        writer.write(edge_map);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Release resources
    cap.release();
    writer.release();

    std::cout << "Done. Output saved as " << output_file << std::endl;
    return 0;
}
