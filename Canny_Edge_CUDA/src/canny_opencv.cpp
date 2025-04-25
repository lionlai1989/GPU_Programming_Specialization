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

    std::cout << "Processing video: " << width << "x" << height << " @ " << fps << " fps" << std::endl;

    // Process each frame
    cv::Mat frame;
    int frame_count = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    while (cap.read(frame)) {
        // Convert to grayscale first
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply Canny edge detection
        cv::Mat processed_frame;
        cv::Canny(gray, processed_frame, 100, 200);

        writer.write(processed_frame);

        // frame_count++;
        // if (frame_count % 10 == 0) {
        //     std::cout << "Processed frame " << frame_count << " of " << total_frames << std::endl;
        // }
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
