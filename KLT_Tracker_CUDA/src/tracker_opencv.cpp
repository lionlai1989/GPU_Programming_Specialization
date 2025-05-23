/**
 * Use OpenCV's KLT tracker `calcOpticalFlowPyrLK` to track sparse points in videos.
 *
 */

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

void plot_trajectory(cv::Mat &display, const std::vector<std::vector<cv::Point2f>> &trajectory) {
    std::vector<cv::Scalar> line_color = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)};
    std::vector<cv::Scalar> point_color = {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};

    for (size_t i = 0; i < trajectory.size(); ++i) {
        for (size_t j = 1; j < trajectory[i].size(); ++j) {
            cv::line(display, trajectory[i][j - 1], trajectory[i][j], line_color[i], 2);
        }

        cv::circle(display, trajectory[i].back(), 5, point_color[i], -1);
    }
}

int main(int argc, char **argv) {
    std::string input_mp4 = "data/1920x1080_30fps_8s.mp4";
    std::string output_mp4 = "output/tracker_opencv.mp4";

    std::vector<cv::Point2f> prev_pts;
    prev_pts.push_back(cv::Point2f(1676, 654));
    prev_pts.push_back(cv::Point2f(1740, 699));
    prev_pts.push_back(cv::Point2f(1825, 690));

    std::vector<cv::Point2f> next_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Store all tracked points for visualization
    std::vector<std::vector<cv::Point2f>> trajectory(prev_pts.size());

    // Initialize tracked points with starting positions
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        trajectory[i].push_back(prev_pts[i]);
    }

    cv::VideoCapture cap(input_mp4);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open video file" << std::endl;
        return -1;
    }
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::Mat prev_bgr, prev_gray;
    cap.read(prev_bgr);
    if (prev_bgr.empty()) {
        std::cout << "Error: Could not read first frame" << std::endl;
        return -1;
    }
    cv::cvtColor(prev_bgr, prev_gray, cv::COLOR_BGR2GRAY);
    int height = prev_bgr.rows;
    int width = prev_bgr.cols;

    cv::VideoWriter writer(output_mp4, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, cv::Size(width, height));

    long long accum_time = 0;
    cv::Mat next_bgr;
    while (true) {
        if (!cap.read(next_bgr))
            break;

        auto t1 = std::chrono::high_resolution_clock::now(); // start time

        cv::Mat next_gray;
        cv::cvtColor(next_bgr, next_gray, cv::COLOR_BGR2GRAY);
        cv::calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, next_pts, status, err);

        auto t2 = std::chrono::high_resolution_clock::now(); // end time
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        for (size_t i = 0; i < prev_pts.size(); ++i) {
            trajectory[i].push_back(next_pts[i]);
        }

        // Uncomment to visualize the result
        // cv::Mat display = next_bgr.clone();
        // plot_trajectory(display, trajectory);
        // writer.write(display);

        prev_gray = next_gray;
        prev_pts = next_pts;
    }

    std::cout << "Total time of all frames: " << accum_time << " microseconds. "
              << "Each frame average time: " << accum_time / total_frames << " microseconds." << std::endl;

    cap.release();
    writer.release();
    return 0;
}
