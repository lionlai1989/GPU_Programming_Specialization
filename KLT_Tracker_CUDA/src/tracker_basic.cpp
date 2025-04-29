/**
 * Use OpenCV's KLT tracker to track the points in the video.
 *
 * Build and run:
 * rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4 && ./build/tracker_basic 1920_1080_30fps.mp4
 *
 */

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

void sparse_pyramid_lucas_kanade(cv::Mat &prev_gray, cv::Mat &next_gray, std::vector<cv::Point2f> &prev_pts,
                                 std::vector<cv::Point2f> &next_pts, std::vector<uchar> &status,
                                 std::vector<float> &err) {}

int main(int argc, char **argv) {
    std::string input_mp4 = "1920_1080_30fps.mp4";
    std::string output_mp4 = "basic.mp4";

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
    cv::namedWindow("KLT Tracker", cv::WINDOW_AUTOSIZE);

    cv::Mat next_rgb;
    while (true) {
        if (!cap.read(next_rgb))
            break;
        cv::Mat next_gray;
        cv::cvtColor(next_rgb, next_gray, cv::COLOR_BGR2GRAY);

        sparse_pyramid_lucas_kanade(prev_gray, next_gray, prev_pts, next_pts, status, err);

        for (size_t i = 0; i < prev_pts.size(); ++i) {
            trajectory[i].push_back(next_pts[i]);
        }

        cv::Mat display = next_rgb.clone();

        plot_trajectory(display, trajectory);
        writer.write(display);

        // Show result
        cv::imshow("KLT Tracker", display);
        char key = cv::waitKey(30);
        if (key == 27) // ESC key
            break;

        prev_gray = next_gray;
        prev_pts = next_pts;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}