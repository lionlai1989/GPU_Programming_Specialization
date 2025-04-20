/**
 * Use opencv's KLT tracker to track the points in the video. I manually select the tracking points in advance.
 * x=1676, y=654
 * x=1740, y=699
 * x=1825, y=690
 *
 * Build and run:
 * rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4 && ./build/klt_tracker_opencv 1920_1080_30fps.mp4
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    // Open video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open video file" << std::endl;
        return -1;
    }

    // Read first frame
    cv::Mat frame, prevFrame, prevFrameGray;
    cap.read(prevFrame);
    if (prevFrame.empty()) {
        std::cout << "Error: Could not read first frame" << std::endl;
        return -1;
    }

    // Convert to grayscale (keep as CV_8U)
    cv::cvtColor(prevFrame, prevFrameGray, cv::COLOR_BGR2GRAY);

    // KLT parameters
    const int maxLevel = 3; // pyramid levels
    const cv::Size winSize(15, 15);
    const cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

    // manually select the points and assign them to prevPts
    std::vector<cv::Point2f> prevPts;
    prevPts.push_back(cv::Point2f(1676, 654)); // Point 0
    // prevPts.push_back(cv::Point2f(1740, 699)); // Point 1
    // prevPts.push_back(cv::Point2f(1825, 690)); // Point 2

    if (prevPts.empty()) {
        std::cout << "Error: No valid feature points could be selected" << std::endl;
        return -1;
    }

    // Create window for visualization
    cv::namedWindow("KLT Tracker", cv::WINDOW_AUTOSIZE);

    // Process video
    int frameCount = 0;
    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Store all tracked points for visualization
    std::vector<std::vector<cv::Point2f>> trackedPoints(prevPts.size());
    std::vector<bool> pointActive(prevPts.size(), true); // Track if points are still valid

    // Initialize tracked points with starting positions
    for (size_t i = 0; i < prevPts.size(); ++i) {
        trackedPoints[i].push_back(prevPts[i]);
    }

    // Colors for each point
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0), // Green
        // cv::Scalar(255, 0, 0),  // Blue
        // cv::Scalar(0, 255, 255) // Yellow
    };

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        if (!prevPts.empty()) {
            // Track points using OpenCV's KLT tracker with parameters
            cv::calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPts, nextPts, status, err, winSize, maxLevel,
                                     termcrit);

            std::cout << "nextPts.size(): " << nextPts.size() << std::endl;

            // Update tracking history and status
            for (size_t i = 0; i < prevPts.size(); ++i) {
                if (status[i] && pointActive[i]) {
                    trackedPoints[i].push_back(nextPts[i]);
                } else {
                    pointActive[i] = false; // Mark point as lost
                }
            }

            // Draw tracking results
            cv::Mat display = frame.clone();

            // Draw paths and current points
            for (size_t i = 0; i < trackedPoints.size(); ++i) {
                std::cout << "trackedPoints[i].size(): " << trackedPoints[i].size() << std::endl;

                if (trackedPoints[i].size() > 1) {
                    // Draw the path
                    for (size_t j = 1; j < trackedPoints[i].size(); ++j) {
                        cv::line(display, trackedPoints[i][j - 1], trackedPoints[i][j], colors[i], 2);
                    }

                    // Draw current point if active
                    if (pointActive[i]) {
                        cv::circle(display, trackedPoints[i].back(), 5, cv::Scalar(0, 0, 255), -1);

                        // Add point label
                        std::stringstream ss;
                        ss << "P" << i;
                        cv::putText(display, ss.str(), trackedPoints[i].back() + cv::Point2f(10, 10),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2);
                    }
                }
            }

            // Add frame counter
            std::stringstream ss;
            ss << "Frame: " << frameCount;
            cv::putText(display, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),
                        2);

            // Show result
            cv::imshow("KLT Tracker", display);
            char key = cv::waitKey(30);
            if (key == 27) // ESC key
                break;

            // Update for next iteration
            prevFrameGray = frameGray.clone();
            prevPts = nextPts;
        }

        frameCount++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}