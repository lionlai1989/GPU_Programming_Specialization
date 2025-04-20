/**
 * Implement optical flow using Lucas-Kanade algorithm with pyramids. I manually select the tracking points in advance.
 * x=1676, y=654
 * x=1740, y=699
 * x=1825, y=690
 *
 * Build and run:
 * rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4 && ./build/klt_tracker_basic 1920_1080_30fps.mp4
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

// Function to build image pyramid with proper border handling
void buildPyramid(const cv::Mat &img, std::vector<cv::Mat> &pyramid, int maxLevel) {
    pyramid.clear();
    pyramid.push_back(img);

    // Add border for proper derivative computation
    cv::Mat temp;
    cv::copyMakeBorder(img, temp, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    for (int l = 1; l <= maxLevel; ++l) {
        cv::Mat down;
        cv::pyrDown(pyramid[l - 1], down);
        cv::copyMakeBorder(down, temp, 1, 1, 1, 1, cv::BORDER_REPLICATE);
        pyramid.push_back(down);
    }
}

// Function to scale points according to pyramid level
std::vector<cv::Point2f> scalePoints(const std::vector<cv::Point2f> &points, float scale) {
    std::vector<cv::Point2f> scaled;
    scaled.reserve(points.size());
    for (const auto &p : points) {
        scaled.push_back(p * scale);
    }
    return scaled;
}

// Function to compute Scharr derivatives
void computeScharrDeriv(const cv::Mat &img, cv::Mat &Ix, cv::Mat &Iy) {
    cv::Mat f;
    img.convertTo(f, CV_32F, 1.0 / 255.0);

    // Scharr kernels for better derivative estimation
    cv::Mat scharr_x = (cv::Mat_<float>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
    cv::Mat scharr_y = (cv::Mat_<float>(3, 3) << -3, -10, -3, 0, 0, 0, 3, 10, 3);

    cv::filter2D(f, Ix, CV_32F, scharr_x);
    cv::filter2D(f, Iy, CV_32F, scharr_y);
}

// Function to compute temporal derivative
void computeTemporalDerivative(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &It) {
    cv::Mat f1, f2;
    img1.convertTo(f1, CV_32F, 1.0 / 255.0);
    img2.convertTo(f2, CV_32F, 1.0 / 255.0);
    cv::subtract(f2, f1, It, cv::noArray(), CV_32F);
}

// Function to check if a point is valid (within image bounds)
bool isValidPoint(const cv::Point2f &p, const cv::Mat &img, int halfWindow) {
    float m = static_cast<float>(halfWindow) + 0.5f;
    return (p.x >= m && p.y >= m && p.x < img.cols - m && p.y < img.rows - m);
}

// Function to track points using Lucas-Kanade algorithm with pyramids
void calcOpticalFlowPyrLK(const std::vector<cv::Mat> &prevPyr, const std::vector<cv::Mat> &nextPyr,
                          const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                          std::vector<uchar> &status, std::vector<float> &err, int maxLevel = 3, int maxIter = 30,
                          float epsilon = 1e-2, int windowSize = 31) {
    const int hw = windowSize / 2;
    nextPts = prevPts;
    status.assign(prevPts.size(), 1);
    err.assign(prevPts.size(), 0.f);

    // Loop from coarsest to finest
    for (int level = maxLevel; level >= 0; --level) {
        float scale = 1.f / float(1 << level);

        // Scale pts for this level
        std::vector<cv::Point2f> p0 = scalePoints(prevPts, scale);
        std::vector<cv::Point2f> p1 = scalePoints(nextPts, scale);

        const cv::Mat &I0 = prevPyr[level];
        const cv::Mat &I1 = nextPyr[level];

        // Compute derivatives for this level
        cv::Mat Ix, Iy, It;
        computeScharrDeriv(I0, Ix, Iy);
        computeTemporalDerivative(I0, I1, It);

        for (size_t i = 0; i < p0.size(); ++i) {
            if (!status[i])
                continue;

            cv::Point2f pt = p1[i];
            if (!isValidPoint(pt, I0, hw)) {
                status[i] = 0;
                continue;
            }

            // Initialize subpixel center
            float ix0 = pt.x, iy0 = pt.y;

            for (int it = 0; it < maxIter; ++it) {
                int cx = int(std::round(ix0));
                int cy = int(std::round(iy0));
                cv::Rect roi(cx - hw, cy - hw, windowSize, windowSize);

                if ((roi & cv::Rect(0, 0, I0.cols, I0.rows)) != roi) {
                    status[i] = 0;
                    break;
                }

                // Build normal equations
                float A11 = 0, A12 = 0, A22 = 0, b1 = 0, b2 = 0;
                for (int r = 0; r < windowSize; ++r) {
                    for (int c = 0; c < windowSize; ++c) {
                        float ix = Ix.at<float>(cy - hw + r, cx - hw + c);
                        float iy = Iy.at<float>(cy - hw + r, cx - hw + c);
                        float it = It.at<float>(cy - hw + r, cx - hw + c);

                        A11 += ix * ix;
                        A12 += ix * iy;
                        A22 += iy * iy;
                        b1 += ix * it;
                        b2 += iy * it;
                    }
                }

                float det = A11 * A22 - A12 * A12;
                if (det < FLT_EPSILON) {
                    status[i] = 0;
                    break;
                }

                // Solve for delta
                float dx = (A22 * b1 - A12 * b2) / det;
                float dy = (-A12 * b1 + A11 * b2) / det;

                ix0 += dx;
                iy0 += dy;

                if (std::sqrt(dx * dx + dy * dy) < epsilon)
                    break;
            }

            // Save tracking result
            nextPts[i] = cv::Point2f(ix0, iy0) * float(1 << level);
        }
    }
}

// rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4 && ./build/KLT_Tracker_CPP 1920_1080_30fps.mp4
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

    // Convert to grayscale (keep as uint8)
    cv::cvtColor(prevFrame, prevFrameGray, cv::COLOR_BGR2GRAY);

    // KLT parameters
    const int maxLevel = 3;    // pyramid levels
    const int windowSize = 31; // Larger window for more stable tracking
    const int halfWindow = windowSize / 2;
    const int maxIter = 30;
    const float epsilon = 1e-2;

    // Forward-backward tracking parameters
    const float fbThreshold = 2.0f; // Increased threshold to 2 pixels

    // manually select the points and assign them to prevPts
    std::vector<cv::Point2f> prevPts;
    prevPts.push_back(cv::Point2f(1676, 654)); // Point 0
    prevPts.push_back(cv::Point2f(1740, 699)); // Point 1
    prevPts.push_back(cv::Point2f(1825, 690)); // Point 2

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
        cv::Scalar(0, 255, 0),  // Green
        cv::Scalar(255, 0, 0),  // Blue
        cv::Scalar(0, 255, 255) // Yellow
    };

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        if (!prevPts.empty()) {
            // Build pyramids
            std::vector<cv::Mat> prevPyr, nextPyr;
            buildPyramid(prevFrameGray, prevPyr, maxLevel);
            buildPyramid(frameGray, nextPyr, maxLevel);

            // forward tracking
            calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, nextPts, status, err, maxLevel, maxIter, epsilon,
                                 windowSize);

            // Store forward tracking results
            std::vector<cv::Point2f> forwardPts = nextPts;
            std::vector<uchar> forwardStatus = status;

            // backward tracking
            std::vector<cv::Point2f> backwardPts = prevPts;
            std::vector<uchar> backwardStatus = status;
            calcOpticalFlowPyrLK(nextPyr, prevPyr, forwardPts, backwardPts, backwardStatus, err, maxLevel, maxIter,
                                 epsilon, windowSize);

            // forward-backward consistency check
            const float fbThreshold = 2.0f;
            std::cout << "\nForward-Backward Check:" << std::endl;
            for (size_t i = 0; i < prevPts.size(); ++i) {
                if (forwardStatus[i] && backwardStatus[i]) {
                    float fbError = cv::norm(backwardPts[i] - prevPts[i]);
                    std::cout << "Point " << i << " FB error: " << fbError << std::endl;
                    if (fbError > fbThreshold) {
                        status[i] = 0;
                        std::cout << "Point " << i << " failed FB check" << std::endl;
                    } else {
                        nextPts[i] = forwardPts[i];
                        std::cout << "Point " << i << " passed FB check" << std::endl;
                    }
                } else {
                    status[i] = 0;
                    if (!forwardStatus[i])
                        std::cout << "Point " << i << " failed forward tracking" << std::endl;
                    if (!backwardStatus[i])
                        std::cout << "Point " << i << " failed backward tracking" << std::endl;
                }
            }

            // Debug prints
            std::cout << "\nFrame " << frameCount << ":" << std::endl;
            std::cout << "Status: ";
            for (size_t i = 0; i < status.size(); ++i) {
                std::cout << (int)status[i] << " ";
            }
            std::cout << std::endl;

            // Update tracking history and status
            for (size_t i = 0; i < prevPts.size(); ++i) {
                if (status[i]) {
                    trackedPoints[i].push_back(nextPts[i]);
                    std::cout << "Point " << i << " tracked. History size: " << trackedPoints[i].size() << std::endl;
                } else {
                    pointActive[i] = false;
                    std::cout << "Point " << i << " lost." << std::endl;
                }
            }

            // Draw tracking results
            cv::Mat display = frame.clone();

            // Draw paths and current points
            for (size_t i = 0; i < trackedPoints.size(); ++i) {
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

        if (frameCount > 50)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}