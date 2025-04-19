#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// Function to build image pyramid
void buildPyramid(const cv::Mat &img, std::vector<cv::Mat> &pyramid, int maxLevel) {
    pyramid.clear();
    pyramid.push_back(img);
    for (int l = 1; l <= maxLevel; ++l) {
        cv::Mat down;
        cv::pyrDown(pyramid[l - 1], down);
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

// Function to compute spatial derivatives
void computeDerivatives(const cv::Mat &img, cv::Mat &Ix, cv::Mat &Iy) {
    cv::Sobel(img, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(img, Iy, CV_32F, 0, 1, 3);
}

// Function to compute temporal derivative
void computeTemporalDerivative(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &It) { It = img2 - img1; }

// Function to check if a point is valid (within image bounds)
bool isValidPoint(const cv::Point2f &p, const cv::Mat &img, int halfWindow) {
    return p.x >= halfWindow && p.y >= halfWindow && p.x <= img.cols - halfWindow && p.y <= img.rows - halfWindow;
}

// Function to track points using Lucas-Kanade algorithm with pyramids
void calcOpticalFlowPyrLK(const std::vector<cv::Mat> &prevPyr, const std::vector<cv::Mat> &nextPyr,
                          const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                          std::vector<uchar> &status, std::vector<float> &err, int maxLevel = 3, int maxIter = 30,
                          float epsilon = 0.01, int windowSize = 15) {
    int halfWindow = windowSize / 2;
    nextPts = prevPts;
    status.assign(prevPts.size(), 1);
    err.assign(prevPts.size(), 0);

    for (int l = maxLevel; l >= 0; --l) {
        const cv::Mat &I0 = prevPyr[l];
        const cv::Mat &I1 = nextPyr[l];
        float scale = 1.f / (1 << l);

        // Scale points for current level
        std::vector<cv::Point2f> pts0 = scalePoints(prevPts, scale);
        std::vector<cv::Point2f> pts1 = scalePoints(nextPts, scale);

        // For each feature
        for (size_t i = 0; i < pts0.size(); ++i) {
            if (!status[i])
                continue;

            cv::Point2f p = pts1[i];
            if (!isValidPoint(p, I0, halfWindow)) {
                status[i] = 0;
                continue;
            }

            // Iterative LK
            for (int iter = 0; iter < maxIter; ++iter) {
                // Extract window around p
                cv::Rect roi(p.x - halfWindow, p.y - halfWindow, windowSize, windowSize);
                cv::Mat templateImg = I0(roi);
                cv::Mat searchImg = I1(roi);

                // Compute derivatives
                cv::Mat Ix, Iy, It;
                computeDerivatives(templateImg, Ix, Iy);
                computeTemporalDerivative(templateImg, searchImg, It);

                // Build Hessian and vector b
                float A11 = 0, A12 = 0, A22 = 0;
                float b1 = 0, b2 = 0;

                for (int y = 0; y < windowSize; ++y) {
                    for (int x = 0; x < windowSize; ++x) {
                        float ix = Ix.at<float>(y, x);
                        float iy = Iy.at<float>(y, x);
                        float it = It.at<float>(y, x);

                        A11 += ix * ix;
                        A12 += ix * iy;
                        A22 += iy * iy;

                        b1 += ix * it;
                        b2 += iy * it;
                    }
                }

                // Check eigenvalues of Hessian
                float det = A11 * A22 - A12 * A12;
                float trace = A11 + A22;
                float minEig = (trace - std::sqrt(trace * trace - 4 * det)) / 2;
                if (minEig < 1e-6) {
                    status[i] = 0;
                    break;
                }

                // Solve system
                float dx = (A22 * b1 - A12 * b2) / det;
                float dy = (A11 * b2 - A12 * b1) / det;

                // Update position
                p.x += dx;
                p.y += dy;

                // Check convergence
                if (std::abs(dx) + std::abs(dy) < epsilon)
                    break;
            }

            pts1[i] = p;
        }

        // Scale points back to original scale
        nextPts = scalePoints(pts1, 1 / scale);
    }
}

// rm -rf build/ && cmake -S . -B build/ && cmake --build build/ -j 4 &&./build/KLT_Tracker_CPP pres_debate.avi
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
    cv::Mat frame, prevFrame, nextFrame;
    cap.read(prevFrame);
    if (prevFrame.empty()) {
        std::cout << "Error: Could not read first frame" << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);
    prevFrame.convertTo(prevFrame, CV_32F);

    // Select feature points within image bounds
    const int windowSize = 15;
    const int halfWindow = windowSize / 2;
    const int margin = halfWindow + 5; // Add some margin for safety

    std::vector<cv::Point2f> prevPts;
    int step = 50; // Distance between points

    for (int y = margin; y < prevFrame.rows - margin; y += step) {
        for (int x = margin; x < prevFrame.cols - margin; x += step) {
            prevPts.push_back(cv::Point2f(x, y));
        }
    }

    if (prevPts.empty()) {
        std::cout << "Error: No valid feature points could be selected" << std::endl;
        return -1;
    }

    // Create window for visualization
    cv::namedWindow("KLT Tracker", cv::WINDOW_AUTOSIZE);

    // Process video
    int frameCount = 0;
    while (cap.read(nextFrame)) {
        // Convert to grayscale
        cv::cvtColor(nextFrame, nextFrame, cv::COLOR_BGR2GRAY);
        nextFrame.convertTo(nextFrame, CV_32F);

        // Build pyramids
        std::vector<cv::Mat> prevPyr, nextPyr;
        buildPyramid(prevFrame, prevPyr, 3);
        buildPyramid(nextFrame, nextPyr, 3);

        // Track points
        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;
        calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, nextPts, status, err);

        // Filter out invalid points
        std::vector<cv::Point2f> validPrevPts, validNextPts;
        for (size_t i = 0; i < prevPts.size(); ++i) {
            if (status[i] && isValidPoint(nextPts[i], nextFrame, halfWindow)) {
                validPrevPts.push_back(prevPts[i]);
                validNextPts.push_back(nextPts[i]);
            }
        }

        // Draw tracking results
        cv::Mat display;
        cv::cvtColor(nextFrame, display, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < validPrevPts.size(); ++i) {
            cv::circle(display, validNextPts[i], 3, cv::Scalar(0, 255, 0), -1);
            cv::line(display, validPrevPts[i], validNextPts[i], cv::Scalar(0, 255, 0), 1);
        }

        // Show result
        cv::imshow("KLT Tracker", display);
        if (cv::waitKey(30) >= 0)
            break;

        // Update for next iteration
        prevFrame = nextFrame;
        prevPts = validNextPts;
        frameCount++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}