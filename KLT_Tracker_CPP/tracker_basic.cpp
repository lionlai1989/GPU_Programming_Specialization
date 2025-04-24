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

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>
#include <vector>

void computeTemporalDerivative(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &It) {
    cv::Mat f1, f2;
    img1.convertTo(f1, CV_32F);
    img2.convertTo(f2, CV_32F);
    // apply gaussian blur to the images. will it help?
    cv::GaussianBlur(f1, f1, cv::Size(5, 5), 0, 0);
    cv::GaussianBlur(f2, f2, cv::Size(5, 5), 0, 0);
    cv::subtract(f2, f1, It, cv::noArray(), CV_32F);
}

bool isValidPoint(const cv::Point2f &p, const cv::Mat &img, int halfWindow) {
    return (p.x >= halfWindow && p.y >= halfWindow && p.x < img.cols - halfWindow && p.y < img.rows - halfWindow);
}

void computeNormalEquations(const cv::Mat &Ix, const cv::Mat &Iy, const cv::Mat &It, float &A11, float &A12, float &A22,
                            float &b1, float &b2) {
    A11 = A12 = A22 = b1 = b2 = 0.f;
    for (int r = 0; r < Ix.rows; ++r) {
        for (int c = 0; c < Ix.cols; ++c) {
            float ix = Ix.at<float>(r, c);
            float iy = Iy.at<float>(r, c);
            float it = It.at<float>(r, c);
            A11 += ix * ix;
            A12 += ix * iy;
            A22 += iy * iy;
            b1 += ix * it;
            b2 += iy * it;
        }
    }
}

void calcOpticalFlowPyrLK(const cv::Mat &prevImg, const cv::Mat &nextImg, const cv::Point2f &prevPt,
                          cv::Point2f &nextPt, std::vector<uchar> &status, std::vector<float> &err,
                          cv::Size winSize = cv::Size(15, 15), int maxLevel = 3,
                          cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                                                       20, 0.03)) {
    CV_Assert(prevImg.channels() == 1 && nextImg.channels() == 1);

    std::vector<cv::Mat> prevPyr, nextPyr;
    cv::buildOpticalFlowPyramid(prevImg, prevPyr, winSize, maxLevel, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT);
    cv::buildOpticalFlowPyramid(nextImg, nextPyr, winSize, maxLevel, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT);

    // Verify pyramid size is correct
    CV_Assert(prevPyr.size() == 2 * (maxLevel + 1));
    CV_Assert(nextPyr.size() == 2 * (maxLevel + 1));

    status.assign(1, 1);
    err.assign(1, 0.f);

    cv::Point2f g(0.f, 0.f);
    int halfW = winSize.width / 2;

    // Iterate from coarsest..finest
    for (int level = maxLevel; level >= 0; --level) {
        // Map original pt into this level
        float scale = 1.f / float(1 << level);
        cv::Point2f u0 = prevPt * scale;
        cv::Point2f warped = u0 + g;

        // **Correct indexing** into interleaved pyramid :contentReference[oaicite:6]{index=6}
        const cv::Mat &I0 = prevPyr[2 * level + 0]; // image
        const cv::Mat &I1 = nextPyr[2 * level + 0];
        const cv::Mat &D0 = prevPyr[2 * level + 1]; // derivative map (CV_16SC2)

        std::vector<cv::Mat> ch(2);
        cv::split(D0, ch);
        cv::Mat Ix, Iy;
        ch[0].convertTo(Ix, CV_32F);
        ch[1].convertTo(Iy, CV_32F);

        // Validate start position
        if (!status[0] || !isValidPoint(warped, I0, halfW)) {
            status[0] = 0;
            throw std::runtime_error("Invalid start position");
        }

        // Subpixel refinement loop
        float ix0 = warped.x, iy0 = warped.y;
        for (int it = 0; it < criteria.maxCount; ++it) {
            // Extract patches
            cv::Mat patchI0, patchI1, patchIx, patchIy, patchIt;
            if (!isValidPoint({ix0, iy0}, I0, halfW)) {
                status[0] = 0;
                throw std::runtime_error("Invalid point");
                break;
            }
            cv::getRectSubPix(I0, winSize, {ix0, iy0}, patchI0);
            cv::getRectSubPix(I1, winSize, {ix0, iy0}, patchI1);
            cv::getRectSubPix(Ix, winSize, {ix0, iy0}, patchIx);
            cv::getRectSubPix(Iy, winSize, {ix0, iy0}, patchIy);

            // Temporal derivative
            computeTemporalDerivative(patchI0, patchI1, patchIt);

            // Build & solve normal equations
            float A11, A12, A22, b1, b2;
            computeNormalEquations(patchIx, patchIy, patchIt, A11, A12, A22, b1, b2);
            float det = A11 * A22 - A12 * A12;
            if (det < 1e-4f) {
                status[0] = 0;
                throw std::runtime_error("Det is too small");
                break;
            }
            float dx = (A22 * b1 - A12 * b2) / det;
            float dy = (A11 * b2 - A12 * b1) / det;
            ix0 += dx;
            iy0 += dy;
            if (std::sqrt(dx * dx + dy * dy) < criteria.epsilon)
                break;
        }

        cv::Point2f d(ix0 + warped.x, iy0 + warped.y);
        g = 2.f * (g + d);
    }

    // Final full-resolution location
    nextPt.x = prevPt.x + g.x;
    nextPt.y = prevPt.y - g.y;

    std::cout << "Tracked from " << prevPt << " to " << nextPt << std::endl;
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
    std::vector<cv::Point2f> nextPts = prevPts;
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
            // Track points using our KLT tracker
            calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPts[0], nextPts[0], status, err, winSize, maxLevel,
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
        if (frameCount > 50) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}