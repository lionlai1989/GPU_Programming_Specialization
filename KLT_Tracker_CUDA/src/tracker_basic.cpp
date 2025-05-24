/**
 * The goal of this code is to implement the KLT tracker in C++ without using cv::calcOpticalFlowPyrLK.
 * I tried to make the code as cuda-friendly as possible.
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

static inline bool in_bound(const float x, const float y, int half_win, const int width, const int height) {
    return x >= half_win && x < width - half_win && y >= half_win && y < height - half_win;
}

std::tuple<float, float, bool> lucas_kanade(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &grad_x,
                                            const cv::Mat &grad_y, cv::Mat &template_patch, cv::Mat &patch,
                                            cv::Mat &patch_grad_x, cv::Mat &patch_grad_y, float u, float v, float x_l,
                                            float y_l, float scaled_win_size, int origin_win_size, int max_iter,
                                            float eps, float min_eig) {

    // Extract the template patch from img1
    cv::getRectSubPix(img1, cv::Size(origin_win_size, origin_win_size), {x_l, y_l}, template_patch);
    template_patch.convertTo(template_patch, CV_32F); // uint8 to float32
    assert(template_patch.type() == CV_32F);
    // DEBUG
    // cv::imwrite("tracker_basic_template_patch.png", template_patch);

    bool success = true;

    for (int iter = 0; iter < max_iter; iter++) {

        float xc = x_l + u;
        float yc = y_l + v;

        // Check if the shifted patch is inside I2
        if (!in_bound(xc, yc, int(scaled_win_size / 2), img2.cols, img2.rows)) {
            // std::cout << "lucas_kanade Out of bound at iteration " << iter << std::endl;
            success = false;
            break;
        }

        // Sample the patch in img2 and the gradients at (xc, yc)
        cv::getRectSubPix(img2, cv::Size(origin_win_size, origin_win_size), {xc, yc}, patch);
        cv::getRectSubPix(grad_x, cv::Size(origin_win_size, origin_win_size), {xc, yc}, patch_grad_x);
        cv::getRectSubPix(grad_y, cv::Size(origin_win_size, origin_win_size), {xc, yc}, patch_grad_y);
        patch.convertTo(patch, CV_32F); // uint8 to float32
        assert(patch.type() == CV_32F);
        // DEBUG
        // cv::imwrite("tracker_basic_patch.png", patch);

        // Compute error image (template - patch)
        cv::Mat error = template_patch - patch;
        assert(error.type() == CV_32F);

        error = error.reshape(1, error.total());                    // ravel()
        cv::Mat gx = patch_grad_x.reshape(1, patch_grad_x.total()); // ravel()
        cv::Mat gy = patch_grad_y.reshape(1, patch_grad_y.total()); // ravel()

        // Build elements of the normal equations matrix
        float Gxx = cv::sum(gx.mul(gx))[0]; // cv::Scalar
        float Gxy = cv::sum(gx.mul(gy))[0];
        float Gyy = cv::sum(gy.mul(gy))[0];

        // Check for degeneracy and compute minimum eigenvalue
        float det = Gxx * Gyy - Gxy * Gxy;
        if (det <= 0) {
            // std::cout << "lucas_kanade det <= 0 at iteration " << iter << std::endl;
            success = false;
            break;
        }
        float trace = Gxx + Gyy;
        float lambda_min = (trace - std::sqrt(trace * trace - 4 * det)) / 2.0;
        // Filter by minimum eigenvalue (normalized by patch size)
        if (lambda_min / (origin_win_size * origin_win_size) < min_eig) {
            // std::cout << "lucas_kanade lambda_min < min_eig at iteration " << iter << std::endl;
            success = false;
            break;
        }

        float b1 = cv::sum(gx.mul(error))[0];
        float b2 = cv::sum(gy.mul(error))[0];

        // Solve for [du, dv] using Cramer's rule
        float inv_det = 1.0f / det;
        float du = (Gyy * b1 - Gxy * b2) * inv_det;
        float dv = (-Gxy * b1 + Gxx * b2) * inv_det;

        u += du;
        v += dv;

        if (std::abs(du) < eps && std::abs(dv) < eps) {
            break;
        }
    }

    return std::make_tuple(u, v, success);
}

std::tuple<float, float, bool> pyramid_lucas_kanade(const std::vector<cv::Mat> &pyr1, const std::vector<cv::Mat> &pyr2,
                                                    const std::vector<cv::Mat> &pyr2_grad_x,
                                                    const std::vector<cv::Mat> &pyr2_grad_y, cv::Mat &template_patch,
                                                    cv::Mat &patch, cv::Mat &patch_grad_x, cv::Mat &patch_grad_y,
                                                    const cv::Point2f &pt, int levels, int win_size, int max_iter,
                                                    float eps, float min_eig) {
    float u_prev = 0.0;
    float v_prev = 0.0;
    bool success = true;

    for (int lvl = levels; lvl >= 0; lvl--) { // 3, 2, 1, 0
        float scale = 1.0f / (1 << lvl);

        float x_l = pt.x * scale;
        float y_l = pt.y * scale;

        float u = u_prev * 2.0;
        float v = v_prev * 2.0;

        // Check if the patch around (x_l, y_l) is inside img1
        if (!in_bound(x_l, y_l, int(scale * win_size / 2), pyr1[lvl].cols, pyr1[lvl].rows)) {
            // std::cout << "Out of bound at level " << lvl << std::endl;
            success = false;
            break;
        }

        auto [new_u, new_v, success] =
            lucas_kanade(pyr1[lvl], pyr2[lvl], pyr2_grad_x[lvl], pyr2_grad_y[lvl], template_patch, patch, patch_grad_x,
                         patch_grad_y, u, v, x_l, y_l, scale * win_size, win_size, max_iter, eps, min_eig);

        if (!success) {
            // std::cout << "Failed to converge at level " << lvl << std::endl;
            break;
        }

        // Save refined flow for this level
        u_prev = new_u;
        v_prev = new_v;
    }

    return std::make_tuple(u_prev, v_prev, success);
}

void build_pyramid(std::vector<cv::Mat> &frames, int levels) {
    // Read https://stackoverflow.com/questions/27088969/is-it-a-bug-of-design-of-opencvs-function-pyrdown

    for (int i = 0; i < levels; i++) {
        cv::pyrDown(frames[i], frames[i + 1]);
    }
}

class SparseOpticalFlow {
  private:
    int levels;
    int height, width;
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> next_pts;
    std::vector<cv::Mat> pyr1;
    std::vector<cv::Mat> pyr2;
    std::vector<cv::Mat> pyr2_grad_x;
    std::vector<cv::Mat> pyr2_grad_y;

    // Pre-allocate memory for template_patch, patch, patch_grad_x, patch_grad_y. It only has header, no data.
    cv::Mat template_patch;
    cv::Mat patch;
    cv::Mat patch_grad_x;
    cv::Mat patch_grad_y;

    int win_size;
    int max_iter;
    float eps;
    float min_eig;

  public:
    SparseOpticalFlow(std::vector<cv::Point2f> &pts, cv::Mat &init_gray, int levels = 3, int win_size = 31,
                      int max_iter = 30, float eps = 0.01, float min_eig = 1e-4);
    ~SparseOpticalFlow();

    std::vector<cv::Point2f> track(cv::Mat &next_bgr);
};

SparseOpticalFlow::SparseOpticalFlow(std::vector<cv::Point2f> &pts, cv::Mat &init_gray, int levels, int win_size,
                                     int max_iter, float eps, float min_eig)
    : levels(levels), win_size(win_size), max_iter(max_iter), eps(eps), min_eig(min_eig) {
    // Get frame dimensions
    height = init_gray.rows;
    width = init_gray.cols;

    // Pre-allocate pyramid memory
    pyr1.reserve(levels + 1);
    pyr1.push_back(init_gray);          // 0
    for (int i = 1; i <= levels; i++) { // 1, 2, 3
        pyr1.push_back(cv::Mat());      // Initialize as empty
    }
    // Build initial pyramid
    build_pyramid(pyr1, levels);

    // Pre-allocate memory for second pyramid and gradients
    pyr2 = std::vector<cv::Mat>(levels + 1, cv::Mat());
    pyr2_grad_x = std::vector<cv::Mat>(levels + 1, cv::Mat());
    pyr2_grad_y = std::vector<cv::Mat>(levels + 1, cv::Mat());

    // Initialize prev_pts and next_pts
    prev_pts = pts;
    next_pts = std::vector<cv::Point2f>(pts.size(), cv::Point2f(0, 0)); // Initialize with zeros

    // Pre-allocate memory for template_patch, patch, patch_grad_x, patch_grad_y with fixed window size
    template_patch = cv::Mat(win_size, win_size, CV_32F);
    patch = cv::Mat(win_size, win_size, CV_32F);
    patch_grad_x = cv::Mat(win_size, win_size, CV_32F);
    patch_grad_y = cv::Mat(win_size, win_size, CV_32F);
}

std::vector<cv::Point2f> SparseOpticalFlow::track(cv::Mat &next_bgr) {
    cv::Mat next_gray;
    cv::cvtColor(next_bgr, next_gray, cv::COLOR_BGR2GRAY);

    // Copy next frame to first level of pyramid
    next_gray.copyTo(pyr2[0]);
    build_pyramid(pyr2, this->levels);

    // Pre-compute gradients for all pyramid levels
    for (int lvl = 0; lvl <= levels; lvl++) {
        cv::Sobel(pyr2[lvl], pyr2_grad_x[lvl], CV_32F, 1, 0, 3);
        cv::Sobel(pyr2[lvl], pyr2_grad_y[lvl], CV_32F, 0, 1, 3);
    }
    // DEBUG
    // cv::imwrite("tracker_basic_pyr2_grad_x0.png", pyr2_grad_x[0]);
    // cv::imwrite("tracker_basic_pyr2_grad_y0.png", pyr2_grad_y[0]);
    // cv::imwrite("tracker_basic_pyr2_grad_x3.png", pyr2_grad_x[3]);
    // cv::imwrite("tracker_basic_pyr2_grad_y3.png", pyr2_grad_y[3]);

    // Track each point
    for (size_t i = 0; i < prev_pts.size(); i++) {
        float x = prev_pts[i].x;
        float y = prev_pts[i].y;

        auto [u, v, success] =
            pyramid_lucas_kanade(pyr1, pyr2, pyr2_grad_x, pyr2_grad_y, template_patch, patch, patch_grad_x,
                                 patch_grad_y, prev_pts[i], levels, win_size, max_iter, eps, min_eig);

        if (success) {
            next_pts[i].x = x + u;
            next_pts[i].y = y + v;
        } else {
            next_pts[i].x = x;
            next_pts[i].y = y;
        }
    }

    /**
     * Explain why `pyr2 = pyr1` is not working.
     * Assigning one std::vector to another copies the elements of the right‐hand vector into the
     * left. That is, it calls cv::Mat's copy assignment for each element.
     * OpenCV's cv::Mat is a reference-counted container. Its assignment operator makes the new Mat share the same pixel
     * data with the source, merely bumping an internal reference count, rather than allocating fresh storage.
     */
    // for (int i = 0; i <= levels; i++) {
    //     pyr2[i].copyTo(pyr1[i]);
    // }
    std::swap(pyr1, pyr2);

    /**
     * Why is it OK to do this?
     * Point2f's assignment operator does member-wise copy.
     */
    prev_pts = next_pts;

    return next_pts;
}

SparseOpticalFlow::~SparseOpticalFlow() {
    pyr1.clear();
    pyr2.clear();
    pyr2_grad_x.clear();
    pyr2_grad_y.clear();

    template_patch.release();
    patch.release();
    patch_grad_x.release();
    patch_grad_y.release();
}

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
    std::string output_mp4 = "output/tracker_basic.mp4";

    std::vector<cv::Point2f> prev_pts;
    prev_pts.push_back(cv::Point2f(1676, 654));
    prev_pts.push_back(cv::Point2f(1740, 699));
    prev_pts.push_back(cv::Point2f(1825, 690));

    std::vector<cv::Point2f> next_pts;

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
    std::cout << "Total frames: " << total_frames << std::endl;

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

    // Init SparseOpticalFlow
    SparseOpticalFlow sof(prev_pts, prev_gray);

    long long accum_time = 0;
    cv::Mat next_bgr;
    while (true) {
        if (!cap.read(next_bgr))
            break;

        auto t1 = std::chrono::high_resolution_clock::now(); // start time

        std::vector<cv::Point2f> next_pts = sof.track(next_bgr);

        auto t2 = std::chrono::high_resolution_clock::now(); // end time
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        for (size_t i = 0; i < prev_pts.size(); ++i) {
            trajectory[i].push_back(next_pts[i]);
        }

        // Uncomment to visualize the result
        // cv::Mat display = next_bgr.clone();
        // plot_trajectory(display, trajectory);
        // writer.write(display);
    }

    std::cout << "Total time of all frames: " << accum_time << " microseconds. "
              << "Each frame average time: " << accum_time / total_frames << " microseconds." << std::endl;

    cap.release();
    writer.release();
    return 0;
}
