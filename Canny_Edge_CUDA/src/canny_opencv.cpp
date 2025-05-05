#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

class BlockingQueue {
  private:
    std::queue<cv::Mat> q;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    bool is_done = false;
    const size_t max_size = 1500; // Maximum number of frames to buffer

  public:
    void enqueue(cv::Mat &&frame) { // Use move semantics
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [this]() { return q.size() < max_size || is_done; });
        if (!is_done) {
            q.push(std::move(frame)); // Move the frame instead of copying
            cv_not_empty.notify_one();
        }
    }

    bool dequeue(cv::Mat &frame) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [this]() { return !q.empty() || is_done; });
        if (q.empty() && is_done) {
            return false;
        }
        frame = std::move(q.front()); // Move the frame instead of copying
        q.pop();
        cv_not_full.notify_one();
        return true;
    }

    void set_done() {
        std::unique_lock<std::mutex> lock(mtx);
        is_done = true;
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }
};

void reader_thread(BlockingQueue &queue, cv::VideoCapture &cap) {
    cv::Mat frame;
    while (cap.read(frame)) {
        queue.enqueue(std::move(frame));
    }
    queue.set_done();
}

void writer_thread(BlockingQueue &queue, cv::VideoWriter &writer) {
    cv::Mat frame;
    while (queue.dequeue(frame)) {
        writer.write(frame);
    }
}

// ./build/canny_opencv
int main(int argc, char *argv[]) {
    std::string input_file{"data/output.mp4"};
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
    std::cout << "Start timing ..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    BlockingQueue reader_queue;
    BlockingQueue writer_queue;
    std::thread reader_thr(reader_thread, std::ref(reader_queue), std::ref(cap));
    std::thread writer_thr(writer_thread, std::ref(writer_queue), std::ref(writer));

    // Process each frame
    cv::Mat frame;
    cv::Mat gray(height, width, CV_8UC1);
    cv::Mat edge_map(height, width, CV_8UC1); // Why can it be instantiated here?
    long long accum_time = 0;
    int frame_count = 0;
    while (reader_queue.dequeue(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edge_map, 100, 200);
        frame_count += 1;

        auto t2 = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        // if edge_map is moved, why can it still be used in the next iteration?
        writer_queue.enqueue(std::move(edge_map));
    }
    assert(frame_count == total_frames);

    std::cout << "Total time of all frame: " << accum_time
              << " us. Each frame average time: " << accum_time / total_frames << " us." << std::endl;

    reader_queue.set_done();
    writer_queue.set_done();
    reader_thr.join();
    writer_thr.join();

    // Release resources
    writer.release();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    std::cout << "Done. Output saved as " << output_file << std::endl;
    return 0;
}
