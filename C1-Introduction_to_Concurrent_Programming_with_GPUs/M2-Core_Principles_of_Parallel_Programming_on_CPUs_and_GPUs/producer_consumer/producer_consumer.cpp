#include <iostream>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>

std::mutex mu, cout_mu;
std::condition_variable cond;

class Buffer
{
public:
    void add(int num)
    {
        std::unique_lock<std::mutex> locker(mu);
        cond.wait(locker, [this]()
                  { return buffer_.size() < size_; });
        buffer_.push_back(num);
        locker.unlock();
        cond.notify_one(); // Notify only one waiting consumer
    }

    int remove()
    {
        std::unique_lock<std::mutex> locker(mu);
        cond.wait(locker, [this]()
                  { return !buffer_.empty(); }); // Slightly better check
        int back = buffer_.back();
        buffer_.pop_back();
        locker.unlock();
        cond.notify_one(); // Notify only one waiting producer
        return back;
    }

    Buffer() {}

private:
    std::deque<int> buffer_;
    const unsigned int size_ = 10;
};

class Producer
{
public:
    Producer(Buffer *buffer, std::string name) : buffer_(buffer), name_(name) {}

    void run()
    {
        while (true)
        {
            int num = std::rand() % 100;
            buffer_->add(num);

            int sleep_time = std::rand() % 100;
            {
                std::lock_guard<std::mutex> lock(cout_mu);
                std::cout << "Name: " << name_ << "   Produced: " << num << "   Sleep time: " << sleep_time << "ms" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
    }

private:
    Buffer *buffer_;
    std::string name_;
};

class Consumer
{
public:
    Consumer(Buffer *buffer, std::string name) : buffer_(buffer), name_(name) {}

    void run()
    {
        while (true)
        {
            int num = buffer_->remove();

            int sleep_time = std::rand() % 100;
            {
                std::lock_guard<std::mutex> lock(cout_mu);
                std::cout << "Name: " << name_ << "   Consumed: " << num << "   Sleep time: " << sleep_time << "ms" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
    }

private:
    Buffer *buffer_;
    std::string name_;
};

int main()
{
    Buffer b;
    Producer p1(&b, "producer1");
    Producer p2(&b, "producer2");
    Producer p3(&b, "producer3");
    Consumer c1(&b, "consumer1");
    Consumer c2(&b, "consumer2");
    Consumer c3(&b, "consumer3");

    std::thread producer_thread1(&Producer::run, &p1);
    std::thread producer_thread2(&Producer::run, &p2);
    std::thread producer_thread3(&Producer::run, &p3);

    std::thread consumer_thread1(&Consumer::run, &c1);
    std::thread consumer_thread2(&Consumer::run, &c2);
    std::thread consumer_thread3(&Consumer::run, &c3);

    // Let threads run for some time
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // No `join()` because the threads run indefinitely
    return 0;
}
