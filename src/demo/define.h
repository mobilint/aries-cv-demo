#ifndef DEMO_INCLUDE_DEFINE_H_
#define DEMO_INCLUDE_DEFINE_H_

#include <condition_variable>
#include <map>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "qbruntime/type.h"
#include "opencv2/opencv.hpp"

enum class FeederType { CAMERA, VIDEO, IPCAMERA, YOUTUBE };
enum class InputDataType { UINT8, FLOAT32 };
enum class PipelineType { YOLO11_DET, YOLO26_DET, YOLO_ANCHORLESS_DET };

struct FeederSetting {
    FeederType feeder_type;
    std::vector<std::string> sources;
};

struct PipelineConfig {
    int num_classes = 2;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    bool decode_bbox = true;
    bool draw_score_text = true;
};

struct ModelSetting {
    PipelineType pipeline_type = PipelineType::YOLO26_DET;
    InputDataType input_type = InputDataType::FLOAT32;
    std::string mxq_path;
    int dev_no;
    int num_core = 1;
    std::vector<mobilint::CoreId> core_id;
    bool use_core_id = false;
    PipelineConfig pipeline_config;
};

struct BackgroundImageLayout {
    cv::Mat img;
    cv::Rect roi;
};

struct WorkerLayout {
    int feeder_index;
    int model_index;
    cv::Rect roi;
};

struct LayoutSetting {
    cv::Size canvas_size;
    std::string preview_asset;
    std::vector<std::string> splash_assets;
    std::vector<BackgroundImageLayout> background_images;
    std::vector<WorkerLayout> worker_tiles;
};

struct DemoUISetting {
    std::string overlay_style;
};

struct DemoModeSetting {
    std::string layout_setting;
    std::string feeder_setting;
    std::string model_setting;
};

struct DemoManifest {
    std::string id;
    std::string title;
    std::string manifest_path;
    std::string manifest_dir;
    std::string active_mode;
    std::map<std::string, DemoModeSetting> modes;
    LayoutSetting layout;
    std::vector<FeederSetting> feeders;
    std::vector<ModelSetting> models;
    DemoUISetting ui;
};

struct Item {
    int index;
    cv::Mat img;
    double fps;
    double time;
    size_t count;
};

// Main To Feeder, Worker
// Feeder와 Worker에서 Display할 Mat을 push하고
// Main에서 pop하여 Display한다.
// Main에서 close하면 Watchdog은 break하고 join된다.
template <typename T>
class ThreadSafeQueue {
public:
    enum StatusCode { OK = 0, CLOSED = 1, EMPTY = 2 };

    StatusCode push(const T& value) {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mQueue.push(value);
        }
        mCV.notify_one();
        return OK;
    }

    StatusCode pop(T& value) {
        std::unique_lock<std::mutex> lk(mMutex);
        mCV.wait(lk, [this] { return !mQueue.empty() || !mOn; });
        if (mQueue.empty()) {
            return CLOSED;
        }
        value = std::move(mQueue.front());
        mQueue.pop();
        return OK;
    }

    StatusCode tryPop(T& value) {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mQueue.empty()) {
            return mOn ? EMPTY : CLOSED;
        }
        value = std::move(mQueue.front());
        mQueue.pop();
        return OK;
    }

    void clear() {
        std::unique_lock<std::mutex> lk(mMutex);
        while (!mQueue.empty()) {
            mQueue.pop();
        }
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = false;
        }
        mCV.notify_all();
    }

private:
    std::mutex mMutex;
    std::condition_variable mCV;
    std::queue<T> mQueue;
    bool mOn = true;
};

using ItemQueue = ThreadSafeQueue<Item>;

// Feeder To Worker
// Feeder에서 공급된 Frame을 put하고
// Worker에서 get하여 infer한다.
// Worker는 Feeder가 죽어 close 된 상태이면 get 이후 break한다.
template <typename T>
class ThreadSafeBuffer {
public:
    enum StatusCode { OK = 0, CLOSED = 1 };

    StatusCode put(const T& value) {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mBuffer = value;
            mBufferIndex++;
        }
        mCV.notify_all();
        return OK;
    }

    StatusCode get(T& value, int64_t& index) {
        std::unique_lock<std::mutex> lk(mMutex);
        mCV.wait(lk, [this, index] { return mBufferIndex > index || !mOn; });
        if (!mOn) {
            return CLOSED;
        }
        value = mBuffer;
        index = mBufferIndex;

        return OK;
    }

    // Returns the latest buffered value without waiting for a new one.
    // - If no frame has been put yet, value may be default-constructed (e.g., empty cv::Mat).
    StatusCode getLatest(T& value, int64_t& index) {
        std::lock_guard<std::mutex> lk(mMutex);
        if (!mOn) {
            return CLOSED;
        }
        value = mBuffer;
        index = mBufferIndex;
        return OK;
    }

    void open() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = true;
        }
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = false;
        }
        mCV.notify_all();
    }

private:
    mutable std::mutex mMutex;
    std::condition_variable mCV;
    T mBuffer;
    int64_t mBufferIndex = 0;
    bool mOn = true;
};

using MatBuffer = ThreadSafeBuffer<cv::Mat>;

// Main to Worker
// Main에서 Display 할 Size를 update하고
// Worker에서 checkUpdate하여 resize한다.
// WorkerStart 이전에 open을 하고, WorkerStop시 close를 한다.
template <typename T>
class ThreadSafeState {
public:
    enum StatusCode { OK = 0, CLOSED = 1 };

    StatusCode update(const T& value) {
        {
            std::unique_lock<std::mutex> lk(mMutex);
            mCheckTarget = value;
            mIsUpdated = true;
        }
        mCV.notify_all();
        return OK;
    }

    StatusCode checkUpdate(T& value) {
        std::unique_lock<std::mutex> lk(mMutex);
        T empty = T();  // 기본생성자, 즉 별도의 값이 없는 빈 상태이면
        mCV.wait(lk, [this, empty] { return mCheckTarget != empty || !mOn; });
        if (!mOn) {
            return CLOSED;
        }

        if (mIsUpdated) {
            value = mCheckTarget;
            mIsUpdated = false;
        }

        return OK;
    }

    void open() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = true;
            mIsUpdated = true;
        }
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = false;
        }
        mCV.notify_all();
    }

private:
    std::mutex mMutex;
    std::condition_variable mCV;
    T mCheckTarget;
    bool mIsUpdated = false;
    bool mOn = false;
};

using SizeState = ThreadSafeState<cv::Size>;

// Copy from https://modoocode.com/285
class ThreadPool {
public:
    ThreadPool(int num_threads) : mOn(true) {
        mThreads.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            mThreads.emplace_back([this]() { this->worker(); });
        }
    }
    ~ThreadPool() {
        mOn = false;
        mCV.notify_all();

        for (auto& t : mThreads) {
            t.join();
        }
    }

    template <class F, class... Args>
    std::future<void> enqueue(F&& f, Args&&... args) {
        // future 예외처리를 어찌해야할지 모르겠다.
        // 우선 빈 future를 반환하고, valid를 체크하자..
        if (!mOn) {
            return std::future<void>();
        }
        auto task = std::make_shared<std::packaged_task<void()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<void> future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mTaskQueue.push([task]() { (*task)(); });
        }
        mCV.notify_one();

        return future;
    }

private:
    std::vector<std::thread> mThreads;
    std::queue<std::function<void()>> mTaskQueue;
    std::condition_variable mCV;
    std::mutex mMutex;
    bool mOn;

    void worker() {
        while (true) {
            std::unique_lock<std::mutex> lock(mMutex);
            mCV.wait(lock, [this]() { return !this->mTaskQueue.empty() || !mOn; });
            if (!mOn && this->mTaskQueue.empty()) {
                return;
            }

            std::function<void()> task = std::move(mTaskQueue.front());
            mTaskQueue.pop();
            lock.unlock();

            task();
        }
    }
};
#endif
