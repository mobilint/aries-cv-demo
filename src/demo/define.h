#ifndef DEMO_INCLUDE_DEFINE_H_
#define DEMO_INCLUDE_DEFINE_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "qbruntime/type.h"
#include "opencv2/opencv.hpp"

enum class FeederType { CAMERA, VIDEO, IPCAMERA, YOUTUBE };
enum class InputDataType { UINT8, FLOAT32 };
enum class ModelType { SSD, FACE, POSE, OBJECT, SEGMENTATION, CLASSIFICATION };
enum class PostProcessType { NONE, ANCHOR, ANCHORLESS, DFLFREE, NMSFREE, YOLO11, YOLO26_INLINE, CLASSIFICATION };
enum class WorkerSchedulePolicy { PER_TILE_THREAD, MODEL_CORE_POOL };

struct FeederSetting {
    FeederType feeder_type;
    std::vector<std::string> sources;
};

struct PipelineConfig {
    int num_classes = 2;
    int topk = 3;
    std::vector<std::string> labels;
    // If non-empty, only detections whose label name is in this list are visualized.
    // Empty (default) means visualize every detection.
    std::vector<std::string> display_labels;
    // Per-label BGR color override (label name → {B, G, R}). Labels not present
    // fall back to the built-in color palette.
    std::map<std::string, std::array<int, 3>> label_colors;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int bbox_thickness = 1;
    bool decode_bbox = true;
    bool draw_label_text = true;
    bool draw_score_text = true;
    bool draw_detection_border = false;
    // Segmentation: when false, the seg postprocessor skips mask matmul+resize
    // and the renderer draws boxes only. No effect on non-seg pipelines.
    bool draw_mask = true;
};

struct ModelSetting {
    ModelType model_type = ModelType::OBJECT;
    InputDataType input_type = InputDataType::FLOAT32;
    PostProcessType post_type = PostProcessType::NONE;
    std::string mxq_path;
    int dev_no = 0;
    int num_core = 1;
    // Inference worker threads (0 = num_core). Ignored for async models, which
    // always use num_core lanes.
    int worker_threads = 0;
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
    double time;  // NPU FPS converted from the latest NPU latency.
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
            if (!mOn) return CLOSED;
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

    void open() {
        std::lock_guard<std::mutex> lk(mMutex);
        mOn = true;
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

    StatusCode peek(int64_t index, bool& next_frame_exists) const {
        std::lock_guard<std::mutex> lk(mMutex);
        if (!mOn) {
            next_frame_exists = false;
            return CLOSED;
        }
        next_frame_exists = mBufferIndex > index;
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

inline std::atomic<bool>& demoVerboseLogEnabled() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

inline void setDemoVerboseLogEnabled(bool enabled) {
    demoVerboseLogEnabled().store(enabled, std::memory_order_relaxed);
}

inline bool isDemoVerboseLogEnabled() {
    return demoVerboseLogEnabled().load(std::memory_order_relaxed);
}

#endif
