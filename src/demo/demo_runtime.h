#ifndef DEMO_INCLUDE_DEMO_RUNTIME_H_
#define DEMO_INCLUDE_DEMO_RUNTIME_H_

#include <atomic>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/demo_catalog.h"
#include "demo/feeder.h"
#include "demo/model.h"
#include "demo/overlay.h"
#include "qbruntime/qbruntime.h"

enum class RuntimeExitCode { QUIT_APPLICATION };

class DemoRuntime {
public:
    explicit DemoRuntime(const DemoDefinition& definition);
    RuntimeExitCode run();
    void setDebugMode(bool enabled) { mDebugMode = enabled; }

private:
    void initWindow();
    void loadManifest(const std::string& mode = "");
    void loadLayout();
    void loadModels();
    void loadFeeders();
    void startFeederAll();
    void stopFeederAll();
    void startWorkerAll();
    void stopWorkerAll();
    void ensureWorkerStateStorage(size_t n);
    void fillWorkerTileWithBackground(int worker_index);
    void fillWorkerTileWithBackgroundAll();
    void startProcessing();
    void stopProcessing();
    void display();
    bool keyHandler(int key);
    int getWorkerIndex(int x, int y) const;
    float smoothDisplayFPS(int worker_index, float instant_fps);
    bool isWorkerValid(int worker_index) const;
    bool processWorkerOnce(int worker_index);
    void emitWorkerResult(int worker_index, const cv::Mat& result);
    void perTileWorkerLoop(int worker_index);
    void modelCoreWorkerLoop(size_t model_index, int lane_index, int lane_count);
    void modelCoreAsyncLane(size_t model_index, int lane_index, int lane_count);
    void drainWorkerOutputQueue();
    float computeAverageFPS() const;
    float updateNpuFPS(int worker_index, double npu_ms);
    float computeAverageNpuFPS() const;
    bool isUltralyticsDemo() const;
    PerformanceDisplayMode defaultPerformanceDisplayMode() const;
    PerformanceDisplayMode nextPerformanceDisplayMode(PerformanceDisplayMode mode) const;

    static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

    const std::string mWindowName = "Mobilint CV Demo";
    const DemoDefinition& mDefinition;
    DemoManifest mManifest;
    std::unique_ptr<OverlayRenderer> mOverlayRenderer;

    cv::Mat mDisplay;
    cv::Mat mDisplayBase;
    Benchmarker mBenchmarker;

    PerformanceDisplayMode mPerformanceDisplayMode = PerformanceDisplayMode::OFF;
    bool mDisplayTimeMode = false;
    bool mFullscreen = true;
    bool mDebugMode = false;
    std::string mCurrentMode;

    std::map<int, std::unique_ptr<mobilint::Accelerator>> mAccs;
    std::vector<std::unique_ptr<Model>> mModels;
    std::vector<std::shared_ptr<Feeder>> mFeeders;
    std::vector<uint8_t> mWorkerLayoutValid;
    std::vector<std::vector<int>> mWorkersByModel;
    std::vector<int64_t> mWorkerLastFrameIndex;
    ItemQueue mWorkerOutputQueue;
    WorkerSchedulePolicy mWorkerSchedulePolicy = WorkerSchedulePolicy::MODEL_CORE_POOL;

    std::atomic<bool> mProcessingOn{false};
    std::vector<std::thread> mInferThreads;
    std::vector<std::thread> mFeederThreads;

    std::unique_ptr<std::atomic<uint8_t>[]> mWorkerEnabled;
    size_t mWorkerEnabledSize = 0;
    std::vector<Benchmarker> mWorkerInferBench;
    std::vector<Benchmarker> mWorkerDisplayFPSBench;
    std::vector<float> mWorkerDisplayFPSAvg;
    std::vector<float> mWorkerNpuFPSAvg;
    long mDisplayLogCounter = 0;
};

#endif
