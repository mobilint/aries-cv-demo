#ifndef DEMO_INCLUDE_DEMO_RUNTIME_H_
#define DEMO_INCLUDE_DEMO_RUNTIME_H_

#include <atomic>
#include <map>
#include <memory>
#include <thread>

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
    void startProcessing();
    void stopProcessing();
    void display();
    bool keyHandler(int key);
    int getWorkerIndex(int x, int y) const;
    float smoothDisplayFPS(int worker_index, float instant_fps);
    void modelInferLoop(size_t model_index, int core_index, int core_count);
    float computeAverageFPS() const;

    static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

    const std::string mWindowName = "Mobilint CV Demo";
    const DemoDefinition& mDefinition;
    DemoManifest mManifest;
    std::unique_ptr<OverlayRenderer> mOverlayRenderer;

    std::mutex mDisplayMutex;
    cv::Mat mDisplay;
    cv::Mat mDisplayBase;
    Benchmarker mBenchmarker;

    bool mDisplayFPSMode = true;
    bool mDisplayTimeMode = true;
    bool mFullscreen = false;
    bool mDebugMode = false;
    std::string mCurrentMode;

    std::map<int, std::unique_ptr<mobilint::Accelerator>> mAccs;
    std::vector<std::unique_ptr<Model>> mModels;
    std::vector<std::unique_ptr<Feeder>> mFeeders;
    std::vector<uint8_t> mWorkerLayoutValid;
    std::vector<std::vector<int>> mWorkersByModel;

    std::atomic<bool> mProcessingOn{false};
    std::vector<std::thread> mInferThreads;
    std::vector<std::thread> mFeederThreads;
    ItemQueue mRenderQueue;

    std::unique_ptr<std::atomic<uint8_t>[]> mWorkerEnabled;
    size_t mWorkerEnabledSize = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> mWorkerInFlight;
    size_t mWorkerInFlightSize = 0;
    std::vector<Benchmarker> mWorkerInferBench;
    std::vector<Benchmarker> mWorkerDisplayFPSBench;
    std::vector<float> mWorkerDisplayFPSAvg;
};

#endif
