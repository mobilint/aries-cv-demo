#include "demo/demo_runtime.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "qbruntime/qbruntime.h"

using mobilint::Accelerator;
using mobilint::StatusCode;

namespace {
void sleepForMS(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

class InFlightGuard {
   public:
    explicit InFlightGuard(std::atomic<uint8_t>* slot) : mSlot(slot) {}

    bool tryAcquire() {
        if (mSlot == nullptr) return false;
        uint8_t expected = 0;
        mAcquired = mSlot->compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        return mAcquired;
    }

    ~InFlightGuard() {
        if (mAcquired && mSlot != nullptr) {
            mSlot->store(0, std::memory_order_relaxed);
        }
    }

   private:
    std::atomic<uint8_t>* mSlot = nullptr;
    bool mAcquired = false;
};
}  // namespace

DemoRuntime::DemoRuntime(const DemoDefinition& definition) : mDefinition(definition) {}

void DemoRuntime::initWindow() {
    cv::namedWindow(mWindowName, cv::WINDOW_GUI_NORMAL);
    cv::moveWindow(mWindowName, 0, 0);
    cv::setMouseCallback(mWindowName, onMouseEvent, this);
}

void DemoRuntime::loadManifest(const std::string& mode) {
    const std::string requested_mode = !mode.empty() ? mode : mCurrentMode;
    mManifest = mDefinition.loadManifest(requested_mode);
    mCurrentMode = mManifest.active_mode;
    mOverlayRenderer = mDefinition.createOverlayRenderer(mManifest);
}

void DemoRuntime::loadLayout() {
    mDisplay = cv::Mat(mManifest.layout.canvas_size, CV_8UC3, cv::Scalar(255, 255, 255));
    mDisplayBase = mDisplay.clone();
    for (const auto& image_layout : mManifest.layout.background_images) {
        if (image_layout.img.empty()) continue;
        image_layout.img.copyTo(mDisplayBase(image_layout.roi));
    }
    mDisplayBase.copyTo(mDisplay);

    const size_t n = mManifest.layout.worker_tiles.size();
    ensureWorkerStateStorage(n);
    for (size_t i = 0; i < n; ++i) {
        mWorkerEnabled[i].store(1, std::memory_order_relaxed);
        mWorkerInFlight[i].store(0, std::memory_order_relaxed);
    }
    mWorkerInferBench.assign(n, Benchmarker());
    mWorkerDisplayFPSBench.assign(n, Benchmarker());
    for (auto& bench : mWorkerDisplayFPSBench) {
        bench.start();
    }
    mWorkerDisplayFPSAvg.assign(n, 0.0f);
}

void DemoRuntime::loadModels() {
    mModels.clear();
    mAccs.clear();
    auto pipelines = mDefinition.createPipelines(mManifest);
    mModels.resize(mManifest.models.size());
    for (size_t i = 0; i < mManifest.models.size(); ++i) {
        const auto& model_setting = mManifest.models[i];
        auto it = mAccs.find(model_setting.dev_no);
        if (it == mAccs.end()) {
            StatusCode sc;
            mAccs.emplace(model_setting.dev_no, Accelerator::create(model_setting.dev_no, sc));
        }
        mModels[i] = std::make_unique<Model>(model_setting, *mAccs[model_setting.dev_no],
                                             std::move(pipelines[i]));
        if (mDebugMode) mModels[i]->setDebugMode(true);
    }
}

void DemoRuntime::loadFeeders() {
    stopFeederAll();
    mFeeders.resize(mManifest.feeders.size());
    mFeederThreads.clear();
    mFeederThreads.resize(mManifest.feeders.size());
    for (size_t i = 0; i < mManifest.feeders.size(); ++i) {
        mFeeders[i] = std::make_unique<Feeder>(mManifest.feeders[i]);
    }
}

void DemoRuntime::ensureWorkerStateStorage(size_t n) {
    if (mWorkerEnabledSize != n) {
        mWorkerEnabled = std::make_unique<std::atomic<uint8_t>[]>(n);
        mWorkerEnabledSize = n;
    }
    if (mWorkerInFlightSize != n) {
        mWorkerInFlight = std::make_unique<std::atomic<uint8_t>[]>(n);
        mWorkerInFlightSize = n;
    }
}

void DemoRuntime::startFeederAll() {
    if (mFeederThreads.size() != mFeeders.size()) {
        mFeederThreads.clear();
        mFeederThreads.resize(mFeeders.size());
    }
    for (size_t i = 0; i < mFeeders.size(); ++i) {
        if (!mFeeders[i]) continue;
        if (mFeederThreads[i].joinable()) continue;
        mFeeders[i]->start();
        mFeederThreads[i] = std::thread(&Feeder::produceFrames, mFeeders[i].get());
    }
}

void DemoRuntime::stopFeederAll() {
    for (auto& feeder : mFeeders) {
        if (feeder) feeder->stop();
    }
    for (auto& thread : mFeederThreads) {
        if (thread.joinable()) thread.join();
    }
    mFeederThreads.clear();
}

void DemoRuntime::startWorkerAll() {
    for (size_t i = 0; i < mWorkerEnabledSize; ++i) {
        mWorkerEnabled[i].store(1, std::memory_order_relaxed);
    }
}

void DemoRuntime::stopWorkerAll() {
    for (size_t i = 0; i < mWorkerEnabledSize; ++i) {
        mWorkerEnabled[i].store(0, std::memory_order_relaxed);
    }
}

void DemoRuntime::startProcessing() {
    mWorkerLayoutValid.assign(mManifest.layout.worker_tiles.size(), 0);
    mWorkersByModel.assign(mModels.size(), {});

    for (size_t wi = 0; wi < mManifest.layout.worker_tiles.size(); ++wi) {
        const auto& worker = mManifest.layout.worker_tiles[wi];
        const bool valid = worker.feeder_index >= 0 &&
                           worker.feeder_index < static_cast<int>(mFeeders.size()) &&
                           worker.model_index >= 0 &&
                           worker.model_index < static_cast<int>(mModels.size());
        if (!valid) continue;
        mWorkerLayoutValid[wi] = 1;
        mWorkersByModel[worker.model_index].push_back(static_cast<int>(wi));
    }

    for (size_t mi = 0; mi < mModels.size(); ++mi) {
        mModels[mi]->initWorkers(mWorkersByModel[mi]);
    }

    if (mProcessingOn.exchange(true)) return;
    mInferThreads.clear();
    for (size_t mi = 0; mi < mManifest.models.size(); ++mi) {
        int core_count = mManifest.models[mi].num_core;
        if (core_count <= 0) continue;
        for (int ci = 0; ci < core_count; ++ci) {
            mInferThreads.emplace_back(&DemoRuntime::modelInferLoop, this, mi, ci, core_count);
        }
    }
}

void DemoRuntime::stopProcessing() {
    mProcessingOn.store(false, std::memory_order_relaxed);
    for (auto& thread : mInferThreads) {
        if (thread.joinable()) thread.join();
    }
    mInferThreads.clear();
}

int DemoRuntime::getWorkerIndex(int x, int y) const {
    for (size_t i = 0; i < mManifest.layout.worker_tiles.size(); ++i) {
        if (mManifest.layout.worker_tiles[i].roi.contains(cv::Point(x, y))) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void DemoRuntime::onMouseEvent(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) return;

    auto* runtime = static_cast<DemoRuntime*>(userdata);
    const int worker_index = runtime->getWorkerIndex(x, y);
    if (worker_index < 0) return;
    if (static_cast<size_t>(worker_index) >= runtime->mWorkerEnabledSize) return;

    runtime->mWorkerEnabled[worker_index].store(event == cv::EVENT_LBUTTONDOWN ? 1 : 0,
                                                std::memory_order_relaxed);
}

float DemoRuntime::smoothDisplayFPS(int worker_index, float instant_fps) {
    if (worker_index < 0 || static_cast<size_t>(worker_index) >= mWorkerDisplayFPSAvg.size()) {
        return instant_fps;
    }
    float& avg = mWorkerDisplayFPSAvg[worker_index];
    if (avg <= 0.0f) {
        avg = instant_fps;
    } else {
        avg = 0.8f * avg + 0.2f * instant_fps;
    }
    return avg;
}

void DemoRuntime::modelInferLoop(size_t model_index, int core_index, int core_count) {
    if (model_index >= mWorkersByModel.size()) return;
    if (core_index < 0 || core_count <= 0 || core_index >= core_count) return;

    const auto& workers = mWorkersByModel[model_index];
    if (workers.empty()) return;

    while (mProcessingOn.load(std::memory_order_relaxed)) {
        for (size_t wi = static_cast<size_t>(core_index); wi < workers.size();
             wi += static_cast<size_t>(core_count)) {
            if (!mProcessingOn.load(std::memory_order_relaxed)) return;

            const int worker_index = workers[wi];
            if (worker_index < 0 ||
                static_cast<size_t>(worker_index) >= mManifest.layout.worker_tiles.size())
                continue;
            if (!mWorkerLayoutValid[worker_index]) continue;
            if (mWorkerEnabled[worker_index].load(std::memory_order_relaxed) == 0) continue;

            const auto& worker = mManifest.layout.worker_tiles[worker_index];

            cv::Mat frame;
            if (!mFeeders[worker.feeder_index]->readFrame(frame)) continue;
            if (frame.empty()) continue;

            Benchmarker& infer_bench = mWorkerInferBench[worker_index];
            infer_bench.start();
            cv::Mat result =
                mModels[worker.model_index]->inference(frame, worker.roi.size(), worker_index);
            infer_bench.end();
            if (result.empty() || result.size() != worker.roi.size()) continue;

            Benchmarker& fps_bench = mWorkerDisplayFPSBench[worker_index];
            float display_fps = 0.0f;
            if (fps_bench.isStarted()) {
                fps_bench.end();
                display_fps = smoothDisplayFPS(worker_index, fps_bench.getFPS());
            }
            fps_bench.start();

            if (mDisplayFPSMode) {
                double npu_ms = mModels[worker.model_index]->getLastNpuMs(worker_index);
                Item item{worker_index, result, display_fps, npu_ms,
                          infer_bench.getCount()};
                mOverlayRenderer->renderFrameMetrics(item);
                result = item.img;
            }

            result.copyTo(mDisplay(worker.roi));
        }
    }
}

float DemoRuntime::computeAverageFPS() const {
    float sum = 0.0f;
    int count = 0;
    for (size_t i = 0; i < mWorkerEnabledSize && i < mWorkerDisplayFPSAvg.size(); ++i) {
        if (mWorkerEnabled[i].load(std::memory_order_relaxed) == 0) continue;
        if (mWorkerDisplayFPSAvg[i] <= 0.0f) continue;
        sum += mWorkerDisplayFPSAvg[i];
        count++;
    }
    if (count == 0) return 0.0f;
    return sum / static_cast<float>(count);
}

void DemoRuntime::display() {
    mOverlayRenderer->renderDisplayMetrics(mDisplay, mDisplayBase, mDisplayFPSMode,
                                           mDisplayTimeMode, computeAverageFPS(),
                                           mBenchmarker.getTimeSinceCreated());
    cv::imshow(mWindowName, mDisplay);
}

bool DemoRuntime::keyHandler(int key) {
    if (key == -1) return true;
    if (key >= 128) key -= 128;
    key = std::tolower(key);

    if (key == 'd')
        mDisplayFPSMode = !mDisplayFPSMode;
    else if (key == 't')
        mDisplayTimeMode = !mDisplayTimeMode;
    else if (key == 'm') {
        mFullscreen = !mFullscreen;
        cv::setWindowProperty(mWindowName, cv::WND_PROP_FULLSCREEN,
                              mFullscreen ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        if (!mFullscreen) {
            cv::resizeWindow(mWindowName, mManifest.layout.canvas_size / 2);
        }
    } else if (key == 'c') {
        stopWorkerAll();
    } else if (key == 'f') {
        startWorkerAll();
    } else if (key == 'q' || key == 27) {
        mProcessingOn.store(false, std::memory_order_relaxed);
        return false;
    }
    return true;
}

RuntimeExitCode DemoRuntime::run() {
    initWindow();
    loadManifest();
    loadLayout();
    loadModels();
    loadFeeders();
    startWorkerAll();
    startProcessing();
    cv::resizeWindow(mWindowName, mManifest.layout.canvas_size / 2);

    while (true) {
        display();
        if (!keyHandler(cv::waitKey(10))) break;
    }

    stopProcessing();
    stopFeederAll();
    cv::setMouseCallback(mWindowName, nullptr, nullptr);
    cv::destroyWindow(mWindowName);
    return RuntimeExitCode::QUIT_APPLICATION;
}
