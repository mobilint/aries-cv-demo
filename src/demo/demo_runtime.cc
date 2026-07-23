#include "demo/demo_runtime.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "qbruntime/qbruntime.h"

using mobilint::Accelerator;
using mobilint::StatusCode;

namespace {
void sleepForMS(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

const char* feederTypeToString(FeederType type) {
    switch (type) {
        case FeederType::CAMERA:
            return "CAMERA";
        case FeederType::VIDEO:
            return "VIDEO";
        case FeederType::IPCAMERA:
            return "IPCAMERA";
        case FeederType::YOUTUBE:
            return "YOUTUBE";
        default:
            return "UNKNOWN";
    }
}

std::string feederSettingKey(const FeederSetting& setting) {
    std::ostringstream oss;
    oss << feederTypeToString(setting.feeder_type);
    for (const auto& source : setting.sources) {
        oss << '\n' << source;
    }
    return oss.str();
}
}  // namespace

DemoRuntime::DemoRuntime(const DemoDefinition& definition) : mDefinition(definition) {}

void DemoRuntime::initWindow() {
    cv::namedWindow(mWindowName, cv::WINDOW_GUI_NORMAL);
    cv::moveWindow(mWindowName, 0, 0);
    cv::setMouseCallback(mWindowName, onMouseEvent, this);
}

void DemoRuntime::loadManifest(const std::string& mode) {
    const std::string requested_mode = !mode.empty() ? mode : mCurrentMode;
    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] loadManifest begin"
                  << " demo_id=" << mDefinition.id()
                  << " requested_mode=" << (requested_mode.empty() ? "auto" : requested_mode)
                  << std::endl;
    }
    mManifest = mDefinition.loadManifest(requested_mode);
    mCurrentMode = mManifest.active_mode;
    mOverlayRenderer = mDefinition.createOverlayRenderer(mManifest);

    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] loadManifest end"
                  << " active_mode=" << mManifest.active_mode
                  << " feeders=" << mManifest.feeders.size()
                  << " workers=" << mManifest.layout.worker_tiles.size()
                  << " models=" << mManifest.models.size()
                  << " canvas=" << mManifest.layout.canvas_size.width << "x"
                  << mManifest.layout.canvas_size.height << std::endl;
    }

    mPerformanceDisplayMode = defaultPerformanceDisplayMode();
}

void DemoRuntime::loadLayout() {
    mDisplay = cv::Mat(mManifest.layout.canvas_size, CV_8UC3, cv::Scalar(0, 0, 0));
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
    }
    mWorkerInferBench.assign(n, Benchmarker());
    mWorkerDisplayFPSBench.assign(n, Benchmarker());
    for (auto& bench : mWorkerDisplayFPSBench) {
        bench.start();
    }
    mWorkerDisplayFPSAvg.assign(n, 0.0f);
    mWorkerNpuFPSAvg.assign(n, 0.0f);
}

void DemoRuntime::loadModels() {
    mModels.clear();
    mAccs.clear();
    auto pipelines = mDefinition.createPipelines(mManifest);
    mModels.resize(mManifest.models.size());
    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] loadModels count=" << mManifest.models.size() << std::endl;
    }
    for (size_t i = 0; i < mManifest.models.size(); ++i) {
        const auto& model_setting = mManifest.models[i];
        if (isDemoVerboseLogEnabled()) {
            std::cerr << "[RUNTIME] model"
                      << " index=" << i
                      << " dev_no=" << model_setting.dev_no
                      << " num_core=" << model_setting.num_core
                      << " use_core_id=" << (model_setting.use_core_id ? "true" : "false")
                      << " mxq=" << model_setting.mxq_path << std::endl;
        }
        auto it = mAccs.find(model_setting.dev_no);
        if (it == mAccs.end()) {
            StatusCode sc;
            auto acc = Accelerator::create(model_setting.dev_no, sc);
            if (!sc || !acc) {
                throw std::runtime_error("Failed to create accelerator dev_no=" +
                                         std::to_string(model_setting.dev_no) +
                                         ", status=" + std::to_string(static_cast<int>(sc)));
            }
            mAccs.emplace(model_setting.dev_no, std::move(acc));
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
    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] loadFeeders count=" << mManifest.feeders.size() << std::endl;
    }
    std::unordered_map<std::string, std::shared_ptr<Feeder>> feeder_by_source;
    std::unordered_map<std::string, size_t> first_index_by_source;
    for (size_t i = 0; i < mManifest.feeders.size(); ++i) {
        const std::string key = feederSettingKey(mManifest.feeders[i]);
        auto it = feeder_by_source.find(key);
        if (it != feeder_by_source.end()) {
            mFeeders[i] = it->second;
            if (isDemoVerboseLogEnabled()) {
                std::cerr << "[RUNTIME] reuse feeder decoder"
                          << " logical_index=" << i
                          << " physical_index=" << first_index_by_source[key]
                          << " source="
                          << (mManifest.feeders[i].sources.empty() ? "" : mManifest.feeders[i].sources.front())
                          << std::endl;
            }
            continue;
        }

        auto feeder = std::make_shared<Feeder>(mManifest.feeders[i], static_cast<int>(i));
        feeder_by_source.emplace(key, feeder);
        first_index_by_source.emplace(key, i);
        mFeeders[i] = std::move(feeder);
    }
    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] loadFeeders physical_decoders=" << feeder_by_source.size()
                  << " logical_feeders=" << mManifest.feeders.size() << std::endl;
    }
}

void DemoRuntime::ensureWorkerStateStorage(size_t n) {
    if (mWorkerEnabledSize != n) {
        mWorkerEnabled = std::make_unique<std::atomic<uint8_t>[]>(n);
        mWorkerEnabledSize = n;
    }
}

void DemoRuntime::fillWorkerTileWithBackground(int worker_index) {
    if (worker_index < 0) return;
    const size_t wi = static_cast<size_t>(worker_index);
    if (wi >= mManifest.layout.worker_tiles.size()) return;

    const auto& roi = mManifest.layout.worker_tiles[wi].roi;
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
        roi.x + roi.width > mDisplay.cols || roi.y + roi.height > mDisplay.rows) {
        return;
    }

    if (mDisplayBase.empty() || roi.x + roi.width > mDisplayBase.cols ||
        roi.y + roi.height > mDisplayBase.rows) {
        return;
    }
    mDisplayBase(roi).copyTo(mDisplay(roi));
}

void DemoRuntime::fillWorkerTileWithBackgroundAll() {
    for (size_t i = 0; i < mManifest.layout.worker_tiles.size(); ++i) {
        fillWorkerTileWithBackground(static_cast<int>(i));
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
        bool already_started = false;
        for (size_t j = 0; j < i; ++j) {
            if (mFeeders[j].get() == mFeeders[i].get()) {
                already_started = true;
                break;
            }
        }
        if (already_started) {
            if (isDemoVerboseLogEnabled()) {
                std::cerr << "[RUNTIME] skip duplicate feeder thread logical_index=" << i << std::endl;
            }
            continue;
        }
        if (isDemoVerboseLogEnabled()) {
            std::cerr << "[RUNTIME] start feeder thread index=" << i << std::endl;
        }
        mFeeders[i]->start();
        mFeederThreads[i] = std::thread(&Feeder::produceFrames, mFeeders[i].get());
    }
}

void DemoRuntime::stopFeederAll() {
    std::set<Feeder*> stopped;
    for (auto& feeder : mFeeders) {
        if (!feeder) continue;
        if (!stopped.insert(feeder.get()).second) continue;
        feeder->stop();
    }
    for (auto& thread : mFeederThreads) {
        if (thread.joinable()) thread.join();
    }
    mFeederThreads.clear();
}

void DemoRuntime::startWorkerAll() {
    for (size_t i = 0; i < mWorkerEnabledSize; ++i) {
        mWorkerEnabled[i].store(1, std::memory_order_relaxed);
        if (i < mManifest.layout.worker_tiles.size()) {
            const auto& roi = mManifest.layout.worker_tiles[i].roi;
            if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
                roi.x + roi.width <= mDisplay.cols && roi.y + roi.height <= mDisplay.rows) {
                mDisplayBase(roi).copyTo(mDisplay(roi));
            }
        }
    }
}

void DemoRuntime::stopWorkerAll() {
    for (size_t i = 0; i < mWorkerEnabledSize; ++i) {
        mWorkerEnabled[i].store(0, std::memory_order_relaxed);
    }
    fillWorkerTileWithBackgroundAll();
}

void DemoRuntime::startProcessing() {
    mWorkerLayoutValid.assign(mManifest.layout.worker_tiles.size(), 0);
    mWorkersByModel.assign(mModels.size(), {});
    mWorkerLastFrameIndex.assign(mManifest.layout.worker_tiles.size(), 0);
    mWorkerNpuFPSAvg.assign(mManifest.layout.worker_tiles.size(), 0.0f);
    mWorkerOutputQueue.open();
    mWorkerOutputQueue.clear();

    for (size_t wi = 0; wi < mManifest.layout.worker_tiles.size(); ++wi) {
        const auto& worker = mManifest.layout.worker_tiles[wi];
        const bool valid = worker.feeder_index >= 0 &&
                           worker.feeder_index < static_cast<int>(mFeeders.size()) &&
                           worker.model_index >= 0 &&
                           worker.model_index < static_cast<int>(mModels.size());
        if (!valid) {
            if (isDemoVerboseLogEnabled()) {
                std::cerr << "[RUNTIME] invalid worker layout"
                          << " index=" << wi
                          << " feeder_index=" << worker.feeder_index
                          << " model_index=" << worker.model_index
                          << " feeders=" << mFeeders.size()
                          << " models=" << mModels.size() << std::endl;
            }
            continue;
        }
        mWorkerLayoutValid[wi] = 1;
        mWorkersByModel[worker.model_index].push_back(static_cast<int>(wi));
    }

    if (isDemoVerboseLogEnabled()) {
        for (size_t mi = 0; mi < mWorkersByModel.size(); ++mi) {
            std::cerr << "[RUNTIME] model worker assignment"
                      << " model_index=" << mi
                      << " worker_count=" << mWorkersByModel[mi].size() << std::endl;
        }
    }

    for (size_t mi = 0; mi < mModels.size(); ++mi) {
        mModels[mi]->initWorkers(mWorkersByModel[mi]);
    }

    if (mProcessingOn.exchange(true)) return;
    mInferThreads.clear();

    if (isDemoVerboseLogEnabled()) {
        std::cerr << "[RUNTIME] startProcessing policy="
                  << (mWorkerSchedulePolicy == WorkerSchedulePolicy::PER_TILE_THREAD ? "PER_TILE_THREAD"
                                                                                      : "MODEL_CORE_POOL")
                  << std::endl;
    }

    if (mWorkerSchedulePolicy == WorkerSchedulePolicy::PER_TILE_THREAD) {
        mInferThreads.reserve(mManifest.layout.worker_tiles.size());
        for (size_t wi = 0; wi < mManifest.layout.worker_tiles.size(); ++wi) {
            if (!mWorkerLayoutValid[wi]) continue;
            mInferThreads.emplace_back(&DemoRuntime::perTileWorkerLoop, this, static_cast<int>(wi));
        }
    } else {
        // Async models keep cores fed via depth-2 pipelining, so they use
        // num_core lanes; only sync models honor the worker_threads knob.
        auto lane_count_for = [&](size_t mi) {
            const ModelSetting& ms = mManifest.models[mi];
            if (mModels[mi]->supportsAsync()) return std::max(1, ms.num_core);
            return ms.worker_threads > 0 ? ms.worker_threads : std::max(1, ms.num_core);
        };
        size_t reserve_count = 0;
        for (size_t mi = 0; mi < mManifest.models.size(); ++mi) {
            if (mi >= mWorkersByModel.size() || mWorkersByModel[mi].empty()) continue;
            reserve_count += static_cast<size_t>(lane_count_for(mi));
        }
        mInferThreads.reserve(reserve_count);
        for (size_t mi = 0; mi < mManifest.models.size(); ++mi) {
            if (mi >= mWorkersByModel.size() || mWorkersByModel[mi].empty()) continue;
            const int lane_count = lane_count_for(mi);
            for (int lane = 0; lane < lane_count; ++lane) {
                if (isDemoVerboseLogEnabled()) {
                    std::cerr << "[RUNTIME] start infer thread"
                              << " model_index=" << mi
                              << " lane=" << lane << "/" << lane_count << std::endl;
                }
                mInferThreads.emplace_back(&DemoRuntime::modelCoreWorkerLoop, this, mi, lane,
                                           lane_count);
            }
        }
    }
}

void DemoRuntime::stopProcessing() {
    mProcessingOn.store(false, std::memory_order_relaxed);
    mWorkerOutputQueue.close();
    for (auto& thread : mInferThreads) {
        if (thread.joinable()) thread.join();
    }
    mInferThreads.clear();
    mWorkerOutputQueue.clear();
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

    const bool enable = event == cv::EVENT_LBUTTONDOWN;
    runtime->mWorkerEnabled[worker_index].store(enable ? 1 : 0, std::memory_order_relaxed);
    if (enable) {
        const auto& roi = runtime->mManifest.layout.worker_tiles[worker_index].roi;
        if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
            roi.x + roi.width <= runtime->mDisplay.cols &&
            roi.y + roi.height <= runtime->mDisplay.rows) {
            runtime->mDisplayBase(roi).copyTo(runtime->mDisplay(roi));
        }
    } else {
        runtime->fillWorkerTileWithBackground(worker_index);
    }
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

bool DemoRuntime::isWorkerValid(int worker_index) const {
    if (worker_index < 0) return false;
    const size_t wi = static_cast<size_t>(worker_index);
    if (wi >= mManifest.layout.worker_tiles.size()) return false;
    if (!mWorkerLayoutValid.empty() && !mWorkerLayoutValid[wi]) return false;

    const auto& worker = mManifest.layout.worker_tiles[wi];
    return worker.feeder_index >= 0 && static_cast<size_t>(worker.feeder_index) < mFeeders.size() &&
           worker.model_index >= 0 && static_cast<size_t>(worker.model_index) < mModels.size();
}

bool DemoRuntime::processWorkerOnce(int worker_index) {
    if (!isWorkerValid(worker_index)) return false;

    const auto& worker = mManifest.layout.worker_tiles[worker_index];
    if (mWorkerEnabled[worker_index].load(std::memory_order_relaxed) == 0) return false;

    if (static_cast<size_t>(worker_index) >= mWorkerLastFrameIndex.size()) return false;

    cv::Mat frame;
    if (!mFeeders[worker.feeder_index]->consumeFrame(frame, mWorkerLastFrameIndex[worker_index])) {
        return false;
    }
    if (frame.empty()) return false;

    Benchmarker& infer_bench = mWorkerInferBench[worker_index];
    infer_bench.start();
    cv::Mat result = mModels[worker.model_index]->inference(frame, worker.roi.size(), worker_index);
    infer_bench.end();
    if (result.empty() || result.size() != worker.roi.size()) return false;

    emitWorkerResult(worker_index, result);
    return true;
}

void DemoRuntime::emitWorkerResult(int worker_index, const cv::Mat& result) {
    const auto& worker = mManifest.layout.worker_tiles[worker_index];
    if (result.empty() || result.size() != worker.roi.size()) return;

    Benchmarker& fps_bench = mWorkerDisplayFPSBench[worker_index];
    float display_fps = 0.0f;
    if (fps_bench.isStarted()) {
        fps_bench.end();
        display_fps = smoothDisplayFPS(worker_index, fps_bench.getFPS());
    }
    fps_bench.start();

    const double npu_ms = mModels[worker.model_index]->getLastNpuMs(worker_index);
    const float npu_fps = updateNpuFPS(worker_index, npu_ms);
    Item item{worker_index, result, display_fps, npu_fps};
    if (mPerformanceDisplayMode == PerformanceDisplayMode::TILE_FPS) {
        mOverlayRenderer->renderFrameMetrics(item);
    }
    mWorkerOutputQueue.push(item);
}

void DemoRuntime::perTileWorkerLoop(int worker_index) {
    if (!isWorkerValid(worker_index)) return;

    while (mProcessingOn.load(std::memory_order_relaxed)) {
        if (!processWorkerOnce(worker_index)) {
            const bool enabled = worker_index >= 0 &&
                                 static_cast<size_t>(worker_index) < mWorkerEnabledSize &&
                                 mWorkerEnabled[worker_index].load(std::memory_order_relaxed) != 0;
            sleepForMS(enabled ? 1 : 10);
        }
    }
}

void DemoRuntime::modelCoreWorkerLoop(size_t model_index, int lane_index, int lane_count) {
    if (model_index >= mWorkersByModel.size() || lane_index < 0 || lane_count <= 0 ||
        lane_index >= lane_count) {
        return;
    }

    const auto& workers = mWorkersByModel[model_index];
    if (workers.empty()) return;

    if (mModels[model_index]->supportsAsync()) {
        modelCoreAsyncLane(model_index, lane_index, lane_count);
        return;
    }

    while (mProcessingOn.load(std::memory_order_relaxed)) {
        bool did_work = false;
        for (size_t wi = static_cast<size_t>(lane_index); wi < workers.size();
             wi += static_cast<size_t>(lane_count)) {
            if (!mProcessingOn.load(std::memory_order_relaxed)) return;
            did_work = processWorkerOnce(workers[wi]) || did_work;
        }
        if (!did_work) sleepForMS(1);
    }
}

void DemoRuntime::modelCoreAsyncLane(size_t model_index, int lane_index, int lane_count) {
    const auto& workers = mWorkersByModel[model_index];
    Model* model = mModels[model_index].get();

    constexpr size_t kDepth = 2;
    struct InFlight {
        int worker_index;
        cv::Mat frame;
        cv::Size size;
        mobilint::Future<float> future;
    };
    std::deque<InFlight> inflight;

    auto complete_front = [&]() {
        InFlight f = std::move(inflight.front());
        inflight.pop_front();
        cv::Mat result = model->completeAsync(f.future, f.frame, f.size, f.worker_index);
        emitWorkerResult(f.worker_index, result);
    };
    auto in_flight_has = [&](int wi) {
        for (const auto& f : inflight)
            if (f.worker_index == wi) return true;
        return false;
    };

    while (mProcessingOn.load(std::memory_order_relaxed)) {
        bool did_work = false;
        for (size_t k = static_cast<size_t>(lane_index); k < workers.size();
             k += static_cast<size_t>(lane_count)) {
            if (!mProcessingOn.load(std::memory_order_relaxed)) break;
            const int wi = workers[k];
            if (!isWorkerValid(wi)) continue;
            if (static_cast<size_t>(wi) >= mWorkerEnabledSize ||
                mWorkerEnabled[wi].load(std::memory_order_relaxed) == 0) {
                continue;
            }
            if (static_cast<size_t>(wi) >= mWorkerLastFrameIndex.size()) continue;

            const auto& worker = mManifest.layout.worker_tiles[wi];
            cv::Mat frame;
            if (!mFeeders[worker.feeder_index]->consumeFrame(frame, mWorkerLastFrameIndex[wi])) {
                continue;
            }
            if (frame.empty()) continue;

            // Same worker in flight twice would reuse its input workspace.
            while (in_flight_has(wi)) complete_front();
            while (inflight.size() >= kDepth) complete_front();

            mobilint::StatusCode sc;
            mobilint::Future<float> fut = model->submitAsync(frame, worker.roi.size(), wi, sc);
            if (!sc) {
                cv::Mat fallback;
                cv::resize(frame, fallback, worker.roi.size());
                emitWorkerResult(wi, fallback);
                continue;
            }
            inflight.push_back({wi, frame, worker.roi.size(), std::move(fut)});
            did_work = true;
        }
        if (!did_work) {
            while (!inflight.empty()) complete_front();
            sleepForMS(1);
        }
    }
    while (!inflight.empty()) complete_front();
}

void DemoRuntime::drainWorkerOutputQueue() {
    Item item{};
    while (mWorkerOutputQueue.tryPop(item) == ItemQueue::OK) {
        if (!isWorkerValid(item.index)) continue;
        if (item.img.empty()) continue;
        if (static_cast<size_t>(item.index) < mWorkerEnabledSize &&
            mWorkerEnabled[item.index].load(std::memory_order_relaxed) == 0) {
            continue;
        }

        const auto& worker = mManifest.layout.worker_tiles[item.index];
        if (worker.roi.x < 0 || worker.roi.y < 0 ||
            worker.roi.x + worker.roi.width > mDisplay.cols ||
            worker.roi.y + worker.roi.height > mDisplay.rows) {
            continue;
        }
        if (item.img.size() != worker.roi.size()) {
            cv::resize(item.img, item.img, worker.roi.size());
        }
        item.img.copyTo(mDisplay(worker.roi));
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

float DemoRuntime::updateNpuFPS(int worker_index, double npu_ms) {
    if (worker_index < 0 || static_cast<size_t>(worker_index) >= mWorkerNpuFPSAvg.size() ||
        npu_ms <= 0.0) {
        return 0.0f;
    }

    const float npu_fps = static_cast<float>(1000.0 / npu_ms);
    mWorkerNpuFPSAvg[worker_index] = npu_fps;
    return npu_fps;
}

float DemoRuntime::computeAverageNpuFPS() const {
    float sum = 0.0f;
    int count = 0;
    for (size_t i = 0; i < mWorkerEnabledSize && i < mWorkerNpuFPSAvg.size(); ++i) {
        if (mWorkerEnabled[i].load(std::memory_order_relaxed) == 0) continue;
        if (mWorkerNpuFPSAvg[i] <= 0.0f) continue;
        sum += mWorkerNpuFPSAvg[i];
        count++;
    }
    if (count == 0) return 0.0f;
    return sum / static_cast<float>(count);
}

bool DemoRuntime::isUltralyticsDemo() const { return mManifest.id == "ultralytics"; }

PerformanceDisplayMode DemoRuntime::defaultPerformanceDisplayMode() const {
    return nextPerformanceDisplayMode(PerformanceDisplayMode::OFF);
}

PerformanceDisplayMode DemoRuntime::nextPerformanceDisplayMode(PerformanceDisplayMode mode) const {
    if (isUltralyticsDemo()) {
        return mode == PerformanceDisplayMode::TILE_FPS ? PerformanceDisplayMode::OFF
                                                        : PerformanceDisplayMode::TILE_FPS;
    }
    return mode == PerformanceDisplayMode::AVG_FPS_ONLY ? PerformanceDisplayMode::OFF
                                                        : PerformanceDisplayMode::AVG_FPS_ONLY;
}

void DemoRuntime::display() {
    drainWorkerOutputQueue();
    const float avg_fps = computeAverageFPS();
    const float avg_npu_fps = computeAverageNpuFPS();
    mOverlayRenderer->renderDisplayMetrics(mDisplay, mDisplayBase, mPerformanceDisplayMode,
                                           mDisplayTimeMode, avg_fps, avg_npu_fps,
                                           mBenchmarker.getTimeSinceCreated());
    cv::imshow(mWindowName, mDisplay);

    if (mDebugMode && ++mDisplayLogCounter % 100 == 0) {
        std::cerr << "[FPS] avg display=" << avg_fps << " avg infer=" << avg_npu_fps << std::endl;
    }
}

bool DemoRuntime::keyHandler(int key) {
    if (key == -1) return true;

    if (key == 27) {
        mProcessingOn.store(false, std::memory_order_relaxed);
        return false;
    }

    if (key < 0 || key > 127) return true;
    if (key < 32 || key == 127) return true;

    key = std::tolower(key);

    if (key == 'd') {
        mPerformanceDisplayMode = nextPerformanceDisplayMode(mPerformanceDisplayMode);
    } else if (key == 't') {
        mDisplayTimeMode = !mDisplayTimeMode;
    } else if (key == 'm') {
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
    } else if (key == 'q') {
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
    startFeederAll();
    startWorkerAll();
    startProcessing();
    cv::resizeWindow(mWindowName, mManifest.layout.canvas_size / 2);
    cv::setWindowProperty(mWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    mFullscreen = true;

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
