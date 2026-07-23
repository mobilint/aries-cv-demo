#include "demo/feeder.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

namespace {
namespace fs = std::filesystem;

std::string firstSource(const FeederSetting& setting) {
    return setting.sources.empty() ? std::string() : setting.sources.front();
}

std::string feederLabel(int feeder_index) {
    if (feeder_index < 0) return "[FEEDER]";
    return "[FEEDER " + std::to_string(feeder_index) + "]";
}

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

std::string fourccToString(double value) {
    const int fourcc = static_cast<int>(value);
    std::string out(4, ' ');
    out[0] = static_cast<char>(fourcc & 0xFF);
    out[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    out[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    out[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    for (char& ch : out) {
        if (!std::isprint(static_cast<unsigned char>(ch))) ch = '?';
    }
    return out;
}

std::string backendName(const cv::VideoCapture& cap) {
    if (!cap.isOpened()) return "not-opened";
    try {
        return cap.getBackendName();
    } catch (const cv::Exception& e) {
        return std::string("unknown(") + e.what() + ")";
    }
}

void logVideoSourceFile(int feeder_index, const FeederSetting& setting, const std::string& source) {
    if (!isDemoVerboseLogEnabled()) return;
    if (setting.feeder_type != FeederType::VIDEO) return;
    std::error_code ec;
    const bool exists = fs::exists(source, ec);
    std::cerr << feederLabel(feeder_index) << " video source file"
              << " path=" << source << " exists=" << (exists ? "true" : "false");
    if (exists) {
        const auto size = fs::file_size(source, ec);
        if (!ec) {
            std::cerr << " size=" << size << " bytes"
                      << " (" << std::fixed << std::setprecision(2)
                      << static_cast<double>(size) / (1024.0 * 1024.0) << " MiB)";
        }
    } else if (ec) {
        std::cerr << " fs_error=" << ec.message();
    }
    std::cerr << std::defaultfloat << std::endl;
}

void logCaptureProperties(int feeder_index, const FeederSetting& setting, const cv::VideoCapture& cap,
                          const std::string& prefix) {
    if (!isDemoVerboseLogEnabled()) return;
    std::cerr << feederLabel(feeder_index) << ' ' << prefix
              << " type=" << feederTypeToString(setting.feeder_type)
              << " src=" << firstSource(setting)
              << " opened=" << (cap.isOpened() ? "true" : "false")
              << " backend=" << backendName(cap);
    if (cap.isOpened()) {
        std::cerr << " width=" << cap.get(cv::CAP_PROP_FRAME_WIDTH)
                  << " height=" << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                  << " fps=" << cap.get(cv::CAP_PROP_FPS)
                  << " frame_count=" << cap.get(cv::CAP_PROP_FRAME_COUNT)
                  << " pos_frames=" << cap.get(cv::CAP_PROP_POS_FRAMES)
                  << " fourcc=" << fourccToString(cap.get(cv::CAP_PROP_FOURCC));
    }
    std::cerr << std::endl;
}

void downscaleFrameForBuffer(cv::Mat& frame, int feeder_index) {
    constexpr int kMaxBufferedWidth = 640;
    if (frame.empty() || frame.cols <= kMaxBufferedWidth) return;

    const cv::Size original_size = frame.size();
    const int target_height = std::max(
        1, static_cast<int>(std::lround(static_cast<double>(frame.rows) * kMaxBufferedWidth /
                                        static_cast<double>(frame.cols))));
    cv::resize(frame, frame, cv::Size(kMaxBufferedWidth, target_height), 0.0, 0.0,
               cv::INTER_AREA);
    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(feeder_index) << " frame downscaled"
                  << " from=" << original_size.width << "x" << original_size.height
                  << " to=" << frame.cols << "x" << frame.rows
                  << " max_width=" << kMaxBufferedWidth << std::endl;
    }
}

std::string getYoutubeStream(const std::string& youtube_url) {
#ifdef _MSC_VER
    std::cerr << "Youtube input is not implemented for MSVC.\n";
    return "";
#else
    char buf[128];
    std::string url;
    std::string cmd = "yt-dlp -f \"best[height<=720][width<=1280]\" -g " + youtube_url;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return url;

    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        url += buf;
    }
    pclose(pipe);

    if (!url.empty() && url.back() == '\n') {
        url.pop_back();
    }
    return url;
#endif
}

bool openCaptureBySetting(const FeederSetting& setting, cv::VideoCapture& cap, bool& delay_on,
                          int feeder_index, const char* reason) {
    const std::string source = firstSource(setting);
    const auto start = std::chrono::steady_clock::now();

    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(feeder_index) << " open begin"
                  << " reason=" << reason
                  << " type=" << feederTypeToString(setting.feeder_type)
                  << " src=" << source << std::endl;
    }
    logVideoSourceFile(feeder_index, setting, source);

    switch (setting.feeder_type) {
        case FeederType::CAMERA:
            cap.open(std::stoi(source), cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
            cap.set(cv::CAP_PROP_FPS, 30);
            delay_on = false;
            break;
        case FeederType::VIDEO:
#ifdef _WIN32
            cap.open(source, cv::CAP_FFMPEG);
#else
            cap.open(source);
#endif
            delay_on = true;
            break;
        case FeederType::IPCAMERA:
            cap.open(source);
            delay_on = false;
            break;
        case FeederType::YOUTUBE:
            cap.open(getYoutubeStream(source));
            delay_on = true;
            break;
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(feeder_index) << " open end"
                  << " reason=" << reason
                  << " opened=" << (cap.isOpened() ? "true" : "false")
                  << " elapsed_ms=" << elapsed_ms << std::endl;
    }
    logCaptureProperties(feeder_index, setting, cap, "capture properties after open");

    return cap.isOpened();
}

bool rewindCapture(cv::VideoCapture& cap) {
    if (!cap.isOpened()) return false;
    if (!cap.set(cv::CAP_PROP_POS_FRAMES, 0)) return false;

    const double pos = cap.get(cv::CAP_PROP_POS_FRAMES);
    return pos <= 1.0;
}

void logFeederOpenStatus(int feeder_index, const FeederSetting& setting, const cv::VideoCapture& cap) {
    if (cap.isOpened()) return;

    std::cerr << feederLabel(feeder_index) << " open failure"
              << " type=" << feederTypeToString(setting.feeder_type)
              << " src=" << firstSource(setting) << " opened=false" << std::endl;
}
}  // namespace

Feeder::Feeder(const FeederSetting& feeder_setting, int feeder_index)
    : mFeederSetting(feeder_setting), mFeederIndex(feeder_index) {
    openCaptureBySetting(mFeederSetting, mCap, mDelayOn, mFeederIndex, "constructor");

    if (mFeederSetting.feeder_type == FeederType::VIDEO && mCap.isOpened()) {
        const double fps = mCap.get(cv::CAP_PROP_FPS);
        if (fps >= 1.0 && fps <= 240.0) {
            mVideoFps = fps;
        }
    }

    if (!mCap.isOpened() && !mLoggedOpenFailure) {
        logFeederOpenStatus(mFeederIndex, mFeederSetting, mCap);
        mLoggedOpenFailure = true;
    }
}

bool Feeder::readFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lk(mCapMutex);
    if (!mCap.isOpened()) {
        const bool reopened = openCaptureBySetting(mFeederSetting, mCap, mDelayOn, mFeederIndex,
                                                   "readFrame-reopen");
        if (reopened && mFeederSetting.feeder_type == FeederType::VIDEO) {
            const double fps = mCap.get(cv::CAP_PROP_FPS);
            if (fps >= 1.0 && fps <= 240.0) {
                mVideoFps = fps;
            }
        }
    }
    if (!mCap.isOpened()) {
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        return true;
    }
    mCap >> frame;
    if (frame.empty() && (mFeederSetting.feeder_type == FeederType::VIDEO ||
                          mFeederSetting.feeder_type == FeederType::YOUTUBE)) {
        if (!rewindCapture(mCap)) {
            mCap.release();
            openCaptureBySetting(mFeederSetting, mCap, mDelayOn, mFeederIndex,
                                 "readFrame-reopen-after-rewind-failure");
        }
        mCap >> frame;
    }
    return !frame.empty();
}

bool Feeder::consumeFrame(cv::Mat& frame, int64_t& frame_index) {
    int64_t latest_index = frame_index;
    auto sc = mFeederBuffer.getLatest(frame, latest_index);
    if (sc != MatBuffer::OK) return false;
    if (latest_index == frame_index) return false;
    frame_index = latest_index;
    return !frame.empty();
}

void Feeder::start() {
    if (isDemoVerboseLogEnabled()) std::cerr << feederLabel(mFeederIndex) << " start" << std::endl;
    mFeederBuffer.open();
    mIsRunning.store(true, std::memory_order_relaxed);
}

void Feeder::stop() {
    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(mFeederIndex) << " stop requested produced_frames="
                  << mProducedFrameCount << std::endl;
    }
    mIsRunning.store(false, std::memory_order_relaxed);
    mFeederBuffer.close();
}

void Feeder::produceFrames() {
    mFeederBuffer.open();
    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(mFeederIndex) << " producer loop begin" << std::endl;
    }
    while (mIsRunning.load(std::memory_order_relaxed)) {
        if (mCap.isOpened()) {
            const bool is_loop_source = mFeederSetting.feeder_type == FeederType::VIDEO ||
                                        mFeederSetting.feeder_type == FeederType::YOUTUBE;
            int delay_ms = 0;
            if (mDelayOn) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO && mVideoFps >= 24.0 &&
                    mVideoFps <= 240.0) {
                    delay_ms = std::max(1, static_cast<int>(std::lround(1000.0 / mVideoFps)));
                } else {
                    delay_ms = 33;
                }
            }
            if (isDemoVerboseLogEnabled()) {
                std::cerr << feederLabel(mFeederIndex) << " produceFramesInternal begin"
                          << " delay_ms=" << delay_ms
                          << " loop_source=" << (is_loop_source ? "true" : "false")
                          << std::endl;
            }
            produceFramesInternal(mCap, delay_ms);
            if (!mIsRunning.load(std::memory_order_relaxed)) break;

            if (is_loop_source) {
                const bool rewind_ok = rewindCapture(mCap);
                if (isDemoVerboseLogEnabled()) {
                    std::cerr << feederLabel(mFeederIndex) << " rewind after stream end"
                              << " ok=" << (rewind_ok ? "true" : "false")
                              << " opened=" << (mCap.isOpened() ? "true" : "false")
                              << " pos_frames=" << mCap.get(cv::CAP_PROP_POS_FRAMES)
                              << std::endl;
                }
                if (!rewind_ok || !mCap.isOpened()) {
                    mCap.release();
                    openCaptureBySetting(mFeederSetting, mCap, mDelayOn, mFeederIndex,
                                         "produceFrames-reopen-after-rewind-failure");
                    if (mFeederSetting.feeder_type == FeederType::VIDEO && mCap.isOpened()) {
                        const double fps = mCap.get(cv::CAP_PROP_FPS);
                        if (fps >= 1.0 && fps <= 240.0) {
                            mVideoFps = fps;
                        }
                    }
                }
            } else {
                rewindCapture(mCap);
            }
        } else {
            const bool reopened = openCaptureBySetting(mFeederSetting, mCap, mDelayOn, mFeederIndex,
                                                       "produceFrames-reopen-not-opened");
            if (reopened) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO) {
                    const double fps = mCap.get(cv::CAP_PROP_FPS);
                    if (fps >= 1.0 && fps <= 240.0) {
                        mVideoFps = fps;
                    }
                }
                continue;
            }
            if (!mLoggedOpenFailure) {
                logFeederOpenStatus(mFeederIndex, mFeederSetting, mCap);
                mLoggedOpenFailure = true;
            }
            produceFramesInternalDummy();
        }
    }
    if (isDemoVerboseLogEnabled()) {
        std::cerr << feederLabel(mFeederIndex) << " producer loop end produced_frames="
                  << mProducedFrameCount << std::endl;
    }
    mFeederBuffer.close();
}

void Feeder::produceFramesInternal(cv::VideoCapture& cap, int delay_ms) {
    Benchmarker benchmarker;
    bool has_produced_frame = false;
    while (true) {
        benchmarker.start();
        cv::Mat frame;
        const auto read_start = std::chrono::steady_clock::now();
        cap >> frame;
        const auto read_end = std::chrono::steady_clock::now();
        const auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count();
        if (!mIsRunning.load(std::memory_order_relaxed)) break;
        if (frame.empty()) {
            if (!has_produced_frame && !mLoggedInitialReadFailure) {
                std::cerr << feederLabel(mFeederIndex) << " empty initial frame"
                          << " type=" << feederTypeToString(mFeederSetting.feeder_type)
                          << " src=" << firstSource(mFeederSetting)
                          << " backend=" << backendName(cap)
                          << " read_ms=" << read_ms
                          << " pos_frames=" << cap.get(cv::CAP_PROP_POS_FRAMES)
                          << " frame_count=" << cap.get(cv::CAP_PROP_FRAME_COUNT)
                          << " produced_frames=" << mProducedFrameCount
                          << std::endl;
                mLoggedInitialReadFailure = true;
            } else if (has_produced_frame && isDemoVerboseLogEnabled()) {
                std::cerr << feederLabel(mFeederIndex) << " stream ended or empty frame"
                          << " src=" << firstSource(mFeederSetting)
                          << " backend=" << backendName(cap)
                          << " read_ms=" << read_ms
                          << " pos_frames=" << cap.get(cv::CAP_PROP_POS_FRAMES)
                          << " frame_count=" << cap.get(cv::CAP_PROP_FRAME_COUNT)
                          << " produced_frames=" << mProducedFrameCount
                          << std::endl;
            }
            break;
        }
        if (!mLoggedFirstFrame && isDemoVerboseLogEnabled()) {
            std::cerr << feederLabel(mFeederIndex) << " first frame read"
                      << " src=" << firstSource(mFeederSetting)
                      << " backend=" << backendName(cap)
                      << " read_ms=" << read_ms
                      << " size=" << frame.cols << "x" << frame.rows
                      << " channels=" << frame.channels()
                      << " type=" << frame.type()
                      << " pos_frames=" << cap.get(cv::CAP_PROP_POS_FRAMES)
                      << std::endl;
        }
        mLoggedFirstFrame = true;
        downscaleFrameForBuffer(frame, mFeederIndex);
        mFeederBuffer.put(frame);
        has_produced_frame = true;
        ++mProducedFrameCount;

        if (delay_ms > 0) {
            int remaining_ms = delay_ms;
            while (remaining_ms > 0 && mIsRunning.load(std::memory_order_relaxed)) {
                const int chunk_ms = std::min(remaining_ms, 5);
                std::this_thread::sleep_for(std::chrono::milliseconds(chunk_ms));
                remaining_ms -= chunk_ms;
            }
        }

        benchmarker.end();
    }
}

void Feeder::produceFramesInternalDummy() {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();
        cv::Mat frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190), cv::FONT_HERSHEY_DUPLEX,
                    1.5, cv::Scalar(0, 255, 0), 2);
        if (frame.empty() || !mIsRunning.load(std::memory_order_relaxed)) break;
        mFeederBuffer.put(frame);
        ++mProducedFrameCount;
        for (int remaining_ms = 30; remaining_ms > 0 && mIsRunning.load(std::memory_order_relaxed);
             remaining_ms -= 5) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(remaining_ms, 5)));
        }
        benchmarker.end();
    }
}
