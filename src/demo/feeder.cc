#include "demo/feeder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

namespace {
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
}  // namespace

Feeder::Feeder(const FeederSetting& feeder_setting) : mFeederSetting(feeder_setting) {
    const std::string source =
        mFeederSetting.sources.empty() ? std::string() : mFeederSetting.sources.front();

    switch (mFeederSetting.feeder_type) {
        case FeederType::CAMERA:
            mCap.open(std::stoi(source), cv::CAP_V4L2);
            mCap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            mCap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            mCap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
            mCap.set(cv::CAP_PROP_FPS, 30);
            mDelayOn = false;
            break;
        case FeederType::VIDEO:
            mCap.open(source);
            mDelayOn = true;
            break;
        case FeederType::IPCAMERA:
            mCap.open(source);
            mDelayOn = false;
            break;
        case FeederType::YOUTUBE:
            mCap.open(getYoutubeStream(source));
            mDelayOn = true;
            break;
    }

    if (mFeederSetting.feeder_type == FeederType::VIDEO && mCap.isOpened()) {
        const double fps = mCap.get(cv::CAP_PROP_FPS);
        if (fps >= 1.0 && fps <= 240.0) {
            mVideoFps = fps;
        }
    }
}

bool Feeder::readFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lk(mCapMutex);
    if (!mCap.isOpened()) {
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        return true;
    }
    mCap >> frame;
    if (frame.empty() && mFeederSetting.feeder_type == FeederType::VIDEO) {
        mCap.set(cv::CAP_PROP_POS_FRAMES, 0);
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

void Feeder::stop() {
    mIsRunning.store(false, std::memory_order_relaxed);
}

void Feeder::produceFrames() {
    mFeederBuffer.open();
    while (mIsRunning.load(std::memory_order_relaxed)) {
        if (mCap.isOpened()) {
            int delay_ms = 0;
            if (mDelayOn) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO && mVideoFps >= 24.0 &&
                    mVideoFps <= 240.0) {
                    delay_ms = std::max(1, static_cast<int>(std::lround(1000.0 / mVideoFps)));
                } else {
                    delay_ms = 33;
                }
            }
            produceFramesInternal(mCap, delay_ms);
            mCap.set(cv::CAP_PROP_POS_FRAMES, 0);
        } else {
            produceFramesInternalDummy();
        }
    }
    mFeederBuffer.close();
}

void Feeder::produceFramesInternal(cv::VideoCapture& cap, int delay_ms) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty() || !mIsRunning.load(std::memory_order_relaxed)) break;
        mFeederBuffer.put(frame);

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
        for (int remaining_ms = 30; remaining_ms > 0 && mIsRunning.load(std::memory_order_relaxed);
             remaining_ms -= 5) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(remaining_ms, 5)));
        }
        benchmarker.end();
    }
}
