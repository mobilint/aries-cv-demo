#ifndef DEMO_INCLUDE_FEEDER_H_
#define DEMO_INCLUDE_FEEDER_H_

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

class Feeder {
public:
    Feeder() = delete;
    explicit Feeder(const FeederSetting& feeder_setting, int feeder_index = -1);
    ~Feeder() = default;

    bool consumeFrame(cv::Mat& frame, int64_t& frame_index);
    bool readFrame(cv::Mat& frame);
    void produceFrames();
    MatBuffer& getMatBuffer() { return mFeederBuffer; }
    void start();
    void stop();

private:
    void produceFramesInternal(cv::VideoCapture& cap, int delay_ms);
    void produceFramesInternalDummy();

    FeederSetting mFeederSetting;
    int mFeederIndex = -1;
    MatBuffer mFeederBuffer;
    std::atomic<bool> mIsRunning{true};
    std::mutex mCapMutex;
    cv::VideoCapture mCap;
    bool mDelayOn = false;
    double mVideoFps = 30.0;
    std::chrono::steady_clock::time_point mVideoClock{};
    bool mLoggedOpenFailure = false;
    bool mLoggedInitialReadFailure = false;
    bool mLoggedFirstFrame = false;
    int64_t mProducedFrameCount = 0;
};

#endif
