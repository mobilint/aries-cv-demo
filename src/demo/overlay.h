#ifndef DEMO_INCLUDE_OVERLAY_H_
#define DEMO_INCLUDE_OVERLAY_H_

#include <memory>
#include <string>

#include "demo/define.h"

enum class PerformanceDisplayMode {
    TILE_FPS = 0,
    AVG_FPS_ONLY,
    OFF,
};

class OverlayRenderer {
public:
    virtual ~OverlayRenderer() = default;
    virtual void renderFrameMetrics(Item& item) = 0;
    virtual void renderDisplayMetrics(cv::Mat& display, const cv::Mat& display_base,
                                      PerformanceDisplayMode perf_mode, bool show_time,
                                      float avg_fps, float avg_npu_ms,
                                      float elapsed_sec) = 0;
};

std::unique_ptr<OverlayRenderer> createOverlayRenderer(const std::string& overlay_style);

#endif
