#include "demo/overlay.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace {
std::string secToString(int sec) {
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
    return std::string(buf);
}

struct OverlayPreset {
    bool frame_metrics = false;
    bool avg_fps = false;
    cv::Rect fps_box;
    cv::Rect time_box;
    cv::Point time_label;
    cv::Point time_value;
    cv::Point fps_label;
    cv::Point fps_value;
    double display_font_scale = 0.8;
    double time_font_scale = 0.9;
};

const OverlayPreset& getPreset(const std::string& name) {
    static const std::map<std::string, OverlayPreset> kPresets = {
        {"tile_compact", {true, false, {}, {1380, 46, 260, 44}, {1392, 76}, {1474, 76},
                          {}, {}, 0.8, 0.9}},
        {"summary_banner", {false, true, {1400, 50, 320, 40}, {480, 324, 160, 32},
                            {488, 346}, {548, 346}, {1408, 76}, {1518, 76}, 0.8, 0.55}},
    };
    return kPresets.at(name);
}

class PresetOverlayRenderer : public OverlayRenderer {
public:
    explicit PresetOverlayRenderer(const OverlayPreset& preset) : mPreset(preset) {}

    void renderFrameMetrics(Item& item) override {
        if (!mPreset.frame_metrics) return;

        const cv::Rect box(5, 5, 110, 40);
        cv::Mat roi = item.img(box);
        cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
        cv::addWeighted(overlay, 0.60, roi, 0.40, 0, roi);

        char fps_val[24];
        char npu_val[24];
        std::snprintf(fps_val, sizeof(fps_val), "%.2f", item.fps);
        std::snprintf(npu_val, sizeof(npu_val), "%.1fms", item.time);

        const double font_scale = 0.5;
        const int thickness = 1;
        int baseline = 0;
        int label_w = std::max(
            cv::getTextSize("FPS", cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseline)
                .width,
            cv::getTextSize("NPU", cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseline)
                .width);
        const int lx = box.x + 4;
        const int rx = lx + label_w + 8;

        cv::putText(item.img, "FPS", cv::Point(lx, 21), cv::FONT_HERSHEY_DUPLEX, font_scale,
                    cv::Scalar(230, 230, 230), thickness, cv::LINE_AA);
        cv::putText(item.img, fps_val, cv::Point(rx, 21), cv::FONT_HERSHEY_DUPLEX, font_scale,
                    cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
        cv::putText(item.img, "NPU", cv::Point(lx, 37), cv::FONT_HERSHEY_DUPLEX, font_scale,
                    cv::Scalar(230, 230, 230), thickness, cv::LINE_AA);
        cv::putText(item.img, npu_val, cv::Point(rx, 37), cv::FONT_HERSHEY_DUPLEX, font_scale,
                    cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    }

    void renderDisplayMetrics(cv::Mat& display, const cv::Mat& display_base, bool show_fps,
                              bool show_time, float avg_fps, float elapsed_sec) override {
        if (mPreset.avg_fps && inside(display, mPreset.fps_box)) {
            display_base(mPreset.fps_box).copyTo(display(mPreset.fps_box));
            if (show_fps) {
                cv::Mat roi = display(mPreset.fps_box);
                cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
                cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);
                char buf[20];
                std::snprintf(buf, sizeof(buf), "%8.2f", avg_fps);
                cv::putText(display, "AVG FPS", mPreset.fps_label, cv::FONT_HERSHEY_DUPLEX,
                            mPreset.display_font_scale, cv::Scalar(255, 255, 255), 1);
                cv::putText(display, buf, mPreset.fps_value, cv::FONT_HERSHEY_DUPLEX,
                            mPreset.display_font_scale, cv::Scalar(0, 255, 0), 1);
            }
        }

        if (inside(display, mPreset.time_box)) {
            display_base(mPreset.time_box).copyTo(display(mPreset.time_box));
            if (!show_time) return;

            cv::Mat roi = display(mPreset.time_box);
            cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
            cv::addWeighted(overlay, 0.45, roi, 0.55, 0, roi);
            cv::putText(display, "Time", mPreset.time_label, cv::FONT_HERSHEY_DUPLEX,
                        mPreset.time_font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(display, secToString(static_cast<int>(elapsed_sec)), mPreset.time_value,
                        cv::FONT_HERSHEY_DUPLEX, mPreset.time_font_scale,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }

private:
    static bool inside(const cv::Mat& display, const cv::Rect& rect) {
        return rect.width > 0 && rect.height > 0 && rect.x >= 0 && rect.y >= 0 &&
               rect.x + rect.width <= display.cols && rect.y + rect.height <= display.rows;
    }

    OverlayPreset mPreset;
};
}  // namespace

std::unique_ptr<OverlayRenderer> createOverlayRenderer(const std::string& overlay_style) {
    return std::make_unique<PresetOverlayRenderer>(getPreset(overlay_style));
}
