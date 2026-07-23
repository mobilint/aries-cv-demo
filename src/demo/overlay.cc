#include "demo/overlay.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace {
std::string secToString(int sec) {
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
    return std::string(buf);
}

struct MetricTextLayout {
    cv::Point label_origin;
    cv::Point value_origin;
};

struct MetricLineBox {
    cv::Rect rect;
    MetricTextLayout layout;
};

struct FrameMetricsLayout {
    bool enabled = false;
    cv::Rect box;
    MetricTextLayout fps_line;
    MetricTextLayout npu_line;
    double font_scale = 0.5;
    int thickness = 1;
    double dim_alpha = 0.60;
};

struct AvgMetricsLayout {
    bool enabled = false;
    cv::Rect box;
    MetricTextLayout fps_line;
    MetricTextLayout npu_line;
    double font_scale = 0.8;
    int thickness = 1;
    double dim_alpha = 0.5;
};

struct TimeMetricsLayout {
    bool enabled = false;
    cv::Rect box;
    MetricTextLayout line;
    double font_scale = 0.9;
    int thickness = 1;
    double dim_alpha = 0.45;
};

struct OverlaySpec {
    FrameMetricsLayout frame_metrics;
    AvgMetricsLayout avg_metrics;
    TimeMetricsLayout time_metrics;
};

bool inside(const cv::Mat& display, const cv::Rect& rect) {
    return rect.width > 0 && rect.height > 0 && rect.x >= 0 && rect.y >= 0 &&
           rect.x + rect.width <= display.cols && rect.y + rect.height <= display.rows;
}

void dimRect(cv::Mat& image, const cv::Rect& rect, double overlay_alpha) {
    if (!inside(image, rect)) return;
    cv::Mat roi = image(rect);
    cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
    cv::addWeighted(overlay, overlay_alpha, roi, 1.0 - overlay_alpha, 0, roi);
}

MetricLineBox makeMetricLineBox(const char* label, const char* value, cv::Point anchor,
                                double font_scale, int thickness, int label_value_gap,
                                int padding_x, int padding_y) {
    int label_baseline = 0;
    int value_baseline = 0;
    const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, font_scale,
                                                thickness, &label_baseline);
    const cv::Size value_size = cv::getTextSize(value, cv::FONT_HERSHEY_DUPLEX, font_scale,
                                                thickness, &value_baseline);
    const int text_height = std::max(label_size.height, value_size.height);
    const int baseline = std::max(label_baseline, value_baseline);
    const int text_width = label_size.width + label_value_gap + value_size.width;

    MetricLineBox box;
    box.rect = cv::Rect(anchor.x, anchor.y, text_width + padding_x * 2,
                        text_height + baseline + padding_y * 2);

    const int text_baseline_y = box.rect.y + padding_y + text_height;
    box.layout.label_origin = cv::Point(box.rect.x + padding_x, text_baseline_y);
    box.layout.value_origin = cv::Point(box.layout.label_origin.x + label_size.width + label_value_gap,
                                        text_baseline_y);
    return box;
}

void drawMetricLine(cv::Mat& image, const MetricTextLayout& layout, const char* label,
                    const char* value, double font_scale, int thickness) {
    cv::putText(image, label, layout.label_origin, cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(230, 230, 230), thickness, cv::LINE_AA);
    cv::putText(image, value, layout.value_origin, cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
}

OverlaySpec makeTileCompactSpec() {
    OverlaySpec spec;
    spec.frame_metrics.enabled = true;
    spec.frame_metrics.box = cv::Rect(5, 5, 110, 40);
    spec.frame_metrics.fps_line = {cv::Point(9, 21), cv::Point(44, 21)};
    spec.frame_metrics.npu_line = {cv::Point(9, 37), cv::Point(44, 37)};

    spec.avg_metrics.enabled = true;
    spec.avg_metrics.box = cv::Rect(1320, 33, 330, 80);
    spec.avg_metrics.fps_line = {cv::Point(1334, 53), cv::Point(1410, 53)};
    spec.avg_metrics.npu_line = {cv::Point(1334, 91), cv::Point(1410, 91)};
    spec.avg_metrics.font_scale = 0.72;

    spec.time_metrics.enabled = true;
    spec.time_metrics.box = cv::Rect(1080, 48, 220, 50);
    spec.time_metrics.line = {cv::Point(1094, 50), cv::Point(1178, 50)};
    spec.time_metrics.font_scale = 0.72;
    return spec;
}

OverlaySpec resolveOverlaySpec(const std::string& name) {
    if (name == "tile_compact") return makeTileCompactSpec();
    throw std::invalid_argument("Unknown overlay_style: " + name);
}

class PresetOverlayRenderer : public OverlayRenderer {
public:
    explicit PresetOverlayRenderer(OverlaySpec spec) : mSpec(std::move(spec)) {}

    void renderFrameMetrics(Item& item) override {
        if (!mSpec.frame_metrics.enabled) return;
        if (!inside(item.img, mSpec.frame_metrics.box)) return;

        dimRect(item.img, mSpec.frame_metrics.box, mSpec.frame_metrics.dim_alpha);

        char fps_val[24];
        char npu_val[24];
        std::snprintf(fps_val, sizeof(fps_val), "%.2f", item.fps);
        std::snprintf(npu_val, sizeof(npu_val), "%.1f", item.time);

        drawMetricLine(item.img, mSpec.frame_metrics.fps_line, "FPS", fps_val,
                       mSpec.frame_metrics.font_scale, mSpec.frame_metrics.thickness);
        drawMetricLine(item.img, mSpec.frame_metrics.npu_line, "NPU", npu_val,
                       mSpec.frame_metrics.font_scale, mSpec.frame_metrics.thickness);
    }

    void renderDisplayMetrics(cv::Mat& display, const cv::Mat& display_base,
                              PerformanceDisplayMode perf_mode, bool show_time,
                              float avg_fps, float avg_npu_fps,
                              float elapsed_sec) override {
        if (mSpec.avg_metrics.enabled && inside(display, mSpec.avg_metrics.box)) {
            display_base(mSpec.avg_metrics.box).copyTo(display(mSpec.avg_metrics.box));
            if (perf_mode == PerformanceDisplayMode::AVG_FPS_ONLY) {
                char fps_buf[24];
                char npu_buf[24];
                std::snprintf(fps_buf, sizeof(fps_buf), "%5.1f", avg_fps);
                std::snprintf(npu_buf, sizeof(npu_buf), "%5.1f", avg_npu_fps);

                const int label_value_gap = 20;
                const int padding_x = 14;
                const int padding_y = 5;
                const int line_gap = 2;
                const MetricLineBox fps_box = makeMetricLineBox(
                    "FPS", fps_buf, mSpec.avg_metrics.box.tl(), mSpec.avg_metrics.font_scale,
                    mSpec.avg_metrics.thickness, label_value_gap, padding_x, padding_y);
                const MetricLineBox npu_box = makeMetricLineBox(
                    "NPU", npu_buf, cv::Point(mSpec.avg_metrics.box.x,
                                              fps_box.rect.y + fps_box.rect.height + line_gap),
                    mSpec.avg_metrics.font_scale, mSpec.avg_metrics.thickness, label_value_gap,
                    padding_x, padding_y);
                const cv::Rect compact_box = fps_box.rect | npu_box.rect;

                if (inside(display, compact_box)) {
                    dimRect(display, compact_box, mSpec.avg_metrics.dim_alpha);
                    drawMetricLine(display, fps_box.layout, "FPS", fps_buf,
                                   mSpec.avg_metrics.font_scale, mSpec.avg_metrics.thickness);
                    drawMetricLine(display, npu_box.layout, "NPU", npu_buf,
                                   mSpec.avg_metrics.font_scale, mSpec.avg_metrics.thickness);
                }
            }
        }

        if (mSpec.time_metrics.enabled && inside(display, mSpec.time_metrics.box)) {
            display_base(mSpec.time_metrics.box).copyTo(display(mSpec.time_metrics.box));
            if (!show_time) return;

            const std::string time_text = secToString(static_cast<int>(elapsed_sec));
            const MetricLineBox time_box = makeMetricLineBox(
                "Time", time_text.c_str(), mSpec.time_metrics.box.tl(),
                mSpec.time_metrics.font_scale, mSpec.time_metrics.thickness, 12, 8, 5);
            dimRect(display, time_box.rect, mSpec.time_metrics.dim_alpha);
            drawMetricLine(display, time_box.layout, "Time", time_text.c_str(),
                           mSpec.time_metrics.font_scale, mSpec.time_metrics.thickness);
        }
    }

private:
    OverlaySpec mSpec;
};
}  // namespace

std::unique_ptr<OverlayRenderer> createOverlayRenderer(const std::string& overlay_style) {
    return std::make_unique<PresetOverlayRenderer>(resolveOverlaySpec(overlay_style));
}
