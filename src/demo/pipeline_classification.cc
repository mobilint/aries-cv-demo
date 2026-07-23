#include "demo/pipeline_classification.h"

#include "demo/imagenet_labels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <string>
#include <vector>

namespace {
struct ClassificationTL {
    cv::Mat resized;
    cv::Mat crop;
    cv::Mat rgb;
    mobilint::NDArray<uint8_t> u8;
    mobilint::NDArray<float> f32;
    size_t u8_size = 0;
    size_t f32_size = 0;
};

ClassificationTL& tl() {
    static thread_local ClassificationTL b;
    return b;
}

void resizeCenterCropBgr(const cv::Mat& src, cv::Mat& resized, cv::Mat& crop, int dst_w, int dst_h) {
    if (src.empty() || dst_w <= 0 || dst_h <= 0) {
        crop = cv::Mat::zeros(std::max(1, dst_h), std::max(1, dst_w), CV_8UC3);
        return;
    }

    const int short_side = std::min(dst_w, dst_h);
    const int src_short = std::max(1, std::min(src.cols, src.rows));
    const float scale = static_cast<float>(short_side) / static_cast<float>(src_short);
    const int resized_w = std::max(dst_w, static_cast<int>(std::round(src.cols * scale)));
    const int resized_h = std::max(dst_h, static_cast<int>(std::round(src.rows * scale)));
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    const int x = std::max(0, (resized.cols - dst_w) / 2);
    const int y = std::max(0, (resized.rows - dst_h) / 2);
    resized(cv::Rect(x, y, dst_w, dst_h)).copyTo(crop);
}

std::vector<float> flattenOutput(const mobilint::NDArray<float>& output) {
    const float* data = output.data();
    return std::vector<float>(data, data + output.size());
}

std::string classificationLabelName(int label) {
    if (label >= 0 && label < static_cast<int>(demo::kImageNetLabels.size())) {
        return std::string(demo::kImageNetLabels[label]);
    }
    return "class_" + std::to_string(label);
}

void renderClassification(const DetectionResult& result, const cv::Mat& frame,
                          const RenderContext& ctx, PipelineWorkspace& workspace) {
    workspace.result_frame.create(ctx.display_size.height, ctx.display_size.width, frame.type());
    cv::resize(frame, workspace.result_frame, ctx.display_size);

    if (!ctx.pipeline_config.draw_score_text || result.labels.empty()) return;

    const int topk = std::min<int>(ctx.pipeline_config.topk, result.labels.size());
    const int line_h = 22;
    const int pad = 8;
    const int margin = 6;
    const int box_w = 360;
    const int box_h = pad * 2 + std::max(1, topk) * line_h;
    const int clamped_w = std::min(box_w, std::max(1, workspace.result_frame.cols - margin * 2));
    const int clamped_h = std::min(box_h, std::max(1, workspace.result_frame.rows - margin * 2));
    cv::Rect box(std::max(margin, workspace.result_frame.cols - clamped_w - margin),
                 std::max(margin, workspace.result_frame.rows - clamped_h - margin),
                 clamped_w, clamped_h);
    if (box.width <= 0 || box.height <= 0) return;

    cv::Mat roi = workspace.result_frame(box);
    cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
    cv::addWeighted(overlay, 0.55, roi, 0.45, 0, roi);

    for (int i = 0; i < topk; ++i) {
        const int label = result.labels[i];
        const float score = i < static_cast<int>(result.display_scores.size()) ? result.display_scores[i]
                                                                        : result.scores[i];
        const std::string label_name = classificationLabelName(label);
        char text[128];
        std::snprintf(text, sizeof(text), "%d. %s %.2f%%", i + 1, label_name.c_str(), score * 100.0f);
        cv::putText(workspace.result_frame, text, cv::Point(box.x + pad, box.y + pad + 16 + i * line_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.364, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
}

class ClassificationPipeline : public ModelPipeline {
public:
    bool prepareInput(const cv::Mat& frame, const ModelSetting& setting, const WorkerContext&,
                      mobilint::Model& model, PipelineWorkspace& workspace) override {
        if (frame.empty()) return false;
        if (workspace.w == 0) {
            const auto info = model.getInputBufferInfo()[0];
            workspace.w = info.original_width;
            workspace.h = info.original_height;
            workspace.c = info.original_channel;
        }

        auto& b = tl();
        resizeCenterCropBgr(frame, b.resized, b.crop, workspace.w, workspace.h);
        const size_t input_size = static_cast<size_t>(workspace.w) * workspace.h * workspace.c;
        workspace.active_input_type = setting.input_type;

        if (setting.input_type == InputDataType::UINT8) {
            if (b.u8_size != input_size) {
                mobilint::StatusCode sc;
                b.u8 = mobilint::NDArray<uint8_t>({1, workspace.h, workspace.w, workspace.c}, sc);
                if (!sc) return false;
                b.u8_size = input_size;
            }
            cv::Mat input(workspace.h, workspace.w, CV_8UC3, b.u8.data());
            cv::cvtColor(b.crop, input, cv::COLOR_BGR2RGB);
        } else {
            if (b.f32_size != input_size) {
                mobilint::StatusCode sc;
                b.f32 = mobilint::NDArray<float>({1, workspace.h, workspace.w, workspace.c}, sc);
                if (!sc) return false;
                b.f32_size = input_size;
            }
            b.rgb.create(workspace.h, workspace.w, CV_8UC3);
            cv::cvtColor(b.crop, b.rgb, cv::COLOR_BGR2RGB);
            cv::Mat input(workspace.h, workspace.w, CV_32FC3, b.f32.data());
            b.rgb.convertTo(input, CV_32FC3, 1.0f / 255.0f);
        }
        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model, PipelineWorkspace& workspace,
                                              mobilint::StatusCode& sc) override {
        return workspace.active_input_type == InputDataType::UINT8 ? model.infer({tl().u8}, sc)
                                                                   : model.infer({tl().f32}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat& frame, const ModelSetting& setting,
                                const WorkerContext&, PipelineWorkspace&) override {
        DetectionResult result;
        result.coord_size = frame.size();
        if (outputs.empty()) return result;

        auto probs = flattenOutput(outputs[0]);
        const int nclass = setting.pipeline_config.num_classes > 0
                               ? std::min<int>(setting.pipeline_config.num_classes, probs.size())
                               : static_cast<int>(probs.size());
        std::vector<int> order(nclass);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<int>(setting.pipeline_config.topk, nclass),
                          order.end(), [&](int a, int b) { return probs[a] > probs[b]; });

        const int keep = std::min<int>(setting.pipeline_config.topk, nclass);
        result.labels.reserve(keep);
        result.scores.reserve(keep);
        result.display_scores.reserve(keep);
        for (int i = 0; i < keep; ++i) {
            const int label = order[i];
            result.labels.push_back(label);
            result.scores.push_back(probs[label]);
            result.display_scores.push_back(probs[label]);
        }
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame, const RenderContext& ctx,
                PipelineWorkspace& workspace) override {
        renderClassification(result, frame, ctx, workspace);
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createClassificationPipeline() {
    return std::make_unique<ClassificationPipeline>();
}