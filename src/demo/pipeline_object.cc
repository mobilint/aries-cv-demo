#include "demo/pipeline_object.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "demo/coco_labels.h"
#include "demo/post_yolo_dflfree.h"

namespace {
const std::vector<std::array<int, 3>> kColors = {
    {56, 56, 255},  {151, 157, 255}, {31, 112, 255}, {29, 178, 255},  {49, 210, 207},
    {10, 249, 72},  {23, 204, 146},  {134, 219, 61}, {52, 147, 26},   {187, 212, 0},
    {168, 153, 44}, {255, 194, 0},   {147, 69, 52},  {255, 115, 100}, {236, 24, 0},
    {255, 56, 132}, {133, 0, 82},    {255, 56, 203}, {200, 149, 255}, {199, 55, 255}};

void letterboxBgr(const cv::Mat& src, cv::Mat& resized, cv::Mat& padded,
                  int dst_w, int dst_h, LetterboxParams& p) {
    p.src_w = src.cols;
    p.src_h = src.rows;
    p.dst_w = dst_w;
    p.dst_h = dst_h;
    p.scale = std::min(static_cast<float>(dst_w) / std::max(1, src.cols),
                       static_cast<float>(dst_h) / std::max(1, src.rows));
    const int rw = std::max(1, static_cast<int>(std::round(src.cols * p.scale)));
    const int rh = std::max(1, static_cast<int>(std::round(src.rows * p.scale)));
    p.pad_x = (dst_w - rw) / 2;
    p.pad_y = (dst_h - rh) / 2;
    cv::resize(src, resized, cv::Size(rw, rh));
    padded = cv::Mat(dst_h, dst_w, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(p.pad_x, p.pad_y, rw, rh)));
}

std::array<float, 4> undoLetterboxBox(const std::array<float, 4>& box,
                                      const LetterboxParams& p) {
    const float inv = p.scale > 0.0f ? 1.0f / p.scale : 1.0f;
    return {std::max(0.0f, std::min((box[0] - p.pad_x) * inv, static_cast<float>(p.src_w - 1))),
            std::max(0.0f, std::min((box[1] - p.pad_y) * inv, static_cast<float>(p.src_h - 1))),
            std::max(0.0f, std::min((box[2] - p.pad_x) * inv, static_cast<float>(p.src_w - 1))),
            std::max(0.0f, std::min((box[3] - p.pad_y) * inv, static_cast<float>(p.src_h - 1)))};
}

std::string resolveLabelName(int label, const PipelineConfig& cfg) {
    if (label >= 0 && label < static_cast<int>(cfg.labels.size())) return cfg.labels[label];
    if (label >= 0 && label < static_cast<int>(demo::kCocoLabels.size()))
        return std::string(demo::kCocoLabels[label]);
    return "class_" + std::to_string(label);
}

cv::Scalar resolveLabelColor(int label, const std::string& label_name, const PipelineConfig& cfg) {
    const auto it = cfg.label_colors.find(label_name);
    if (it != cfg.label_colors.end()) {
        return cv::Scalar(it->second[0], it->second[1], it->second[2]);
    }
    const auto bgr = kColors[std::max(0, label) % static_cast<int>(kColors.size())];
    return cv::Scalar(bgr[0], bgr[1], bgr[2]);
}

void renderBoxes(const DetectionResult& result, const cv::Mat& frame,
                 const RenderContext& ctx, PipelineWorkspace& workspace) {
    workspace.result_frame.create(ctx.display_size.height, ctx.display_size.width, frame.type());
    cv::resize(frame, workspace.result_frame, ctx.display_size);
    const float sx = static_cast<float>(ctx.display_size.width) / std::max(1, result.coord_size.width);
    const float sy = static_cast<float>(ctx.display_size.height) / std::max(1, result.coord_size.height);
    const auto& display_labels = ctx.pipeline_config.display_labels;
    bool detected = false;
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const float score = i < result.display_scores.size() ? result.display_scores[i] : result.scores[i];
        if (score < ctx.pipeline_config.conf_threshold) continue;
        const int label = i < result.labels.size() ? result.labels[i] : -1;
        const std::string label_name = resolveLabelName(label, ctx.pipeline_config);
        if (!display_labels.empty() &&
            std::find(display_labels.begin(), display_labels.end(), label_name) == display_labels.end()) {
            continue;
        }
        int x1 = std::max(0, std::min(static_cast<int>(result.boxes[i][0] * sx), ctx.display_size.width - 1));
        int y1 = std::max(0, std::min(static_cast<int>(result.boxes[i][1] * sy), ctx.display_size.height - 1));
        int x2 = std::max(0, std::min(static_cast<int>(result.boxes[i][2] * sx), ctx.display_size.width - 1));
        int y2 = std::max(0, std::min(static_cast<int>(result.boxes[i][3] * sy), ctx.display_size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        detected = true;
        const cv::Scalar color = resolveLabelColor(label, label_name, ctx.pipeline_config);
        cv::rectangle(workspace.result_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                      color, ctx.pipeline_config.bbox_thickness);
        if (ctx.pipeline_config.draw_score_text) {
            std::string text;
            if (ctx.pipeline_config.draw_label_text) {
                text = label_name + " ";
            }
            text += std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(workspace.result_frame, text,
                        cv::Point(x1, std::max(14, y1 - 5)), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv::LINE_AA);
        }
    }
    if (detected && ctx.pipeline_config.draw_detection_border) {
        cv::rectangle(workspace.result_frame, cv::Point(0, 0),
                      cv::Point(ctx.display_size.width - 1, ctx.display_size.height - 1),
                      cv::Scalar(0, 0, 255), 3);
    }
}

class ObjectDflFreePipeline : public ModelPipeline {
public:
    bool prepareInput(const cv::Mat& frame, const ModelSetting& setting, const WorkerContext&,
                      mobilint::Model& model, PipelineWorkspace& workspace) override {
        return fillInput(frame, setting, model, workspace);
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model, PipelineWorkspace& workspace,
                                              mobilint::StatusCode& sc) override {
        return workspace.active_input_type == InputDataType::UINT8
                   ? model.infer({workspace.ws_u8}, sc)
                   : model.infer({workspace.ws_f32}, sc);
    }

    bool supportsAsync() const override { return true; }

    mobilint::Future<float> submitAsync(const cv::Mat& frame, const ModelSetting& setting,
                                        const WorkerContext&, mobilint::Model& model,
                                        PipelineWorkspace& workspace,
                                        mobilint::StatusCode& sc) override {
        if (!fillInput(frame, setting, model, workspace)) {
            sc = mobilint::StatusCode::Model_PredictError;
            return mobilint::Future<float>();
        }
        return workspace.active_input_type == InputDataType::UINT8
                   ? model.inferAsync({workspace.ws_u8}, sc)
                   : model.inferAsync({workspace.ws_f32}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat&, const ModelSetting&, const WorkerContext&,
                                PipelineWorkspace& workspace) override {
        DetectionResult result;
        result.coord_size = cv::Size(workspace.letterbox.src_w, workspace.letterbox.src_h);
        auto mutable_outputs = outputs;
        const auto ticket = workspace.postprocessor->enqueue(workspace.ws_padded, mutable_outputs,
            result.boxes, result.scores, result.labels, result.extras);
        workspace.postprocessor->receive(ticket);
        for (auto& box : result.boxes) box = undoLetterboxBox(box, workspace.letterbox);
        result.display_scores = result.scores;
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame, const RenderContext& ctx,
                PipelineWorkspace& workspace) override {
        renderBoxes(result, frame, ctx, workspace);
    }

private:
    bool fillInput(const cv::Mat& frame, const ModelSetting& setting, mobilint::Model& model,
                   PipelineWorkspace& workspace) {
        if (frame.empty()) return false;
        if (workspace.w == 0) {
            const auto info = model.getInputBufferInfo()[0];
            workspace.w = info.original_width;
            workspace.h = info.original_height;
            workspace.c = info.original_channel;
        }
        letterboxBgr(frame, workspace.ws_resized, workspace.ws_padded, workspace.w, workspace.h,
                     workspace.letterbox);
        const size_t input_size = static_cast<size_t>(workspace.w) * workspace.h * workspace.c;
        workspace.active_input_type = setting.input_type;
        if (setting.input_type == InputDataType::UINT8) {
            if (workspace.ws_u8_size != input_size) {
                mobilint::StatusCode sc;
                workspace.ws_u8 =
                    mobilint::NDArray<uint8_t>({1, workspace.h, workspace.w, workspace.c}, sc);
                if (!sc) return false;
                workspace.ws_u8_size = input_size;
            }
            cv::Mat input(workspace.h, workspace.w, CV_8UC3, workspace.ws_u8.data());
            cv::cvtColor(workspace.ws_padded, input, cv::COLOR_BGR2RGB);
        } else {
            if (workspace.ws_f32_size != input_size) {
                mobilint::StatusCode sc;
                workspace.ws_f32 =
                    mobilint::NDArray<float>({1, workspace.h, workspace.w, workspace.c}, sc);
                if (!sc) return false;
                workspace.ws_f32_size = input_size;
            }
            workspace.ws_rgb.create(workspace.h, workspace.w, CV_8UC3);
            cv::cvtColor(workspace.ws_padded, workspace.ws_rgb, cv::COLOR_BGR2RGB);
            cv::Mat input(workspace.h, workspace.w, CV_32FC3, workspace.ws_f32.data());
            workspace.ws_rgb.convertTo(input, CV_32FC3, 1.0f / 255.0f);
        }
        if (!workspace.postprocessor) {
            workspace.postprocessor = std::make_unique<YOLODFLFreePost>(
                setting.pipeline_config.num_classes, workspace.h, workspace.w,
                setting.pipeline_config.conf_threshold, setting.pipeline_config.iou_threshold);
        }
        return true;
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createObjectDflFreePipeline() {
    return std::make_unique<ObjectDflFreePipeline>();
}