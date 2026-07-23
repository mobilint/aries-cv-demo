#include "demo/pipeline_pose.h"

#include <algorithm>
#include <cmath>

#include "demo/post_yolo_dflfree_pose.h"

namespace {
struct PoseTL {
    cv::Mat resized;
    cv::Mat padded;
    cv::Mat rgb;
    mobilint::NDArray<uint8_t> u8;
    mobilint::NDArray<float> f32;
    size_t u8_size = 0;
    size_t f32_size = 0;
};

PoseTL& tl() {
    static thread_local PoseTL b;
    return b;
}

const std::vector<std::array<int, 2>> kSkeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

const std::vector<std::array<int, 3>> kPoseLimbColor = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255},
    {255, 51, 255}, {255, 51, 255}, {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
    {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

const std::vector<std::array<int, 3>> kPoseKptColor = {
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
    {255, 128, 0},  {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
    {51, 153, 255}, {51, 153, 255}};

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

std::array<float, 4> undoLetterboxBox(const std::array<float, 4>& box, const LetterboxParams& p) {
    const float inv = p.scale > 0.0f ? 1.0f / p.scale : 1.0f;
    return {std::max(0.0f, std::min((box[0] - p.pad_x) * inv, static_cast<float>(p.src_w - 1))),
            std::max(0.0f, std::min((box[1] - p.pad_y) * inv, static_cast<float>(p.src_h - 1))),
            std::max(0.0f, std::min((box[2] - p.pad_x) * inv, static_cast<float>(p.src_w - 1))),
            std::max(0.0f, std::min((box[3] - p.pad_y) * inv, static_cast<float>(p.src_h - 1)))};
}

void undoLetterboxKeypoints(std::vector<float>& kpts, const LetterboxParams& p) {
    const float inv = p.scale > 0.0f ? 1.0f / p.scale : 1.0f;
    for (size_t i = 0; i + 2 < kpts.size(); i += 3) {
        kpts[i + 0] = std::max(0.0f, std::min((kpts[i + 0] - p.pad_x) * inv, static_cast<float>(p.src_w - 1)));
        kpts[i + 1] = std::max(0.0f, std::min((kpts[i + 1] - p.pad_y) * inv, static_cast<float>(p.src_h - 1)));
    }
}

void renderPose(const DetectionResult& result, const cv::Mat& frame,
                const RenderContext& ctx, PipelineWorkspace& workspace) {
    workspace.result_frame.create(ctx.display_size.height, ctx.display_size.width, frame.type());
    cv::resize(frame, workspace.result_frame, ctx.display_size);
    const float sx = static_cast<float>(ctx.display_size.width) / std::max(1, result.coord_size.width);
    const float sy = static_cast<float>(ctx.display_size.height) / std::max(1, result.coord_size.height);
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const float score = i < result.display_scores.size() ? result.display_scores[i] : result.scores[i];
        if (score < ctx.pipeline_config.conf_threshold) continue;
        const int label = i < result.labels.size() ? result.labels[i] : 0;
        const auto bgr = kColors[std::max(0, label) % static_cast<int>(kColors.size())];
        const cv::Scalar box_color(bgr[0], bgr[1], bgr[2]);
        int x1 = std::max(0, std::min(static_cast<int>(result.boxes[i][0] * sx), ctx.display_size.width - 1));
        int y1 = std::max(0, std::min(static_cast<int>(result.boxes[i][1] * sy), ctx.display_size.height - 1));
        int x2 = std::max(0, std::min(static_cast<int>(result.boxes[i][2] * sx), ctx.display_size.width - 1));
        int y2 = std::max(0, std::min(static_cast<int>(result.boxes[i][3] * sy), ctx.display_size.height - 1));
        if (x2 > x1 && y2 > y1) cv::rectangle(workspace.result_frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 1);
        if (i >= result.extras.size() || result.extras[i].size() < 51) continue;
        std::vector<cv::Point> pts(17);
        std::vector<float> confs(17, 0.0f);
        for (int k = 0; k < 17; ++k) {
            pts[k] = cv::Point(static_cast<int>(result.extras[i][k * 3 + 0] * sx),
                               static_cast<int>(result.extras[i][k * 3 + 1] * sy));
            confs[k] = result.extras[i][k * 3 + 2];
            if (confs[k] >= 0.4f && pts[k].x >= 0 && pts[k].y >= 0 &&
                pts[k].x < ctx.display_size.width && pts[k].y < ctx.display_size.height) {
                const auto c = kPoseKptColor[k];
                cv::circle(workspace.result_frame, pts[k], 2, cv::Scalar(c[0], c[1], c[2]), -1);
            }
        }
        for (size_t s = 0; s < kSkeleton.size(); ++s) {
            if (s == 17 || s == 18) continue;
            const int a = kSkeleton[s][0] - 1;
            const int b = kSkeleton[s][1] - 1;
            if (a < 0 || a >= 17 || b < 0 || b >= 17) continue;
            if (confs[a] < 0.5f || confs[b] < 0.5f) continue;
            const auto c = kPoseLimbColor[s];
            cv::line(workspace.result_frame, pts[a], pts[b], cv::Scalar(c[0], c[1], c[2]), 1);
        }
    }
}

class PoseDflFreePipeline : public ModelPipeline {
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
        letterboxBgr(frame, b.resized, b.padded, workspace.w, workspace.h, workspace.letterbox);
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
            cv::cvtColor(b.padded, input, cv::COLOR_BGR2RGB);
        } else {
            if (b.f32_size != input_size) {
                mobilint::StatusCode sc;
                b.f32 = mobilint::NDArray<float>({1, workspace.h, workspace.w, workspace.c}, sc);
                if (!sc) return false;
                b.f32_size = input_size;
            }
            b.rgb.create(workspace.h, workspace.w, CV_8UC3);
            cv::cvtColor(b.padded, b.rgb, cv::COLOR_BGR2RGB);
            cv::Mat input(workspace.h, workspace.w, CV_32FC3, b.f32.data());
            b.rgb.convertTo(input, CV_32FC3, 1.0f / 255.0f);
        }
        if (!workspace.postprocessor) {
            workspace.postprocessor = std::make_unique<YOLODFLFreePosePost>(
                setting.pipeline_config.num_classes, workspace.h, workspace.w,
                setting.pipeline_config.conf_threshold, setting.pipeline_config.iou_threshold);
        }
        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model, PipelineWorkspace& workspace,
                                              mobilint::StatusCode& sc) override {
        return workspace.active_input_type == InputDataType::UINT8 ? model.infer({tl().u8}, sc)
                                                                   : model.infer({tl().f32}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat&, const ModelSetting&, const WorkerContext&,
                                PipelineWorkspace& workspace) override {
        DetectionResult result;
        result.coord_size = cv::Size(workspace.letterbox.src_w, workspace.letterbox.src_h);
        auto mutable_outputs = outputs;
        const auto ticket = workspace.postprocessor->enqueue(tl().padded, mutable_outputs,
            result.boxes, result.scores, result.labels, result.extras);
        workspace.postprocessor->receive(ticket);
        for (auto& box : result.boxes) box = undoLetterboxBox(box, workspace.letterbox);
        for (auto& kpts : result.extras) undoLetterboxKeypoints(kpts, workspace.letterbox);
        result.display_scores = result.scores;
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame, const RenderContext& ctx,
                PipelineWorkspace& workspace) override {
        renderPose(result, frame, ctx, workspace);
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createPosePipeline() { return std::make_unique<PoseDflFreePipeline>(); }