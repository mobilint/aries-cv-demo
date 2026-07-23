#include "demo/pipeline_seg.h"

#include <algorithm>
#include <cmath>

#include "demo/coco_labels.h"
#include "demo/post_yolo_dflfree_seg.h"

namespace {
const std::vector<std::array<int, 3>> kColors = {
    {56, 56, 255},  {151, 157, 255}, {31, 112, 255}, {29, 178, 255},  {49, 210, 207},
    {10, 249, 72},  {23, 204, 146},  {134, 219, 61}, {52, 147, 26},   {187, 212, 0},
    {168, 153, 44}, {255, 194, 0},   {147, 69, 52},  {255, 115, 100}, {236, 24, 0},
    {255, 56, 132}, {133, 0, 82},    {255, 56, 203}, {200, 149, 255}, {199, 55, 255}};

struct SegTL {
    cv::Mat resized;
    cv::Mat padded;
    cv::Mat rgb;
    mobilint::NDArray<uint8_t> u8;
    mobilint::NDArray<float> f32;
    size_t u8_size = 0;
    size_t f32_size = 0;
};

SegTL& tl() {
    static thread_local SegTL b;
    return b;
}

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

void unpadAndResizeMask(const cv::Mat& src_model_mask, const LetterboxParams& p,
                        cv::Size out_size, int interp, cv::Mat& dst) {
    const int content_w = std::max(1, p.dst_w - 2 * p.pad_x);
    const int content_h = std::max(1, p.dst_h - 2 * p.pad_y);
    const cv::Rect roi(std::max(0, p.pad_x), std::max(0, p.pad_y),
                       std::min(content_w, src_model_mask.cols - std::max(0, p.pad_x)),
                       std::min(content_h, src_model_mask.rows - std::max(0, p.pad_y)));
    if (roi.width <= 0 || roi.height <= 0) {
        dst.release();
        return;
    }
    cv::resize(src_model_mask(roi), dst, out_size, 0, 0, interp);
}

void renderSeg(const DetectionResult& result, const cv::Mat& frame,
               const RenderContext& ctx, PipelineWorkspace& workspace) {
    workspace.result_frame.create(ctx.display_size.height, ctx.display_size.width, frame.type());
    cv::resize(frame, workspace.result_frame, ctx.display_size);
    if (result.seg_label_mask.empty() || result.seg_score_mask.empty()) return;

    cv::Mat label_display;
    cv::Mat score_display;
    cv::resize(result.seg_label_mask, label_display, ctx.display_size, 0, 0, cv::INTER_NEAREST);
    cv::resize(result.seg_score_mask, score_display, ctx.display_size, 0, 0, cv::INTER_LINEAR);
    cv::Mat colored(ctx.display_size.height, ctx.display_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < colored.rows; ++y) {
        for (int x = 0; x < colored.cols; ++x) {
            if (score_display.at<float>(y, x) <= 0.5f) continue;
            const int cls = static_cast<int>(label_display.at<float>(y, x)) - 1;
            if (cls < 0) continue;
            const auto bgr = kColors[cls % static_cast<int>(kColors.size())];
            colored.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(bgr[0]), static_cast<uchar>(bgr[1]), static_cast<uchar>(bgr[2]));
        }
    }
    cv::addWeighted(workspace.result_frame, 0.9, colored, 0.7, 0.0, workspace.result_frame);
}

class SegDflFreePipeline : public ModelPipeline {
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
            workspace.postprocessor = std::make_unique<YOLODFLFreeSegPost>(
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
        result.display_scores = result.scores;
        auto* seg_post = dynamic_cast<YOLODFLFreeSegPost*>(workspace.postprocessor.get());
        if (seg_post) {
            unpadAndResizeMask(seg_post->getLabelMask(), workspace.letterbox, result.coord_size,
                               cv::INTER_NEAREST, result.seg_label_mask);
            unpadAndResizeMask(seg_post->getFinalMask(), workspace.letterbox, result.coord_size,
                               cv::INTER_LINEAR, result.seg_score_mask);
        }
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame, const RenderContext& ctx,
                PipelineWorkspace& workspace) override {
        renderSeg(result, frame, ctx, workspace);
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createSegPipeline() { return std::make_unique<SegDflFreePipeline>(); }