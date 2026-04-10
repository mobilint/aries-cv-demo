#include "demo/model_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "demo/post_yolo11_det.h"
#include "demo/post_yolo_anchorless_passthrough.h"

namespace {
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

void letterboxBgr(const cv::Mat& src, cv::Mat& dst, cv::Mat& resized, int dst_w, int dst_h,
                  LetterboxParams& p) {
    p.src_w = src.cols;
    p.src_h = src.rows;
    p.dst_w = dst_w;
    p.dst_h = dst_h;

    if (src.empty() || src.cols <= 0 || src.rows <= 0 || dst_w <= 0 || dst_h <= 0) {
        dst = cv::Mat::zeros(std::max(1, dst_h), std::max(1, dst_w), CV_8UC3);
        p.scale = 1.0f;
        p.pad_x = 0;
        p.pad_y = 0;
        return;
    }

    float sx = static_cast<float>(dst_w) / static_cast<float>(src.cols);
    float sy = static_cast<float>(dst_h) / static_cast<float>(src.rows);
    p.scale = std::min(sx, sy);

    int new_w = std::max(1, static_cast<int>(std::round(src.cols * p.scale)));
    int new_h = std::max(1, static_cast<int>(std::round(src.rows * p.scale)));
    p.pad_x = (dst_w - new_w) / 2;
    p.pad_y = (dst_h - new_h) / 2;

    dst.create(dst_h, dst_w, CV_8UC3);
    dst.setTo(cv::Scalar(114, 114, 114));

    resized.create(new_h, new_w, CV_8UC3);
    cv::resize(src, resized, cv::Size(new_w, new_h));
    resized.copyTo(dst(cv::Rect(p.pad_x, p.pad_y, new_w, new_h)));
}

std::array<float, 4> undoLetterbox(const std::array<float, 4>& b, const LetterboxParams& p) {
    if (p.scale <= 0.0f || p.src_w <= 0 || p.src_h <= 0) return b;

    float x1 = (b[0] - static_cast<float>(p.pad_x)) / p.scale;
    float y1 = (b[1] - static_cast<float>(p.pad_y)) / p.scale;
    float x2 = (b[2] - static_cast<float>(p.pad_x)) / p.scale;
    float y2 = (b[3] - static_cast<float>(p.pad_y)) / p.scale;

    return {clampf(x1, 0.0f, static_cast<float>(p.src_w)),
            clampf(y1, 0.0f, static_cast<float>(p.src_h)),
            clampf(x2, 0.0f, static_cast<float>(p.src_w)),
            clampf(y2, 0.0f, static_cast<float>(p.src_h))};
}

struct TLBufs {
    cv::Mat resized;
    cv::Mat letterbox_tmp;
    cv::Mat rgb;
    mobilint::NDArray<float> input_f32;
    mobilint::NDArray<uint8_t> input_u8;
    size_t f32_size = 0;
    size_t u8_size = 0;
};

TLBufs& tlb() {
    static thread_local TLBufs b;
    return b;
}

void renderDetections(const DetectionResult& result, const cv::Mat& frame,
                      const RenderContext& render_context, PipelineWorkspace& workspace) {
    workspace.result_frame.create(render_context.display_size.height,
                                  render_context.display_size.width, frame.type());
    cv::resize(frame, workspace.result_frame, render_context.display_size);

    const float sx = static_cast<float>(render_context.display_size.width) /
                     static_cast<float>(std::max(1, result.coord_size.width));
    const float sy = static_cast<float>(render_context.display_size.height) /
                     static_cast<float>(std::max(1, result.coord_size.height));

    bool detected = false;
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        float score = i < result.display_scores.size() ? result.display_scores[i] : 0.0f;
        if (score < render_context.pipeline_config.conf_threshold) continue;

        int x1 = static_cast<int>(result.boxes[i][0] * sx);
        int y1 = static_cast<int>(result.boxes[i][1] * sy);
        int x2 = static_cast<int>(result.boxes[i][2] * sx);
        int y2 = static_cast<int>(result.boxes[i][3] * sy);

        x1 = std::max(0, std::min(x1, render_context.display_size.width - 1));
        y1 = std::max(0, std::min(y1, render_context.display_size.height - 1));
        x2 = std::max(0, std::min(x2, render_context.display_size.width - 1));
        y2 = std::max(0, std::min(y2, render_context.display_size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        detected = true;
        cv::Scalar clr = (i < result.labels.size() && result.labels[i] == 0)
                             ? cv::Scalar(255, 0, 255)
                             : cv::Scalar(0, 255, 255);
        cv::rectangle(workspace.result_frame, cv::Point(x1, y1), cv::Point(x2, y2), clr, 2);
        if (render_context.pipeline_config.draw_score_text) {
            char conf_text[32];
            std::snprintf(conf_text, sizeof(conf_text), "%.2f", score);
            int text_y = std::max(14, y1 - 6);
            cv::putText(workspace.result_frame, conf_text, cv::Point(x1, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv::LINE_AA);
        }
    }

    if (detected) {
        cv::rectangle(workspace.result_frame, cv::Point(0, 0),
                      cv::Point(render_context.display_size.width - 1,
                                render_context.display_size.height - 1),
                      cv::Scalar(0, 0, 255), 3);
    }
}

class Yolo11DetectionPipeline : public ModelPipeline {
   public:
    bool prepareInput(const cv::Mat& frame, const ModelSetting&, const WorkerContext&,
                      mobilint::Model& model, PipelineWorkspace& workspace) override {
        if (frame.empty()) return false;

        if (workspace.w == 0) {
            const auto info = model.getInputBufferInfo()[0];
            workspace.w = info.original_width;
            workspace.h = info.original_height;
            workspace.c = info.original_channel;
        }
        const size_t input_size = static_cast<size_t>(workspace.w) * workspace.h * workspace.c;
        auto& tl = tlb();
        if (tl.f32_size != input_size) {
            mobilint::StatusCode sc;
            tl.input_f32 =
                mobilint::NDArray<float>({1, workspace.h, workspace.w, workspace.c}, sc);
            if (!sc) return false;
            tl.f32_size = input_size;
        }

        letterboxBgr(frame, tl.resized, tl.letterbox_tmp,
                     workspace.w, workspace.h, workspace.letterbox);
        tl.rgb.create(workspace.h, workspace.w, CV_8UC3);
        cv::cvtColor(tl.resized, tl.rgb, cv::COLOR_BGR2RGB);
        cv::Mat input_mat(workspace.h, workspace.w, CV_32FC3, tl.input_f32.data());
        tl.rgb.convertTo(input_mat, CV_32FC3, 1.0f / 255.0f);

        ensurePost(model, workspace);
        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model,
                                              PipelineWorkspace& workspace,
                                              mobilint::StatusCode& sc) override {
        return model.infer({tlb().input_f32}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat& frame, const ModelSetting& model_setting,
                                const WorkerContext&, PipelineWorkspace& workspace) override {
        DetectionResult result;
        result.coord_size = frame.size();
        auto mutable_outputs = outputs;
        const uint64_t ticket =
            workspace.postprocessor->enqueue(tlb().resized, mutable_outputs, result.boxes,
                                             result.scores, result.labels, result.extras);
        workspace.postprocessor->receive(ticket);
        for (auto& box : result.boxes) {
            box = undoLetterbox(box, workspace.letterbox);
        }
        result.display_scores = result.scores;
        (void)model_setting;
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame,
                const RenderContext& render_context, PipelineWorkspace& workspace) override {
        renderDetections(result, frame, render_context, workspace);
    }

   private:
    void ensurePost(mobilint::Model&, PipelineWorkspace& workspace) {
        if (workspace.postprocessor) return;
        workspace.postprocessor = std::make_unique<YOLO11DetPostProcessor>(
            2, workspace.h, workspace.w, 0.05f, 0.45f);
    }
};

class Yolo26DetectionPipeline : public ModelPipeline {
   public:
    bool prepareInput(const cv::Mat& frame, const ModelSetting&, const WorkerContext&,
                      mobilint::Model& model, PipelineWorkspace& workspace) override {
        if (frame.empty()) return false;

        if (workspace.w == 0) {
            const auto info = model.getInputBufferInfo()[0];
            workspace.output_infos = model.getOutputBufferInfo();
            workspace.w = info.original_width;
            workspace.h = info.original_height;
            workspace.c = info.original_channel;
        }
        const size_t input_size = static_cast<size_t>(workspace.w) * workspace.h * workspace.c;
        auto& tl = tlb();
        if (tl.u8_size != input_size) {
            mobilint::StatusCode sc;
            tl.input_u8 = mobilint::NDArray<uint8_t>({1, workspace.h, workspace.w, workspace.c}, sc);
            if (!sc) return false;
            tl.u8_size = input_size;
        }

        letterboxBgr(frame, tl.resized, tl.letterbox_tmp,
                     workspace.w, workspace.h, workspace.letterbox);
        cv::Mat input_mat(workspace.h, workspace.w, CV_8UC3, tl.input_u8.data());
        cv::cvtColor(tl.resized, input_mat, cv::COLOR_BGR2RGB);

        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model,
                                              PipelineWorkspace&,
                                              mobilint::StatusCode& sc) override {
        return model.infer({tlb().input_u8}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat& frame, const ModelSetting& model_setting,
                                const WorkerContext&, PipelineWorkspace& workspace) override {
        DetectionResult result;
        result.coord_size = frame.size();

        struct OutputView {
            int idx;
            int grid_h;
            int grid_w;
            int ch;
            size_t elem_count;
        };

        std::vector<OutputView> box_outs;
        std::vector<OutputView> cls_outs;
        auto out_infos =
            workspace.output_infos.empty() ? std::vector<mobilint::BufferInfo>() : workspace.output_infos;
        if (out_infos.empty()) {
            result.display_scores = result.scores;
            return result;
        }

        size_t nout = std::min(outputs.size(), out_infos.size());
        for (size_t i = 0; i < nout; i++) {
            int ow = out_infos[i].original_width;
            int oh = out_infos[i].original_height;
            int oc = out_infos[i].original_channel;
            if (ow <= 0 || oh <= 0 || oc <= 0) continue;
            size_t expect_size = static_cast<size_t>(ow) * oh * oc;
            if (expect_size != outputs[i].size()) continue;
            OutputView out = {static_cast<int>(i), oh, ow, oc, expect_size};
            if (oc == 4) box_outs.push_back(out);
            if (oc == model_setting.pipeline_config.num_classes) cls_outs.push_back(out);
        }

        std::sort(box_outs.begin(), box_outs.end(),
                  [](const OutputView& a, const OutputView& b) {
                      return a.elem_count > b.elem_count;
                  });
        std::sort(cls_outs.begin(), cls_outs.end(),
                  [](const OutputView& a, const OutputView& b) {
                      return a.elem_count > b.elem_count;
                  });

        std::vector<std::array<float, 4>> cand_boxes;
        std::vector<float> cand_scores;
        std::vector<int> cand_labels;

        const int total_anchors = (workspace.h / 8) * (workspace.w / 8) +
                                  (workspace.h / 16) * (workspace.w / 16) +
                                  (workspace.h / 32) * (workspace.w / 32);
        ensureFlatAnchors(workspace, total_anchors);

        const float conf_thres = model_setting.pipeline_config.conf_threshold;
        const float inv_conf_thres = std::log(conf_thres / (1.0f - conf_thres));
        const size_t max_det = 300;

        const size_t npairs = std::min(box_outs.size(), cls_outs.size());
        for (size_t p = 0; p < npairs; ++p) {
            const auto& box_view = box_outs[p];
            const auto& cls_view = cls_outs[p];
            if (box_view.grid_h != cls_view.grid_h || box_view.grid_w != cls_view.grid_w) {
                continue;
            }

            const auto& box = outputs[box_view.idx];
            const auto& cls = outputs[cls_view.idx];
            const int grid_h = box_view.grid_h;
            const int grid_w = box_view.grid_w;
            const int ncell = grid_h * grid_w;
                const bool is_flat =
                ((grid_h == 1 || grid_w == 1) && ncell == total_anchors &&
                 workspace.flat_anchor_x.size() == static_cast<size_t>(total_anchors));
            const float stride_x =
                is_flat ? 0.0f : static_cast<float>(workspace.w) / static_cast<float>(grid_w);
            const float stride_y =
                is_flat ? 0.0f : static_cast<float>(workspace.h) / static_cast<float>(grid_h);

            for (int cell = 0; cell < ncell; ++cell) {
                const int cls_base = cell * model_setting.pipeline_config.num_classes;
                int label = 0;
                float best_logit = cls[cls_base];
                for (int cidx = 1; cidx < model_setting.pipeline_config.num_classes; ++cidx) {
                    if (cls[cls_base + cidx] > best_logit) {
                        best_logit = cls[cls_base + cidx];
                        label = cidx;
                    }
                }
                if (best_logit <= inv_conf_thres) continue;

                const float conf = sigmoid(best_logit);
                if (conf < conf_thres) continue;

                const size_t box_base = static_cast<size_t>(cell) * 4;
                const float l = box[box_base + 0];
                const float t = box[box_base + 1];
                const float r = box[box_base + 2];
                const float b = box[box_base + 3];

                float ax = 0.0f;
                float ay = 0.0f;
                float sx = 0.0f;
                float sy = 0.0f;
                if (is_flat) {
                    ax = workspace.flat_anchor_x[cell];
                    ay = workspace.flat_anchor_y[cell];
                    sx = workspace.flat_stride_x[cell];
                    sy = workspace.flat_stride_y[cell];
                } else {
                    const int gx = cell % grid_w;
                    const int gy = cell / grid_w;
                    ax = static_cast<float>(gx) + 0.5f;
                    ay = static_cast<float>(gy) + 0.5f;
                    sx = stride_x;
                    sy = stride_y;
                }

                float x1 = (ax - l) * sx;
                float y1 = (ay - t) * sy;
                float x2 = (ax + r) * sx;
                float y2 = (ay + b) * sy;
                x1 = clampf(x1, 0.0f, static_cast<float>(workspace.w));
                y1 = clampf(y1, 0.0f, static_cast<float>(workspace.h));
                x2 = clampf(x2, 0.0f, static_cast<float>(workspace.w));
                y2 = clampf(y2, 0.0f, static_cast<float>(workspace.h));
                if (x2 <= x1 || y2 <= y1) continue;

                cand_boxes.push_back({x1, y1, x2, y2});
                cand_scores.push_back(conf);
                cand_labels.push_back(label);
            }
        }

        std::vector<int> order(cand_scores.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return cand_scores[a] > cand_scores[b]; });

        const size_t keep_n = std::min(max_det, order.size());
        result.boxes.reserve(keep_n);
        result.scores.reserve(keep_n);
        result.labels.reserve(keep_n);
        for (size_t i = 0; i < keep_n; ++i) {
            const int idx = order[i];
            result.boxes.push_back(undoLetterbox(cand_boxes[idx], workspace.letterbox));
            result.scores.push_back(cand_scores[idx]);
            result.labels.push_back(cand_labels[idx]);
        }

        result.display_scores = result.scores;
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat& frame,
                const RenderContext& render_context, PipelineWorkspace& workspace) override {
        renderDetections(result, frame, render_context, workspace);
    }

   private:
    void ensureFlatAnchors(PipelineWorkspace& workspace, int total_anchors) {
        if (workspace.flat_cache_w == workspace.w && workspace.flat_cache_h == workspace.h &&
            workspace.flat_anchor_x.size() == static_cast<size_t>(total_anchors)) {
            return;
        }
        workspace.flat_anchor_x.clear();
        workspace.flat_anchor_y.clear();
        workspace.flat_stride_x.clear();
        workspace.flat_stride_y.clear();
        workspace.flat_anchor_x.reserve(total_anchors);
        workspace.flat_anchor_y.reserve(total_anchors);
        workspace.flat_stride_x.reserve(total_anchors);
        workspace.flat_stride_y.reserve(total_anchors);
        for (int stride : {8, 16, 32}) {
            const int gh = workspace.h / stride;
            const int gw = workspace.w / stride;
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    workspace.flat_anchor_x.push_back(static_cast<float>(gx) + 0.5f);
                    workspace.flat_anchor_y.push_back(static_cast<float>(gy) + 0.5f);
                    workspace.flat_stride_x.push_back(static_cast<float>(stride));
                    workspace.flat_stride_y.push_back(static_cast<float>(stride));
                }
            }
        }
        workspace.flat_cache_w = workspace.w;
        workspace.flat_cache_h = workspace.h;
    }
};

class YoloAnchorlessDetectionPipeline : public ModelPipeline {
   public:
    bool prepareInput(const cv::Mat& frame, const ModelSetting& model_setting,
                      const WorkerContext&, mobilint::Model& model,
                      PipelineWorkspace& workspace) override {
        if (frame.empty()) return false;

        if (workspace.w == 0) {
            const auto info = model.getInputBufferInfo()[0];
            workspace.w = info.original_width;
            workspace.h = info.original_height;
            workspace.c = info.original_channel;
        }
        const size_t input_size = static_cast<size_t>(workspace.w) * workspace.h * workspace.c;
        auto& tl = tlb();
        cv::resize(frame, tl.resized, cv::Size(workspace.w, workspace.h));
        workspace.active_input_type = model_setting.input_type;

        if (model_setting.input_type == InputDataType::FLOAT32) {
            if (tl.f32_size != input_size) {
                mobilint::StatusCode local_sc;
                tl.input_f32 = mobilint::NDArray<float>({1, workspace.h, workspace.w, workspace.c}, local_sc);
                if (!local_sc) return false;
                tl.f32_size = input_size;
            }
            tl.rgb.create(workspace.h, workspace.w, CV_8UC3);
            cv::cvtColor(tl.resized, tl.rgb, cv::COLOR_BGR2RGB);
            cv::Mat input_mat(workspace.h, workspace.w, CV_32FC3, tl.input_f32.data());
            tl.rgb.convertTo(input_mat, CV_32FC3, 1.0f / 255.0f);
        } else {
            if (tl.u8_size != input_size) {
                mobilint::StatusCode local_sc;
                tl.input_u8 = mobilint::NDArray<uint8_t>({1, workspace.h, workspace.w, workspace.c}, local_sc);
                if (!local_sc) return false;
                tl.u8_size = input_size;
            }
            cv::Mat input_mat(workspace.h, workspace.w, CV_8UC3, tl.input_u8.data());
            cv::cvtColor(tl.resized, input_mat, cv::COLOR_BGR2RGB);
        }

        ensurePost(model, model_setting, workspace);
        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model& model,
                                              PipelineWorkspace& workspace,
                                              mobilint::StatusCode& sc) override {
        if (workspace.active_input_type == InputDataType::UINT8) {
            return model.infer({tlb().input_u8}, sc);
        }
        return model.infer({tlb().input_f32}, sc);
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                const cv::Mat&, const ModelSetting&,
                                const WorkerContext&, PipelineWorkspace& workspace) override {
        DetectionResult result;
        result.coord_size = tlb().resized.size();
        auto mutable_outputs = outputs;
        const uint64_t ticket =
            workspace.postprocessor->enqueue(tlb().resized, mutable_outputs, result.boxes,
                                             result.scores, result.labels, result.extras);
        workspace.postprocessor->receive(ticket);
        result.display_scores = result.scores;
        return result;
    }

    void render(const DetectionResult& result, const cv::Mat&,
                const RenderContext& render_context, PipelineWorkspace& workspace) override {
        renderDetections(result, tlb().resized, render_context, workspace);
    }

   private:
    void ensurePost(mobilint::Model&, const ModelSetting& model_setting,
                    PipelineWorkspace& workspace) {
        if (workspace.postprocessor) return;
        workspace.postprocessor = std::make_unique<YOLOAnchorlessPassthroughPost>(
            model_setting.pipeline_config.num_classes, workspace.h,
            workspace.w, 0.05f, model_setting.pipeline_config.iou_threshold,
            model_setting.pipeline_config.decode_bbox);
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createModelPipeline(const ModelSetting& model_setting) {
    switch (model_setting.pipeline_type) {
        case PipelineType::YOLO11_DET:
            return std::make_unique<Yolo11DetectionPipeline>();
        case PipelineType::YOLO26_DET:
            return std::make_unique<Yolo26DetectionPipeline>();
        case PipelineType::YOLO_ANCHORLESS_DET:
            return std::make_unique<YoloAnchorlessDetectionPipeline>();
    }
    throw std::invalid_argument("Unsupported pipeline type.");
}
