#include "demo/pipeline_ssd.h"

namespace {
class SsdFallbackPipeline : public ModelPipeline {
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
        return true;
    }

    std::vector<mobilint::NDArray<float>> run(mobilint::Model&, PipelineWorkspace&,
                                              mobilint::StatusCode& sc) override {
        sc = mobilint::StatusCode::OK;
        return {};
    }

    DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>&, const cv::Mat& frame,
                                const ModelSetting&, const WorkerContext&,
                                PipelineWorkspace&) override {
        DetectionResult result;
        result.coord_size = frame.size();
        return result;
    }

    void render(const DetectionResult&, const cv::Mat& frame, const RenderContext& ctx,
                PipelineWorkspace& workspace) override {
        workspace.result_frame.create(ctx.display_size.height, ctx.display_size.width, frame.type());
        cv::resize(frame, workspace.result_frame, ctx.display_size);
    }
};
}  // namespace

std::unique_ptr<ModelPipeline> createSsdPipeline() { return std::make_unique<SsdFallbackPipeline>(); }