#ifndef DEMO_INCLUDE_MODEL_PIPELINE_H_
#define DEMO_INCLUDE_MODEL_PIPELINE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "demo/post.h"
#include "qbruntime/model.h"

struct LetterboxParams {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;
};

struct PipelineWorkspace {
    int worker_index = 0;
    cv::Size display_size;
    cv::Mat result_frame;
    LetterboxParams letterbox;
    std::unique_ptr<PostProcessor> postprocessor;
    InputDataType active_input_type = InputDataType::FLOAT32;
    std::vector<mobilint::BufferInfo> output_infos;
    int flat_cache_w = -1;
    int flat_cache_h = -1;
    std::vector<float> flat_anchor_x;
    std::vector<float> flat_anchor_y;
    std::vector<float> flat_stride_x;
    std::vector<float> flat_stride_y;
    Benchmarker npu_bench;
    int w = 0;
    int h = 0;
    int c = 0;

    Benchmarker debug_bench_preprocess;
    Benchmarker debug_bench_infer;
    Benchmarker debug_bench_postprocess;
    Benchmarker debug_bench_render;
};

struct DetectionResult {
    cv::Size coord_size;
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<float> display_scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;
};

struct WorkerContext {
    int worker_index = 0;
};

struct RenderContext {
    cv::Size display_size;
    PipelineConfig pipeline_config;
};

class ModelPipeline {
public:
    virtual ~ModelPipeline() = default;

    virtual bool prepareInput(const cv::Mat& frame, 
                              const ModelSetting& model_setting,
                              const WorkerContext& worker_context, 
                              mobilint::Model& model, 
                              PipelineWorkspace& workspace) = 0;

    virtual std::vector<mobilint::NDArray<float>> run(mobilint::Model& model,
                                                      PipelineWorkspace& workspace,
                                                      mobilint::StatusCode& sc) = 0;

    virtual DetectionResult postprocess(const std::vector<mobilint::NDArray<float>>& outputs,
                                        const cv::Mat& frame, 
                                        const ModelSetting& model_setting,
                                        const WorkerContext& worker_context, 
                                        PipelineWorkspace& workspace) = 0;

    virtual void render(const DetectionResult& result, 
                        const cv::Mat& frame,
                        const RenderContext& render_context, 
                        PipelineWorkspace& workspace) = 0;
};

std::unique_ptr<ModelPipeline> createModelPipeline(const ModelSetting& model_setting);

#endif
