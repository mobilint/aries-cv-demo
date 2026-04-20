#ifndef DEMO_INCLUDE_MODEL_H_
#define DEMO_INCLUDE_MODEL_H_

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "demo/define.h"
#include "demo/model_pipeline.h"
#include "qbruntime/model.h"

class Model {
public:
    Model() = delete;
    Model(const ModelSetting& model_setting, mobilint::Accelerator& acc,
          std::unique_ptr<ModelPipeline> pipeline);
    ~Model();

    cv::Mat inference(const cv::Mat& frame, cv::Size size, int worker_index,
                      bool draw_score_text = true);
    double getLastNpuMs(int worker_index);
    void initWorkers(const std::vector<int>& worker_indices);
    void setDebugMode(bool enabled) { mDebugMode = enabled; }

private:
    struct ScoreEmaTrack {
        std::array<float, 4> box;
        int label = -1;
        int missed = 0;
        float score_ema = 0.0f;
    };

    PipelineWorkspace& getWorkspace(int worker_index);
    void smoothDetectionScores(int worker_index, DetectionResult& result);

    ModelSetting mModelSetting;
    std::unique_ptr<mobilint::Model> mModel;
    std::unique_ptr<ModelPipeline> mPipeline;

    std::unordered_map<int, PipelineWorkspace> mWorkspaceByWorker;
    std::unordered_map<int, std::vector<ScoreEmaTrack>> mScoreTracksByWorker;
    float mScoreEmaAlpha = 0.7f;
    float mScoreEmaMatchIou = 0.3f;
    float mScoreEmaDisplayThres = 0.25f;
    int mScoreEmaMaxMissed = 8;
    bool mDebugMode = false;
};

#endif
