#include "demo/model.h"

#include <algorithm>
#include <stdexcept>

namespace {
float iouXyxy(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);

    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float denom = area_a + area_b - inter;
    if (denom <= 0.0f) return 0.0f;
    return inter / denom;
}
}  // namespace

Model::Model(const ModelSetting& model_setting, mobilint::Accelerator& acc,
             std::unique_ptr<ModelPipeline> pipeline)
    : mModelSetting(model_setting), mPipeline(std::move(pipeline)) {
    if (!mPipeline) {
        throw std::invalid_argument("Model pipeline is required.");
    }

    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;
    if (model_setting.use_core_id) {
        mc.setSingleCoreMode(model_setting.core_id);
    } else {
        mc.setSingleCoreMode(model_setting.num_core);
    }

    mModel = mobilint::Model::create(model_setting.mxq_path, mc, sc);
    mModel->launch(acc);

    if (model_setting.pipeline_type == PipelineType::YOLO11_DET ||
        model_setting.pipeline_type == PipelineType::YOLO26_DET) {
        mScoreEmaAlpha = 0.7f;
        mScoreEmaDisplayThres = model_setting.pipeline_config.conf_threshold;
        mScoreEmaMaxMissed = 8;
    } else {
        mScoreEmaAlpha = 0.5f;
        mScoreEmaDisplayThres = 0.05f;
        mScoreEmaMaxMissed = 30;
    }
}

Model::~Model() {
    if (mModel) {
        mModel->dispose();
    }
}

double Model::getLastNpuMs(int worker_index) {
    return getWorkspace(worker_index).npu_bench.getSec() * 1000.0;
}

void Model::initWorkers(const std::vector<int>& worker_indices) {
    for (int wi : worker_indices) {
        mWorkspaceByWorker[wi];
        mScoreTracksByWorker[wi];
    }
}

PipelineWorkspace& Model::getWorkspace(int worker_index) {
    return mWorkspaceByWorker[worker_index];
}

void Model::smoothDetectionScores(int worker_index, DetectionResult& result) {
    result.display_scores.assign(result.scores.size(), 0.0f);
    auto& tracks = mScoreTracksByWorker[worker_index];
    std::vector<int> track_used(tracks.size(), 0);

    for (size_t i = 0; i < result.scores.size(); ++i) {
        int best_track = -1;
        float best_iou = 0.0f;

        for (size_t j = 0; j < tracks.size(); ++j) {
            if (track_used[j]) continue;
            if (tracks[j].label != result.labels[i]) continue;

            const float iou = iouXyxy(tracks[j].box, result.boxes[i]);
            if (iou > best_iou) {
                best_iou = iou;
                best_track = static_cast<int>(j);
            }
        }

        if (best_track >= 0 && best_iou >= mScoreEmaMatchIou) {
            auto& track = tracks[best_track];
            track.score_ema =
                mScoreEmaAlpha * result.scores[i] + (1.0f - mScoreEmaAlpha) * track.score_ema;
            track.box = result.boxes[i];
            track.label = result.labels[i];
            track.missed = 0;
            track_used[best_track] = 1;
            result.display_scores[i] = track.score_ema;
        } else {
            ScoreEmaTrack new_track;
            new_track.box = result.boxes[i];
            new_track.label = result.labels[i];
            new_track.score_ema = result.scores[i];
            tracks.push_back(new_track);
            track_used.push_back(1);
            result.display_scores[i] = new_track.score_ema;
        }
    }

    for (size_t j = 0; j < tracks.size(); ++j) {
        if (!track_used[j]) tracks[j].missed++;
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [&](const ScoreEmaTrack& track) {
                                    return track.missed > mScoreEmaMaxMissed;
                                }),
                 tracks.end());
}

cv::Mat Model::inference(const cv::Mat& frame, cv::Size size, int worker_index) {
    PipelineWorkspace& workspace = getWorkspace(worker_index);
    workspace.worker_index = worker_index;
    workspace.display_size = size;

    const WorkerContext worker_context{worker_index};

    if (mDebugMode) workspace.debug_bench_preprocess.start();
    if (!mPipeline->prepareInput(frame, mModelSetting, worker_context, *mModel, workspace)) {
        printf("[ERR] prepareInput failed: worker=%d size=%dx%d\n", worker_index, size.width, size.height);
        cv::Mat fallback;
        cv::resize(frame, fallback, size);
        return fallback;
    }
    if (mDebugMode) workspace.debug_bench_preprocess.end();

    if (mDebugMode) workspace.debug_bench_infer.start();
    workspace.npu_bench.start();
    mobilint::StatusCode sc;
    auto outputs = mPipeline->run(*mModel, workspace, sc);
    workspace.npu_bench.end();
    if (!sc) {
        printf("[ERR] run failed: worker=%d, error_code=%d\n", worker_index, static_cast<int>(sc));
        cv::Mat fallback;
        cv::resize(frame, fallback, size);
        return fallback;
    }
    if (mDebugMode) workspace.debug_bench_infer.end();

    if (mDebugMode) workspace.debug_bench_postprocess.start();
    DetectionResult result = mPipeline->postprocess(outputs, frame, mModelSetting, worker_context, workspace);
    if (mDebugMode) workspace.debug_bench_postprocess.end();

    smoothDetectionScores(worker_index, result);

    if (mDebugMode) workspace.debug_bench_render.start();
    const RenderContext render_context{size, mModelSetting.pipeline_config};
    mPipeline->render(result, frame, render_context, workspace);
    if (mDebugMode) workspace.debug_bench_render.end();

    if (workspace.result_frame.empty()) {
        printf("[ERR] render produced empty frame: worker=%d\n", worker_index);
        cv::Mat fallback;
        cv::resize(frame, fallback, size);
        return fallback;
    }

    if (mDebugMode && workspace.debug_bench_preprocess.getCount() > 0 &&
        workspace.debug_bench_preprocess.getCount() % 60 == 0) {
        double pre  = workspace.debug_bench_preprocess.getAvgSec() * 1000.0;
        double npu  = workspace.debug_bench_infer.getAvgSec() * 1000.0;
        double post = workspace.debug_bench_postprocess.getAvgSec() * 1000.0;
        double draw = workspace.debug_bench_render.getAvgSec() * 1000.0;
        printf("[BENCH] w%-2d | pre %7.3f | npu %7.3f | post %7.3f | draw %7.3f | total %7.3f ms\n",
               worker_index, pre, npu, post, draw, pre + npu + post + draw);
    }

    return workspace.result_frame;
}