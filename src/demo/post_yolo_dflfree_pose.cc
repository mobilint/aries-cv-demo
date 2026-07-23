#include "demo/post_yolo_dflfree_pose.h"

#include <algorithm>
#include <numeric>

namespace {
struct PoseLayerHead {
    int box_idx = -1;
    int cls_idx = -1;
    int kpt_idx = -1;
    int num_cells = 0;
    int grid_h = 0;
    int grid_w = 0;
    int stride = 0;
};

void clamp_box(std::array<float, 4>& box, int imw, int imh) {
    box[0] = std::max(0.0f, std::min(box[0], static_cast<float>(imw - 1)));
    box[1] = std::max(0.0f, std::min(box[1], static_cast<float>(imh - 1)));
    box[2] = std::max(0.0f, std::min(box[2], static_cast<float>(imw - 1)));
    box[3] = std::max(0.0f, std::min(box[3], static_cast<float>(imh - 1)));
}
}  // namespace

YOLODFLFreePosePost::YOLODFLFreePosePost(int nc, int imh, int imw, float conf_thres,
                                         float iou_thres)
    : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres) {}

uint64_t YOLODFLFreePosePost::enqueue(cv::Mat& im,
                                      std::vector<mobilint::NDArray<float>>& npu_outs,
                                      std::vector<std::array<float, 4>>& boxes,
                                      std::vector<float>& scores, std::vector<int>& labels,
                                      std::vector<std::vector<float>>& extras) {
    (void)im;
    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();
    if (npu_outs.size() < 3 || mNc <= 0) return ++mTicket;

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;

    auto append_decoded = [&](int box_idx, int cls_idx, int kpt_idx) {
        const int num_det = static_cast<int>(npu_outs[box_idx].size() / 4);
        const int num_kpts = mNextra / 3;
        for (int i = 0; i < num_det; ++i) {
            int best_label = 0;
            float best_score = -1.0f;
            for (int c = 0; c < mNc; ++c) {
                float score = npu_outs[cls_idx][i * mNc + c];
                if (!looksLikeProbability(score)) score = sigmoid(score);
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (best_score < mConfThres) continue;
            const int bb = i * 4;
            std::array<float, 4> box = {npu_outs[box_idx][bb + 0], npu_outs[box_idx][bb + 1],
                                        npu_outs[box_idx][bb + 2], npu_outs[box_idx][bb + 3]};
            if (box[2] <= box[0] || box[3] <= box[1]) {
                const float cx = box[0], cy = box[1], w = box[2], h = box[3];
                box = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
            }
            clamp_box(box, mImw, mImh);
            if (box[2] <= box[0] || box[3] <= box[1]) continue;

            std::vector<float> keypoints(mNextra);
            if (npu_outs[kpt_idx].size() == static_cast<size_t>(num_det * mNextra)) {
                for (int j = 0; j < mNextra; ++j) keypoints[j] = npu_outs[kpt_idx][i * mNextra + j];
            } else {
                for (int k = 0; k < num_kpts; ++k) {
                    const int src = (k * num_det + i) * 3;
                    keypoints[k * 3 + 0] = npu_outs[kpt_idx][src + 0];
                    keypoints[k * 3 + 1] = npu_outs[kpt_idx][src + 1];
                    keypoints[k * 3 + 2] = npu_outs[kpt_idx][src + 2];
                }
            }
            for (int j = 2; j < mNextra; j += 3) {
                if (!looksLikeProbability(keypoints[j])) keypoints[j] = sigmoid(keypoints[j]);
            }
            pred_boxes.push_back(box);
            pred_scores.push_back(best_score);
            pred_labels.push_back(best_label);
            pred_extra.push_back(std::move(keypoints));
        }
    };

    if (npu_outs.size() == 3) {
        int box_idx = -1, cls_idx = -1, kpt_idx = -1;
        for (int i = 0; i < 3 && box_idx < 0; ++i) {
            if (npu_outs[i].size() % 4 != 0) continue;
            const int cells = static_cast<int>(npu_outs[i].size() / 4);
            for (int j = 0; j < 3; ++j) {
                if (i == j || npu_outs[j].size() != static_cast<size_t>(cells * mNc)) continue;
                const int k = 3 - i - j;
                if (npu_outs[k].size() == static_cast<size_t>(cells * mNextra)) {
                    box_idx = i; cls_idx = j; kpt_idx = k; break;
                }
            }
        }
        if (box_idx >= 0) append_decoded(box_idx, cls_idx, kpt_idx);
    } else {
        std::vector<bool> used(npu_outs.size(), false);
        std::vector<PoseLayerHead> layers;
        for (int i = 0; i < static_cast<int>(npu_outs.size()); ++i) {
            if (used[i] || npu_outs[i].size() % 4 != 0) continue;
            const int cells = static_cast<int>(npu_outs[i].size() / 4);
            int cls_idx = -1, kpt_idx = -1;
            for (int j = 0; j < static_cast<int>(npu_outs.size()); ++j) {
                if (!used[j] && j != i && npu_outs[j].size() == static_cast<size_t>(cells * mNc)) { cls_idx = j; break; }
            }
            for (int j = 0; j < static_cast<int>(npu_outs.size()); ++j) {
                if (!used[j] && j != i && j != cls_idx && npu_outs[j].size() == static_cast<size_t>(cells * mNextra)) { kpt_idx = j; break; }
            }
            int gh = 0, gw = 0, stride = 0;
            if (cls_idx < 0 || kpt_idx < 0 || !inferGridAndStride(mImh, mImw, cells, gh, gw, stride)) continue;
            layers.push_back({i, cls_idx, kpt_idx, cells, gh, gw, stride});
            used[i] = used[cls_idx] = used[kpt_idx] = true;
        }
        for (const auto& layer : layers) {
            const int num_kpts = mNextra / 3;
            for (int idx = 0; idx < layer.num_cells; ++idx) {
                int best_label = 0;
                float best_score = -1.0f;
                for (int c = 0; c < mNc; ++c) {
                    const float score = sigmoid(npu_outs[layer.cls_idx][idx * mNc + c]);
                    if (score > best_score) { best_score = score; best_label = c; }
                }
                if (best_score < mConfThres) continue;
                const float ax = static_cast<float>(idx % layer.grid_w) + 0.5f;
                const float ay = static_cast<float>(idx / layer.grid_w) + 0.5f;
                const int bb = idx * 4;
                std::array<float, 4> box = {(ax - npu_outs[layer.box_idx][bb + 0]) * layer.stride,
                                            (ay - npu_outs[layer.box_idx][bb + 1]) * layer.stride,
                                            (ax + npu_outs[layer.box_idx][bb + 2]) * layer.stride,
                                            (ay + npu_outs[layer.box_idx][bb + 3]) * layer.stride};
                clamp_box(box, mImw, mImh);
                if (box[2] <= box[0] || box[3] <= box[1]) continue;
                std::vector<float> keypoints(mNextra);
                for (int k = 0; k < num_kpts; ++k) {
                    const int base = idx * mNextra + k * 3;
                    keypoints[k * 3 + 0] = (npu_outs[layer.kpt_idx][base + 0] + ax) * layer.stride;
                    keypoints[k * 3 + 1] = (npu_outs[layer.kpt_idx][base + 1] + ay) * layer.stride;
                    keypoints[k * 3 + 2] = sigmoid(npu_outs[layer.kpt_idx][base + 2]);
                }
                pred_boxes.push_back(box);
                pred_scores.push_back(best_score);
                pred_labels.push_back(best_label);
                pred_extra.push_back(std::move(keypoints));
            }
        }
    }

    std::vector<int> order(pred_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return pred_scores[a] > pred_scores[b]; });
    std::vector<bool> suppressed(pred_scores.size(), false);
    for (size_t oi = 0; oi < order.size() && boxes.size() < 300; ++oi) {
        const int idx = order[oi];
        if (suppressed[idx]) continue;
        boxes.push_back(pred_boxes[idx]);
        scores.push_back(pred_scores[idx]);
        labels.push_back(pred_labels[idx]);
        extras.push_back(pred_extra[idx]);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int other = order[oj];
            if (pred_labels[idx] == pred_labels[other] && iou(pred_boxes[idx], pred_boxes[other]) > mIouThres) suppressed[other] = true;
        }
    }
    return ++mTicket;
}