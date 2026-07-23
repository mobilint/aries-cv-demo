#include "demo/post_yolo_dflfree.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

struct DflFreeLayerHead {
    int box_idx = -1;
    int cls_idx = -1;
    int num_cells = 0;
    int grid_h = 0;
    int grid_w = 0;
    int stride = 0;
};

bool infer_grid_and_stride(int imh, int imw, int num_cells, int& grid_h, int& grid_w,
                           int& stride) {
    if (imh <= 0 || imw <= 0 || num_cells <= 0) return false;

    const int max_stride = std::max(imh, imw);
    for (int s = 1; s <= max_stride; ++s) {
        if (imh % s != 0 || imw % s != 0) continue;
        const int gh = imh / s;
        const int gw = imw / s;
        if (gh * gw == num_cells) {
            grid_h = gh;
            grid_w = gw;
            stride = s;
            return true;
        }
    }
    return false;
}

float clamp_coord(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

bool looks_like_probability(float v) { return v >= 0.0f && v <= 1.0f; }

}  // namespace

YOLODFLFreePost::YOLODFLFreePost(int nc, int imh, int imw, float conf_thres,
                                 float iou_thres)
    : mNc(nc), mImh(imh), mImw(imw), mConfThres(conf_thres), mIouThres(iou_thres) {}

float YOLODFLFreePost::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

bool YOLODFLFreePost::inferGridAndStride(int imh, int imw, int num_cells, int& grid_h,
                                         int& grid_w, int& stride) {
    return infer_grid_and_stride(imh, imw, num_cells, grid_h, grid_w, stride);
}

float YOLODFLFreePost::clampCoord(float v, float lo, float hi) { return clamp_coord(v, lo, hi); }

bool YOLODFLFreePost::looksLikeProbability(float v) { return looks_like_probability(v); }

float YOLODFLFreePost::iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float aa = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float ba = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float denom = aa + ba - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

uint64_t YOLODFLFreePost::enqueue(cv::Mat& im,
                                  std::vector<mobilint::NDArray<float>>& npu_outs,
                                  std::vector<std::array<float, 4>>& boxes,
                                  std::vector<float>& scores, std::vector<int>& labels,
                                  std::vector<std::vector<float>>& extras) {
    (void)im;
    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();
    if (npu_outs.size() < 2 || mNc <= 0) return ++mTicket;

    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;

    auto append_decoded_pair = [&](int box_idx, int cls_idx) {
        const auto& box_out = npu_outs[box_idx];
        const auto& cls_out = npu_outs[cls_idx];
        const int cells = static_cast<int>(box_out.size() / 4);
        for (int i = 0; i < cells; ++i) {
            const int cls_base = i * mNc;
            int best_label = 0;
            float best_score = -1.0f;
            for (int c = 0; c < mNc; ++c) {
                float score = cls_out[cls_base + c];
                if (!looks_like_probability(score)) score = sigmoid(score);
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (best_score < mConfThres) continue;

            const int box_base = i * 4;
            float x1 = box_out[box_base + 0];
            float y1 = box_out[box_base + 1];
            float x2 = box_out[box_base + 2];
            float y2 = box_out[box_base + 3];
            if (!(x2 > x1 && y2 > y1)) {
                const float cx = x1;
                const float cy = y1;
                const float w = std::max(0.0f, x2);
                const float h = std::max(0.0f, y2);
                x1 = cx - w * 0.5f;
                y1 = cy - h * 0.5f;
                x2 = cx + w * 0.5f;
                y2 = cy + h * 0.5f;
            }
            x1 = clamp_coord(x1, 0.0f, static_cast<float>(mImw - 1));
            y1 = clamp_coord(y1, 0.0f, static_cast<float>(mImh - 1));
            x2 = clamp_coord(x2, 0.0f, static_cast<float>(mImw - 1));
            y2 = clamp_coord(y2, 0.0f, static_cast<float>(mImh - 1));
            if (x2 <= x1 || y2 <= y1) continue;
            cand_boxes.push_back({x1, y1, x2, y2});
            cand_scores.push_back(best_score);
            cand_labels.push_back(best_label);
        }
    };

    auto append_grid_relative_layer = [&](const DflFreeLayerHead& layer) {
        const auto& box_out = npu_outs[layer.box_idx];
        const auto& cls_out = npu_outs[layer.cls_idx];
        for (int i = 0; i < layer.num_cells; ++i) {
            const int cls_base = i * mNc;
            int best_label = 0;
            float best_score = -1.0f;
            for (int c = 0; c < mNc; ++c) {
                float score = cls_out[cls_base + c];
                if (!looks_like_probability(score)) score = sigmoid(score);
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (best_score < mConfThres) continue;

            const int grid_y = i / layer.grid_w;
            const int grid_x = i % layer.grid_w;
            const float anchor_x = static_cast<float>(grid_x) + 0.5f;
            const float anchor_y = static_cast<float>(grid_y) + 0.5f;
            const int box_base = i * 4;
            const float left = box_out[box_base + 0];
            const float top = box_out[box_base + 1];
            const float right = box_out[box_base + 2];
            const float bottom = box_out[box_base + 3];
            float x1 = (anchor_x - left) * static_cast<float>(layer.stride);
            float y1 = (anchor_y - top) * static_cast<float>(layer.stride);
            float x2 = (anchor_x + right) * static_cast<float>(layer.stride);
            float y2 = (anchor_y + bottom) * static_cast<float>(layer.stride);

            x1 = clamp_coord(x1, 0.0f, static_cast<float>(mImw - 1));
            y1 = clamp_coord(y1, 0.0f, static_cast<float>(mImh - 1));
            x2 = clamp_coord(x2, 0.0f, static_cast<float>(mImw - 1));
            y2 = clamp_coord(y2, 0.0f, static_cast<float>(mImh - 1));
            if (x2 <= x1 || y2 <= y1) continue;
            cand_boxes.push_back({x1, y1, x2, y2});
            cand_scores.push_back(best_score);
            cand_labels.push_back(best_label);
        }
    };

    if (npu_outs.size() == 2) {
        int box_idx = -1;
        int cls_idx = -1;
        for (int i = 0; i < static_cast<int>(npu_outs.size()); ++i) {
            if (npu_outs[i].size() % 4 != 0) continue;
            const size_t cells = npu_outs[i].size() / 4;
            const int j = 1 - i;
            if (npu_outs[j].size() == cells * static_cast<size_t>(mNc)) {
                box_idx = i;
                cls_idx = j;
                break;
            }
        }
        if (box_idx < 0) return ++mTicket;
        append_decoded_pair(box_idx, cls_idx);
    } else {
        std::vector<bool> used(npu_outs.size(), false);
        std::vector<DflFreeLayerHead> layers;
        for (int i = 0; i < static_cast<int>(npu_outs.size()); ++i) {
            if (used[i] || npu_outs[i].size() % 4 != 0) continue;
            const int cells = static_cast<int>(npu_outs[i].size() / 4);
            int grid_h = 0;
            int grid_w = 0;
            int stride = 0;
            if (!infer_grid_and_stride(mImh, mImw, cells, grid_h, grid_w, stride)) continue;

            for (int j = 0; j < static_cast<int>(npu_outs.size()); ++j) {
                if (i == j || used[j]) continue;
                if (npu_outs[j].size() != static_cast<size_t>(cells) * static_cast<size_t>(mNc)) {
                    continue;
                }
                layers.push_back({i, j, cells, grid_h, grid_w, stride});
                used[i] = true;
                used[j] = true;
                break;
            }
        }

        if (layers.empty()) return ++mTicket;
        for (const auto& layer : layers) {
            append_grid_relative_layer(layer);
        }
    }

    std::vector<int> order(cand_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return cand_scores[a] > cand_scores[b]; });
    const size_t keep = std::min(order.size(), static_cast<size_t>(300));
    boxes.reserve(keep);
    scores.reserve(keep);
    labels.reserve(keep);
    for (size_t oi = 0; oi < keep; ++oi) {
        const int idx = order[oi];
        boxes.push_back(cand_boxes[idx]);
        scores.push_back(cand_scores[idx]);
        labels.push_back(cand_labels[idx]);
    }
    return ++mTicket;
}

void YOLODFLFreePost::receive(uint64_t receipt_no) { (void)receipt_no; }