#include "demo/post_yolo_dflfree_seg.h"

#include <algorithm>
#include <numeric>

namespace {
struct SegLayerHead {
    int box_idx = -1;
    int cls_idx = -1;
    int mask_idx = -1;
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

YOLODFLFreeSegPost::YOLODFLFreeSegPost(int nc, int imh, int imw, float conf_thres,
                                       float iou_thres)
    : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres),
      mProtoH(std::max(1, imh / mProtoStride)),
      mProtoW(std::max(1, imw / mProtoStride)) {}

const cv::Mat& YOLODFLFreeSegPost::getLabelMask() const { return mLabelMask; }

const cv::Mat& YOLODFLFreeSegPost::getFinalMask() const { return mFinalMask; }

uint64_t YOLODFLFreeSegPost::enqueue(cv::Mat& im,
                                     std::vector<mobilint::NDArray<float>>& npu_outs,
                                     std::vector<std::array<float, 4>>& boxes,
                                     std::vector<float>& scores, std::vector<int>& labels,
                                     std::vector<std::vector<float>>& extras) {
    (void)im;
    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();
    mLabelMask.release();
    mFinalMask.release();
    if (npu_outs.size() < 4 || mNc <= 0) return ++mTicket;

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;

    const size_t proto_expected = static_cast<size_t>(mProtoH) * mProtoW * mNextra;
    int proto_idx = -1;
    for (int i = 0; i < static_cast<int>(npu_outs.size()); ++i) {
        if (npu_outs[i].size() == proto_expected) {
            proto_idx = i;
            break;
        }
    }
    if (proto_idx < 0) return ++mTicket;

    auto append_decoded = [&](int box_idx, int cls_idx, int mask_idx) {
        const int num_det = static_cast<int>(npu_outs[box_idx].size() / 4);
        for (int i = 0; i < num_det; ++i) {
            int best_label = 0;
            float best_score = -1.0f;
            for (int c = 0; c < mNc; ++c) {
                float score = npu_outs[cls_idx][i * mNc + c];
                if (!looksLikeProbability(score)) score = sigmoid(score);
                if (score > best_score) { best_score = score; best_label = c; }
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
            std::vector<float> coeffs(mNextra);
            for (int m = 0; m < mNextra; ++m) coeffs[m] = npu_outs[mask_idx][i * mNextra + m];
            pred_boxes.push_back(box);
            pred_scores.push_back(best_score);
            pred_labels.push_back(best_label);
            pred_extra.push_back(std::move(coeffs));
        }
    };

    if (npu_outs.size() == 4) {
        std::vector<int> det_indices;
        for (int i = 0; i < 4; ++i) if (i != proto_idx) det_indices.push_back(i);
        int box_idx = -1, cls_idx = -1, mask_idx = -1;
        for (int u = 0; u < static_cast<int>(det_indices.size()) && box_idx < 0; ++u) {
            const int i = det_indices[u];
            if (npu_outs[i].size() % 4 != 0) continue;
            const int cells = static_cast<int>(npu_outs[i].size() / 4);
            for (int v = 0; v < static_cast<int>(det_indices.size()); ++v) {
                if (u == v) continue;
                const int j = det_indices[v];
                if (npu_outs[j].size() != static_cast<size_t>(cells * mNc)) continue;
                for (int w = 0; w < static_cast<int>(det_indices.size()); ++w) {
                    if (w == u || w == v) continue;
                    const int k = det_indices[w];
                    if (npu_outs[k].size() == static_cast<size_t>(cells * mNextra)) {
                        box_idx = i; cls_idx = j; mask_idx = k; break;
                    }
                }
                if (box_idx >= 0) break;
            }
        }
        if (box_idx >= 0) append_decoded(box_idx, cls_idx, mask_idx);
    } else {
        std::vector<bool> used(npu_outs.size(), false);
        used[proto_idx] = true;
        std::vector<SegLayerHead> layers;
        for (int i = 0; i < static_cast<int>(npu_outs.size()); ++i) {
            if (used[i] || npu_outs[i].size() % 4 != 0) continue;
            const int cells = static_cast<int>(npu_outs[i].size() / 4);
            int cls_idx = -1, mask_idx = -1;
            for (int j = 0; j < static_cast<int>(npu_outs.size()); ++j) {
                if (!used[j] && j != i && npu_outs[j].size() == static_cast<size_t>(cells * mNc)) { cls_idx = j; break; }
            }
            for (int j = 0; j < static_cast<int>(npu_outs.size()); ++j) {
                if (!used[j] && j != i && j != cls_idx && npu_outs[j].size() == static_cast<size_t>(cells * mNextra)) { mask_idx = j; break; }
            }
            int gh = 0, gw = 0, stride = 0;
            if (cls_idx < 0 || mask_idx < 0 || !inferGridAndStride(mImh, mImw, cells, gh, gw, stride)) continue;
            layers.push_back({i, cls_idx, mask_idx, cells, gh, gw, stride});
            used[i] = used[cls_idx] = used[mask_idx] = true;
        }
        for (const auto& layer : layers) {
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
                std::vector<float> coeffs(mNextra);
                for (int m = 0; m < mNextra; ++m) coeffs[m] = npu_outs[layer.mask_idx][idx * mNextra + m];
                pred_boxes.push_back(box);
                pred_scores.push_back(best_score);
                pred_labels.push_back(best_label);
                pred_extra.push_back(std::move(coeffs));
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

    processMask(npu_outs[proto_idx], extras, boxes, labels);
    return ++mTicket;
}

void YOLODFLFreeSegPost::processMask(const mobilint::NDArray<float>& proto,
                                     const std::vector<std::vector<float>>& masks,
                                     const std::vector<std::array<float, 4>>& boxes,
                                     const std::vector<int>& labels) {
    cv::Mat label_proto = cv::Mat::zeros(mProtoH, mProtoW, CV_32F);
    cv::Mat final_proto = cv::Mat::zeros(mProtoH, mProtoW, CV_32F);
    for (size_t i = 0; i < boxes.size() && i < masks.size(); ++i) {
        if (labels[i] != 0) continue;
        const int x_min = std::max(static_cast<int>(boxes[i][0] / mProtoStride), 0);
        const int y_min = std::max(static_cast<int>(boxes[i][1] / mProtoStride), 0);
        const int x_max = std::min(static_cast<int>(boxes[i][2] / mProtoStride), mProtoW - 1);
        const int y_max = std::min(static_cast<int>(boxes[i][3] / mProtoStride), mProtoH - 1);
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                float v = 0.0f;
                const int proto_base = (y * mProtoW + x) * mNextra;
                for (int c = 0; c < mNextra; ++c) v += masks[i][c] * proto[proto_base + c];
                const float prob = sigmoid(v);
                if (final_proto.at<float>(y, x) < prob) {
                    final_proto.at<float>(y, x) = prob;
                    label_proto.at<float>(y, x) = static_cast<float>(labels[i] + 1);
                }
            }
        }
    }
    cv::resize(label_proto, mLabelMask, cv::Size(mImw, mImh), 0, 0, cv::INTER_NEAREST);
    cv::resize(final_proto, mFinalMask, cv::Size(mImw, mImh), 0, 0, cv::INTER_LINEAR);
}