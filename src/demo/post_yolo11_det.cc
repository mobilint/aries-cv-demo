#include "demo/post_yolo11_det.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "qbruntime/qbruntime.h"

#ifdef DEMO_DEBUG_LOG
#include <iostream>
#endif

YOLO11DetPostProcessor::YOLO11DetPostProcessor(int nc, int imh, int imw, float conf_thres,
                                               float iou_thres)
    : m_nc(nc),
      m_imh(imh),
      m_imw(imw),
      m_conf_thres(conf_thres),
      m_iou_thres(iou_thres) {}

float YOLO11DetPostProcessor::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

namespace {

void softmax16_inplace(std::array<float, 16>& a) {
    float maxv = a[0];
    for (int i = 1; i < 16; i++) maxv = std::max(maxv, a[i]);

    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        a[i] = std::exp(a[i] - maxv);
        sum += a[i];
    }
    if (sum <= 0.0f) return;
    for (int i = 0; i < 16; i++) a[i] /= sum;
}

// DFL decode for one grid cell.
// box_dfl layout: [N, 64] flattened, N = grid_h * grid_w, 64 = 4 * 16 bins.
std::array<float, 4> decode_box_xyxy_dfl(const mobilint::NDArray<float>& box_dfl,
                                         int cell_idx, int grid_x, int grid_y,
                                         int stride) {
    constexpr int kBins = 16;
    constexpr int kPerCell = 4 * kBins;
    const int base = cell_idx * kPerCell;

    float dist[4] = {0, 0, 0, 0};
    for (int side = 0; side < 4; side++) {
        std::array<float, kBins> prob{};
        for (int k = 0; k < kBins; k++) {
            prob[k] = box_dfl[base + side * kBins + k];
        }
        softmax16_inplace(prob);
        float expected = 0.0f;
        for (int k = 0; k < kBins; k++) expected += prob[k] * (float)k;
        dist[side] = expected;
    }

    float xmin = (float)grid_x - dist[0] + 0.5f;
    float ymin = (float)grid_y - dist[1] + 0.5f;
    float xmax = (float)grid_x + dist[2] + 0.5f;
    float ymax = (float)grid_y + dist[3] + 0.5f;

    float x = (xmin + xmax) * 0.5f * (float)stride;
    float y = (ymin + ymax) * 0.5f * (float)stride;
    float w = (xmax - xmin) * (float)stride;
    float h = (ymax - ymin) * (float)stride;

    float x1 = x - w * 0.5f;
    float y1 = y - h * 0.5f;
    float x2 = x + w * 0.5f;
    float y2 = y + h * 0.5f;
    return {x1, y1, x2, y2};
}

}  // namespace

float YOLO11DetPostProcessor::iou_xyxy(const std::array<float, 4>& a,
                                       const std::array<float, 4>& b) {
    float ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
    float bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
    float area_b = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
    float denom = area_a + area_b - inter_area;
    if (denom <= 0.0f) return 0.0f;
    return inter_area / denom;
}

void YOLO11DetPostProcessor::nms_classwise(const std::vector<std::array<float, 4>>& boxes,
                                           const std::vector<float>& scores,
                                           const std::vector<int>& labels,
                                           float iou_thres,
                                           std::vector<int>& keep_indices) {
    keep_indices.clear();
    if (boxes.empty()) return;

    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t _i = 0; _i < order.size(); _i++) {
        int i = order[_i];
        if (suppressed[i]) continue;
        keep_indices.push_back(i);

        for (size_t _j = _i + 1; _j < order.size(); _j++) {
            int j = order[_j];
            if (suppressed[j]) continue;
            if (labels[i] != labels[j]) continue;
            if (iou_xyxy(boxes[i], boxes[j]) > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
}

uint64_t YOLO11DetPostProcessor::enqueue(cv::Mat& im,
                                         std::vector<mobilint::NDArray<float>>& npu_outs,
                                         std::vector<std::array<float, 4>>& boxes,
                                         std::vector<float>& scores,
                                         std::vector<int>& labels,
                                         std::vector<std::vector<float>>& extras) {
    (void)im;
    extras.clear();
    boxes.clear();
    scores.clear();
    labels.clear();

#ifdef DEMO_DEBUG_LOG
    static int dbg_count = 0;
    const bool dbg_print = ((dbg_count++ % 60) == 0);
    if (dbg_print) {
        std::cout << "[YOLO11PP] npu_outs=" << npu_outs.size() << " sizes:";
        for (size_t i = 0; i < npu_outs.size(); i++) {
            std::cout << " " << npu_outs[i].size();
        }
        std::cout << " (expect 6 outs: {20,40,80}x{20,40,80}x{nc,64}, nc=" << m_nc << ")"
                  << std::endl;
    }
#endif

    if (npu_outs.size() != 6) {
#ifdef DEMO_DEBUG_LOG
        if (dbg_print) {
            std::cout << "[YOLO11PP] unexpected output count=" << npu_outs.size()
                      << ", expected 6 (cls/reg split across 3 scales)" << std::endl;
        }
#endif
        return ++m_ticket;
    }

    auto find_by_size = [&](size_t target) -> int {
        for (int i = 0; i < (int)npu_outs.size(); i++) {
            if (npu_outs[i].size() == target) return i;
        }
        return -1;
    };

    if (!m_levels_ready) {
        m_levels = {
            Level{32, m_imh / 32, m_imw / 32, -1, -1},
            Level{16, m_imh / 16, m_imw / 16, -1, -1},
            Level{8, m_imh / 8, m_imw / 8, -1, -1},
        };

        for (auto& lv : m_levels) {
            size_t cls_size = (size_t)lv.grid_h * (size_t)lv.grid_w * (size_t)m_nc;
            size_t box_size = (size_t)lv.grid_h * (size_t)lv.grid_w * (size_t)64;
            lv.cls_idx = find_by_size(cls_size);
            lv.box_idx = find_by_size(box_size);
            if (lv.cls_idx < 0 || lv.box_idx < 0) {
#ifdef DEMO_DEBUG_LOG
                if (dbg_print) {
                    std::cout << "[YOLO11PP] missing outputs for stride=" << lv.stride
                              << " expect cls=" << cls_size << " box=" << box_size
                              << " got_indices(cls,box)=(" << lv.cls_idx << ","
                              << lv.box_idx << ")" << std::endl;
                }
#endif
                return ++m_ticket;
            }
        }

        m_levels_ready = true;
    };

    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;
    cand_boxes.reserve(8400);
    cand_scores.reserve(8400);
    cand_labels.reserve(8400);

    for (const auto& lv : m_levels) {
        const auto& cls = npu_outs[lv.cls_idx];
        const auto& box = npu_outs[lv.box_idx];

        const int ncell = lv.grid_h * lv.grid_w;
        for (int cell = 0; cell < ncell; cell++) {
            int cls_base = cell * m_nc;

            int label = 0;
            float conf = sigmoid(cls[cls_base]);

            for (int c = 0; c < m_nc; c++) {
                float p = sigmoid(cls[cls_base + c]);
                if (p > conf) {
                    conf = p;
                    label = c;
                }
            }

            if (conf < m_conf_thres) continue;

            int gx = cell % lv.grid_w;
            int gy = cell / lv.grid_w;
            std::array<float, 4> xyxy = decode_box_xyxy_dfl(box, cell, gx, gy, lv.stride);

            float x1 = std::max(0.0f, std::min(xyxy[0], (float)m_imw));
            float y1 = std::max(0.0f, std::min(xyxy[1], (float)m_imh));
            float x2 = std::max(0.0f, std::min(xyxy[2], (float)m_imw));
            float y2 = std::max(0.0f, std::min(xyxy[3], (float)m_imh));
            if (x2 <= x1 || y2 <= y1) continue;

            cand_boxes.push_back({x1, y1, x2, y2});
            cand_scores.push_back(conf);
            cand_labels.push_back(label);
        }
    }

    std::vector<int> keep;
    nms_classwise(cand_boxes, cand_scores, cand_labels, m_iou_thres, keep);

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO11PP] cand=" << cand_boxes.size() << " keep=" << keep.size()
                  << " conf_thres=" << m_conf_thres << " iou_thres=" << m_iou_thres
                  << std::endl;
    }
#endif

    boxes.reserve(keep.size());
    scores.reserve(keep.size());
    labels.reserve(keep.size());
    for (int idx : keep) {
        boxes.push_back(cand_boxes[idx]);
        scores.push_back(cand_scores[idx]);
        labels.push_back(cand_labels[idx]);
    }

    return ++m_ticket;
}

void YOLO11DetPostProcessor::receive(uint64_t receipt_no) { (void)receipt_no; }
