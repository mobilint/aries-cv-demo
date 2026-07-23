#ifndef DEMO_INCLUDE_POST_YOLO11_DET_H_
#define DEMO_INCLUDE_POST_YOLO11_DET_H_

#include <array>
#include <cstdint>
#include <vector>

#include "demo/post.h"
#include "qbruntime/qbruntime.h"

class YOLO11DetPostProcessor : public PostProcessor {
public:
    YOLO11DetPostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres);
    ~YOLO11DetPostProcessor() override = default;

    uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;

    void receive(uint64_t receipt_no) override;

private:
    struct Level {
        int stride;
        int grid_h;
        int grid_w;
        int cls_idx;
        int box_idx;
    };
    bool m_levels_ready = false;
    std::array<Level, 3> m_levels;

    int m_nc;
    int m_imh;
    int m_imw;
    float m_conf_thres;
    float m_iou_thres;
    uint64_t m_ticket = 0;

    static float sigmoid(float x);
    static float iou_xyxy(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static void nms_classwise(const std::vector<std::array<float, 4>>& boxes,
                              const std::vector<float>& scores,
                              const std::vector<int>& labels, float iou_thres,
                              std::vector<int>& keep_indices);
};

#endif
