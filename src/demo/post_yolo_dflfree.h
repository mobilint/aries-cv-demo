#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_H_

#include "demo/post.h"

class YOLODFLFreePost : public PostProcessor {
public:
    YOLODFLFreePost(int nc, int imh, int imw, float conf_thres, float iou_thres);

    uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;
    void receive(uint64_t receipt_no) override;

protected:
    int mNc;
    int mImh;
    int mImw;
    float mConfThres;
    float mIouThres;
    uint64_t mTicket = 0;

    static float sigmoid(float x);
    static float iou(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static bool inferGridAndStride(int imh, int imw, int num_cells, int& grid_h,
                                   int& grid_w, int& stride);
    static float clampCoord(float v, float lo, float hi);
    static bool looksLikeProbability(float v);
};

#endif