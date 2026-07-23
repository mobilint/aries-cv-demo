#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_SEG_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_SEG_H_

#include "demo/post_yolo_dflfree.h"

class YOLODFLFreeSegPost : public YOLODFLFreePost {
public:
    YOLODFLFreeSegPost(int nc, int imh, int imw, float conf_thres, float iou_thres);

    uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;

    const cv::Mat& getLabelMask() const;
    const cv::Mat& getFinalMask() const;

private:
    void processMask(const mobilint::NDArray<float>& proto,
                     const std::vector<std::vector<float>>& masks,
                     const std::vector<std::array<float, 4>>& boxes,
                     const std::vector<int>& labels);

    int mNextra = 32;
    int mProtoStride = 4;
    int mProtoH = 0;
    int mProtoW = 0;
    cv::Mat mLabelMask;
    cv::Mat mFinalMask;
};

#endif