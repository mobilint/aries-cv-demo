#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_POSE_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_POSE_H_

#include "demo/post_yolo_dflfree.h"

class YOLODFLFreePosePost : public YOLODFLFreePost {
public:
    YOLODFLFreePosePost(int nc, int imh, int imw, float conf_thres, float iou_thres);

    uint64_t enqueue(cv::Mat& im, std::vector<mobilint::NDArray<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;

private:
    int mNextra = 51;
};

#endif