#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_PASSTHROUGH_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_PASSTHROUGH_H_

#include <array>
#include <vector>

#include "demo/post_yolo_anchorless.h"

class YOLOAnchorlessPassthroughPost : public YOLOAnchorlessPost {
public:
    YOLOAnchorlessPassthroughPost(int nc, int imh, int imw, float conf_thres,
                                  float iou_thres, bool decode_bbox);
    ~YOLOAnchorlessPassthroughPost() override = default;

protected:
    void postDecode(std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                    std::vector<int>& labels, std::vector<std::vector<float>>& extras,
                    const cv::Mat& model_input_image) override;
};

#endif
