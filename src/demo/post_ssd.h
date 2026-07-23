#ifndef DEMO_INCLUDE_POST_SSD_H_
#define DEMO_INCLUDE_POST_SSD_H_

#include "demo/post_yolo_dflfree.h"

class SSDPostProcessor : public YOLODFLFreePost {
public:
    SSDPostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres)
        : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres) {}
};

#endif