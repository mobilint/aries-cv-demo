#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_FACE_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_FACE_H_

#include "demo/post_yolo_anchorless.h"

class YOLOAnchorlessFacePost : public YOLOAnchorlessPost {
public:
    YOLOAnchorlessFacePost(int imh, int imw, float conf_thres, float iou_thres)
        // yolo11s-face uses YOLO anchorless DFL box heads.  In the original
        // multi-channel demo the trailing `false` in the face constructor meant
        // "do not auto-start the worker thread", not "skip bbox decoding".
        // Keep bbox decoding enabled here so 64-channel DFL outputs are decoded
        // into model-input xyxy coordinates before undoing letterbox padding.
        : YOLOAnchorlessPost(1, imh, imw, conf_thres, iou_thres, true) {}
};

#endif