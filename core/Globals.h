#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>

namespace Core {
    const std::string YOLO_WEIGHTS_FILE = "../resources/yolo/yolov4.weights";
    const std::string YOLO_CFG_FILE = "../resources/yolo/yolov4.cfg";
    const std::string RESNET_WEIGHTS_FILE = "../resources/resnet50/resnet50.weights";
    const std::string RESNET_CFG_FILE = "../resources/resnet50/resnet50.cfg";
    const std::string COCO_NAMES_FILE = "../resources/yolo/coco.names";
    // provide own dataset path
    const std::string DATASET_DIR = "../../../datasets/mixkit/";
    const std::string VECTOR_DIR = "../../../datasets/mixkit-vectors/";
    const std::list<std::string> CLASSES = {"walking", "running"};
    // detector constants
    const double CONFIDENCE_THRESHOLD = 0.5;
    const double NON_MAX_SP_THRESHOLD = 0.4;
    const int YOLO_SIZE = 320;
    const int RN50_SIZE = 224;
}

#endif //GLOBALS_H