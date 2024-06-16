#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>


namespace Core {
    const std::string YOLO_WEIGHTS_FILE = "./resources/yolo/yolov4.weights";
    const std::string YOLO_CFG_FILE = "./resources/yolo/yolov4.cfg";
    const std::string COCO_NAMES_FILE = "./resources/yolo/coco.names";
    const std::string DATASET_DIR = "../datasets/mixkit/";
    const std::string VECTOR_DIR = "../datasets/mixkit-vectors/";
    const std::list<std::string> CLASSES = {"walking", "running"};
}

#endif //GLOBALS_H