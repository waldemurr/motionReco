#ifndef GLOBALS_H
#define GLOBALS_H

#include <list>
#include <string>

namespace Core {
    
    // depending on build platform add or remove "../" with pat
    static std::string RESOURCES_PATH {"../resources/"};
    static std::string DATASETS_PATH {"../../../datasets/"};
    const std::string YOLO_WEIGHTS_FILE {RESOURCES_PATH + "yolo/yolov4.weights"};
    const std::string YOLO_CFG_FILE {RESOURCES_PATH+ "yolo/yolov4.cfg"};
    const std::string COCO_NAMES_FILE {RESOURCES_PATH + "yolo/coco.names"};

    const std::string DARKNET_WEIGHTS_FILE {RESOURCES_PATH + "darknet/darknet19.weights"};
    const std::string DARKNET_CFG_FILE {RESOURCES_PATH + "darknet/darknet19_448.cfg"};
    const int DARKNET_OUT_SIZE {1024};
    // provide own dataset path
    const std::string DATASET_DIR {DATASETS_PATH + "mixkit/"};
    const std::string VECTOR_DIR {DATASETS_PATH + "mixkit-vectors/"};
    const std::list<std::string> CLASSES {"walking", "running"};
    // detector constants
    const double CONFIDENCE_THRESHOLD = 0.5;
    const double NON_MAX_SP_THRESHOLD = 0.4;
    const int YOLO_SIZE = 320;
    const int RN50_SIZE = 448;
}

#endif //GLOBALS_H
