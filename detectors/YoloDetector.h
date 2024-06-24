#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>

namespace Video {
    class YoloDetector{
        public:
            explicit YoloDetector();
            /**
             * detectObjects            Returns a vector with all found objects.
             */
            std::vector<cv::Rect> detectObjects(cv::UMat frame);
            /*!
             * grepObjects              Returns vector with all grepped objects.
             */
            std::vector<double> grepObjects(cv::UMat frame, const std::string &className);
        private:
            /**
             * Get the found objects from the net output
             */
            std::vector<cv::Rect> postProcess(cv::InputOutputArray frame, const std::vector<cv::Mat> &layerOuts);
            cv::dnn::Net yoloNet;
            cv::dnn::Net resNet;
            std::map<std::string, int> classes;
            std::vector<cv::String> lastLayerNames;
            const int prependingVals = 5;
    };
}

#endif // YOLO_DETECTOR_H