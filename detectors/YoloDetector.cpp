#include <fstream>
#include <opencv2/dnn.hpp>
#include "YoloDetector.h"
#include "../core/Globals.h"

namespace Video {
    YoloDetector::YoloDetector() {
        // Load all class names (object names) from coco.names
        std::string classesFile = std::string(Core::COCO_NAMES_FILE);
        std::ifstream ifs(classesFile.c_str());
        std::string line;

        int lineCount = 0;
        while (std::getline(ifs, line)) {
            classes.insert(std::pair<std::string, int> (line, lineCount));
            lineCount++;
        }

        // Load yolo model with cfg and weights
        // For yolov4
        yoloNet = cv::dnn::readNetFromDarknet(std::vector<uchar>(Core::YOLO_CFG_FILE.begin(), Core::YOLO_CFG_FILE.end()), 
                                              std::vector<uchar>(Core::YOLO_WEIGHTS_FILE.begin(), Core::YOLO_WEIGHTS_FILE.end()));
        yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        darkNet = cv::dnn::readNetFromDarknet(std::vector<uchar>(Core::DARKNET_CFG_FILE.begin(), Core::DARKNET_CFG_FILE.end()), 
                                              std::vector<uchar>(Core::DARKNET_WEIGHTS_FILE.begin(), Core::DARKNET_WEIGHTS_FILE.end()));
        darkNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        darkNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Find the names of the last layer of the neural network
        // lastLayerNames = yoloNet.getUnconnectedOutLayersNames();
    }
    
    cv::Mat YoloDetector::grepObjects(cv::UMat frame, const std::string &className) {
        cv::Mat ans;
        cv::UMat blob;
        // cv::UMat sub; // used to grep a subimage
        std::vector<cv::Mat> layerOut;   
        std::vector<cv::Rect> returnBox;  
        cv::Rect bestBox;  
        double maxConfidence = Core::CONFIDENCE_THRESHOLD; 

        cv::dnn::blobFromImage(frame, blob, 0.017, cv::Size(Core::YOLO_SIZE, Core::YOLO_SIZE), cv::Scalar(103.94,116.78,123.68));

        yoloNet.setInput(blob);                     // Feed this blob image to the network
        yoloNet.forward(layerOut, lastLayerNames);  // Get the output from the neural network, which is the last layer

        // Now look through each finding and see what objects we detected.
        if (!classes.contains(className)) {
            std::cout << "Failed to find class \"" << className << "\""<< std::endl;
            return ans;
        }
        for (const auto &out : layerOut) {
            auto data = (float *) out.data;
            for (int j=0; j<out.rows; ++j, data+=out.cols) {
                cv::Mat scores = out.row(j).colRange(5, out.cols);

                cv::Point classIdPoint;
                double confidence;
                // This finds the index of the max value in the scores array and returns the value at
                // this index, stored to "confidence" here, and also returns this index in classIdPoint.
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > maxConfidence && 
                    classIdPoint.x < (int) classes.size() && 
                    classes[className] == classIdPoint.x) {
            
                    int centerX = (int) (data[0] * frame.cols);
                    int centerY = (int) (data[1] * frame.rows);
                    int width = (int) (data[2] * frame.cols);
                    int height = (int) (data[3] * frame.rows);
                    // prevent frame overlapping
                    int left = std::max(centerX - width / 2, 0); // Store as int to get rid of decimals
                    int top = std::max(0, centerY - height / 2);
                    width = std::min(width, frame.cols - left);
                    height = std::min(height, frame.rows - top),
                    maxConfidence = confidence;
                    bestBox = cv::Rect(left, top, width, height);
                }
            }
        }

        if (bestBox.empty())
            return ans;
        // std::c
        auto sub = frame(bestBox);
        cv::dnn::blobFromImage(sub, blob,  0.017, cv::Size(Core::RN50_SIZE, Core::RN50_SIZE), cv::Scalar(103.94,116.78,123.68));

        darkNet.setInput(blob);

        cv::Mat result = darkNet.forward("conv_22"); // ignore last 5 layers
        int lastSizes = result.size[2] * result.size[3]; // its 14x14 frames

        for (int i=0; i < result.size[1]; i++) {
            auto s = cv::sum(result.col(i));
            ans.push_back(s[0]/lastSizes);
        }

        cv::transpose(ans, ans);
        return ans;
    }
}
