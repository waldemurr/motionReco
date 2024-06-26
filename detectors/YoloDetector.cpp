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
        yoloNet = cv::dnn::readNetFromDarknet(Core::YOLO_CFG_FILE, Core::YOLO_WEIGHTS_FILE);
        yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        darkNet = cv::dnn::readNetFromDarknet(Core::DARKNET_CFG_FILE, Core::DARKNET_WEIGHTS_FILE);
        darkNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        darkNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Find the names of the last layer of the neural network
        lastLayerNames = yoloNet.getUnconnectedOutLayersNames();
    }

    std::vector<cv::Rect> YoloDetector::detectObjects(cv::UMat frame) {
        std::vector<cv::Rect> boxes;
        cv::UMat blob;
        std::vector<cv::Mat> layerOut;

        /* Yolo net takes the image in a different format so it needs to be converted. Every pixel
        needs to have its color range divided by 255 since yolo wants it in range 0->1. Then
        size should be 320x320 (or 416x416 or 608x608) */
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);

        // Feed this blob image to the network
        yoloNet.setInput(blob);

        // Get the output from the neural network, which is the last layer
        yoloNet.forward(layerOut, lastLayerNames);
        /*
        layer_out is a list of 3 elements. The first element has 507 rows and each row has 85 elements.
        print(len(layer_out)) # Gives 3
        Utils::print(layer_out[0].size()); # Gives (300, 85)
        Utils::print(layer_out[1].size()); # Gives (1200, 85)
        Utils::print(layer_out[2].size()); # Gives (4800, 85)
        The first 5 elements in each row, eg layer_out[0][0][0:5] is the bounding box of the found
        object. 0=center_x, 1=center_y, 2=width, 3=height, 4=? (Those are in percentage of the image)
        After that we have 80 elements. Each of those represents one object in coco.names
        and tells the probability that it is that object. So, lets say that the 6th element is 0.6.
        That means that it is 60% probability that the found object is a person. Or if the second
        object is 0.45, it means a bicycle of 45% probability.
        */

        boxes = postProcess(frame, layerOut);

        return boxes;
    }

    std::vector<cv::Rect> YoloDetector::postProcess(cv::InputOutputArray frame, const std::vector<cv::Mat> &layerOuts) {
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        cv::UMat internalFrame = frame.getUMat();

        // Now look through each finding and see what objects we detected.
        for (const auto &out : layerOuts) {
            //scan all boxes, assign box class name with highest score.
            auto data = (float *) out.data;

            for (int j=0; j<out.rows; ++j, data+=out.cols) {
                // The probabilities comes after the 5th element. Get the probabilities into scores.
                cv::Mat scores = out.row(j).colRange(5, out.cols);

                cv::Point classIdPoint;
                double confidence;

                // This finds the index of the max value in the scores array and returns the value at
                // this index, stored to "confidence" here, and also returns this index in classIdPoint.
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                // Only handle objects with higher confidence than confThreashold and only persons
                if (confidence > Core::CONFIDENCE_THRESHOLD && classIdPoint.x < (int) classes.size() && classes["person"] == classIdPoint.x) {
                    int centerX = (int) (data[0] * internalFrame.cols);
                    int centerY = (int) (data[1] * internalFrame.rows);
                    int width = (int) (data[2] * internalFrame.cols);
                    int height = (int) (data[3] * internalFrame.rows);
                    int left = centerX - width / 2; // Store as int to get rid of decimals
                    int top = centerY - height / 2;

                    confidences.push_back((float) confidence);
                    boxes.emplace_back(left, top, width, height);
                }
            }
        }

        // Reduce overlaping boxes with lower confidence
        std::vector<int> indexes;
        std::vector<cv::Rect> returnBox;
        cv::dnn::NMSBoxes(boxes, confidences, Core::CONFIDENCE_THRESHOLD, Core::NON_MAX_SP_THRESHOLD, indexes);

        for (int idx : indexes) {
            returnBox.push_back(boxes[idx]);
        }
        return returnBox;
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
