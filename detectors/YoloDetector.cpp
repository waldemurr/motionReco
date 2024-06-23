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
        net = cv::dnn::readNetFromDarknet(Core::YOLO_CFG_FILE, Core::YOLO_WEIGHTS_FILE);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Find the names of the last layer of the neural network
        lastLayerNames = net.getUnconnectedOutLayersNames();
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
        net.setInput(blob);

        // Get the output from the neural network, which is the last layer
        net.forward(layerOut, lastLayerNames);
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
    
    std::vector<std::vector<float>> YoloDetector::grepObjects(cv::UMat frame, const std::vector<std::string> names) {
        std::vector<std::vector<float>> ans;
        cv::UMat blob;
        std::vector<cv::Mat> layerOut;
        std::vector<int> classIndexes;



        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);                     // Feed this blob image to the network
        net.forward(layerOut, lastLayerNames);  // Get the output from the neural network, which is the last layer

        auto data = (float *) layerOut[1].data; //
        std::cout << "data vec " << layerOut[1].data;
        for (int j=0; j<layerOut[1].rows; ++j, data+=layerOut[1].cols) {
            cv::Mat scores = layerOut[1].row(j).colRange(5, layerOut[1].cols);
            for (const auto &className : names) {
                if (!classes.contains(className)) {
                    ans.push_back({});
                    std::cout << "Failed to detect class \"" << className << "\""<< std::endl;
                    continue;
                }
                if (data[prependingVals + classes[className]] > Core::CONFIDENCE_THRESHOLD)
                    std::cout << "Conf "<< data[prependingVals + classes[className]] << std::endl;
                // for (const auto &resVec : )
            }



            // if (confidence > Core::CONFIDENCE_THRESHOLD && classIdPoint.x < (int) classes.size() && classes["person"] == classIdPoint.x) {
            //         int centerX = (int) (data[0] * internalFrame.cols);
            //         int centerY = (int) (data[1] * internalFrame.rows);
            //         int width = (int) (data[2] * internalFrame.cols);
            //         int height = (int) (data[3] * internalFrame.rows);
            //         int left = centerX - width / 2; // Store as int to get rid of decimals
            //         int top = centerY - height / 2;

            //         confidences.push_back((float) confidence);
            //         boxes.emplace_back(left, top, width, height);
            //     }
        }
        std::cout << layerOut[1].size() << std::endl; // # Gives (2028, 85)
        std::cout << layerOut[2].size() << std::endl; // # Gives (8112, 85)
        return ans;
    }
}