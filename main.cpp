#include <iostream>
#include <fmt/format.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils/Utils.h"
#include "main.h"
#include "core/Globals.h"

// #include "video_handler.h"

int main() {
    // check for core files and dirs existence
    if (!checkRequiredFiles())
        exit(-1);
    
    if (!std::filesystem::exists(Core::DATASET_DIR)) {
        Utils::print("Dataset directory path " + Core::DATASET_DIR + " does not exist! Make sure to configure right path in Globals.h");
        exit(-1);
    }
    for (const auto& className : Core::CLASSES) {
        if (!std::filesystem::exists(Core::DATASET_DIR + className)) {
            Utils::print("Required class " + className + " does not exist! "
            "Make sure to provide requested folder in "+ Core::DATASET_DIR);
            exit(-1);
        }
    }
    if (!std::filesystem::exists(Core::VECTOR_DIR)) {
        std::filesystem::create_directory(Core::VECTOR_DIR);
    }

    // iterate through every video in dataset and check for vector
    for (const auto& className : Core::CLASSES) {
        for (const auto& dirEntry : std::filesystem::directory_iterator(Core::DATASET_DIR + className)) {
            
            if (!std::filesystem::exists(Core::VECTOR_DIR + className)) {
                std::cout << "creating vector for " << dirEntry << std::endl;
            }
        }
    }
     
    

    cv::VideoCapture video("video.mp4");

    // Create a human action recognition model
    // cv::Ptr<cv::dnn::Net> net;

    // Load pre-trained model weights and architecture
    auto net = cv::dnn::readNetFromCaffe("model.prototxt", "model.caffemodel");

    // Loop through frames in the video
    while (true) {
        cv::Mat frame;
        video >> frame;

        // Preprocess frame (resize, normalize, etc.)

        // Pass frame through the model for action recognition
        cv::Mat output = net.forward();

        // Process output to get human action predictions

        // Display the frame with action predictions

        // Break the loop if video ends
        if (frame.empty()) {
            break;
        }

        cv::imshow("Action Recognition", frame);

        // Wait for key press or exit
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Release video capture and destroy windows
    video.release();
    cv::destroyAllWindows();

    return 0;
}

bool checkRequiredFiles() {
    Utils::StringList required_files;
    required_files.emplace_back(Core::COCO_NAMES_FILE);
    required_files.emplace_back(Core::YOLO_CFG_FILE);
    required_files.emplace_back(Core::YOLO_WEIGHTS_FILE);

    if (!Utils::validateFiles(required_files)) {
        Utils::print("Required yolo files are missing, make sure below files exists!!");
        for (auto &file: required_files) {
            Utils::print(file);
        }
        return false;
    }

    required_files.clear();
    return true;
}