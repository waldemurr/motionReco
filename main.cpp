#include <iostream>
#include <fmt/format.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils/Utils.h"
#include "main.h"
#include "core/Globals.h"
#include "detectors/YoloDetector.h"

// #include "video_handler.h"

int main() {
    const std::string objectName {"person"};
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
    Video::YoloDetector detector; 
    // iterate through every video in dataset and check for vector
    int cnt = 0;
    for (const auto& className : Core::CLASSES) {
        for (const auto& dirEntry : std::filesystem::directory_iterator(Core::DATASET_DIR + className)) {
            if (!std::filesystem::exists(Core::VECTOR_DIR + className)) {
                std::cout << "creating vector for " << dirEntry << std::endl;
                // open video stream
                cv::VideoCapture capture {dirEntry.path().c_str()};
                if (!capture.isOpened()) {
                    Utils::print("Failed to open video stream");
                    break;
                }
                bool running = true;
                cv::UMat frame;
                // Get a new frame from video stream
                while (running) {
                    capture >> frame;
                    if (frame.empty()) {
                        Utils::print("Stream ended. Returning...");
                        running = false;
                        break;
                    }
                    detector.grepObjects(frame, objectName);
                    cv::imshow("Video output", frame);
                    int key = cv::waitKey(10);
                    switch (key) {
                        case 27: // esc
                            running = false;
                            break;
                    }
                    running = false;    
                }
                break;
            
            }
        }
    }
    // cv::destroyAllWindows();

    return 0;
}

bool checkRequiredFiles() {
    Utils::StringList required_files;
    required_files.emplace_back(Core::COCO_NAMES_FILE);
    required_files.emplace_back(Core::YOLO_CFG_FILE);
    required_files.emplace_back(Core::YOLO_WEIGHTS_FILE);

    if (!Utils::validateFiles(required_files)) {
        Utils::print("Required yolo files are missing, make sure below files exist!");
        for (auto &file: required_files) {
            Utils::print(file);
        }
        return false;
    }

    required_files.clear();
    return true;
}