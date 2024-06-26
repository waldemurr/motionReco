#include <iostream>
#include <fmt/format.h>
// #include <thread>
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
    if (!std::filesystem::exists(Core::VECTOR_DIR)) {
        std::filesystem::create_directory(Core::VECTOR_DIR);
    }
    Video::YoloDetector detector; 
    // iterate through every video in dataset and check for vector
    int cnt = 0;
    for (const auto& className : Core::CLASSES) {
        if (!std::filesystem::exists(Core::VECTOR_DIR + className)) {
            std::filesystem::create_directory(Core::VECTOR_DIR + className);
        }
        for (const auto& dirEntry : std::filesystem::directory_iterator(Core::DATASET_DIR + className)) {
            cv::Mat vecData;
            // mat.
            const auto fullname = dirEntry.path().filename().string();
            const auto rawname = fullname.substr(0, fullname.find_last_of("."));
            const auto vecName = Core::VECTOR_DIR + className + "/" + rawname +  ".vec" ;
            if (!std::filesystem::exists(vecName)) {
                std::cout << "creating vector for " << dirEntry << std::endl;
                // continue;
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
                        running = false;
                        break;
                    }
                    auto res = detector.grepObjects(frame, objectName);
                    if (!res.empty()) {
                        if (vecData.empty())
                            res.copyTo(vecData);
                        else
                            vecData.push_back(res);
                    }
                }
                if (!vecData.empty()){
                    // cv::write()
                    std::cout << "Vec size " << rawname << " " << vecData.size << std::endl;
                    cv::FileStorage file(vecName, cv::FileStorage::WRITE);

                    // Write to file!
                    file << rawname << vecData;

                    // Close the file and release all the memory buffers
                    file.release();
                }
                // break;
            
            }
        }
    }
    cv::destroyAllWindows();

    return 0;
}

bool checkRequiredFiles() {
    Utils::StringList required_files;

    required_files.emplace_back(Core::YOLO_CFG_FILE);
    required_files.emplace_back(Core::YOLO_WEIGHTS_FILE);
    required_files.emplace_back(Core::DARKNET_CFG_FILE);
    required_files.emplace_back(Core::DARKNET_WEIGHTS_FILE);
    required_files.emplace_back(Core::COCO_NAMES_FILE);
    for (const auto& className : Core::CLASSES) {
        if (!Utils::validateDir(Core::DATASET_DIR + className)) {
            return false;
        }
    }
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
