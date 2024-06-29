#include <iostream>
#include <fmt/format.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils/Utils.h"
#include "main.h"
#include "core/Globals.h"
#include "detectors/YoloDetector.h"
#include "classifiers/RnnClassifier.h"

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
    int cnt = 0;    // class counter
    int maxRows = 0;
    std::vector<cv::Mat> inputVec;
    cv::Mat input; // input vectors 
    cv::Mat target; // targets ( 0 for walking and 1 for running)
    
    for (const auto& className : Core::CLASSES) {
        if (!std::filesystem::exists(Core::VECTOR_DIR + className)) {
            std::filesystem::create_directory(Core::VECTOR_DIR + className);
        }
        
        // make vectors if not exist
        for (const auto& dirEntry : std::filesystem::directory_iterator(Core::DATASET_DIR + className)) {
            cv::Mat vecData;
            // mat.
            const auto fullname = dirEntry.path().filename().string();
            const auto rawname = fullname.substr(0, fullname.find_last_of("."));
            const auto vecName {Core::VECTOR_DIR + className + "/" + rawname +  ".vec"};
            if (!std::filesystem::exists(vecName)) {
                std::cout << "creating vector for " << dirEntry << std::endl;
                // open video stream
                cv::VideoCapture capture {dirEntry.path().string()};
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
                    // init file
                    cv::FileStorage file(vecName, cv::FileStorage::WRITE);

                    // Write to file!
                    file << rawname << vecData;

                    // Close the file and release all the memory buffers
                    file.release();
                }
                // break;
            
            }
        }
        cv::FileStorage fileRead;
        // now read them and put into matrix
        for (const auto& vecEntry : std::filesystem::directory_iterator(Core::VECTOR_DIR + className)) {
            fileRead.open(vecEntry.path().string(), cv::FileStorage::READ);
            cv::Mat inpEntry;
            const auto fullname = vecEntry.path().filename().string();
            const auto rawname = fullname.substr(0, fullname.find_last_of("."));
            fileRead[rawname] >> inpEntry;
            // proper resizing
            
            int newMatsize[3] = {1, inpEntry.cols, inpEntry.rows};
            maxRows = std::max(maxRows, inpEntry.rows);
            inpEntry = inpEntry.reshape(1, 3, newMatsize);
            inputVec.push_back(inpEntry);
            
            target.push_back(cnt);
            fileRead.release();
        }
        cnt ++;
    }

    classifiers::RnnNet rnn;
    const std::vector<int> layer_neuron_num{Core::DARKNET_OUT_SIZE, 
                                            Core::DARKNET_OUT_SIZE/2, 
                                            Core::DARKNET_OUT_SIZE/4, 
                                            1};

    rnn.initNet(layer_neuron_num);
    rnn.initWeights(0, 0, 0.3);
    rnn.train(input, target, 0.2);

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
