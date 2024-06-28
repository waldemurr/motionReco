# People motion reco project
This project is a CV playground to test modern C++ ability for writing custom neural networks and process visual data.
## Special thanks: 
- [mixkit](https://mixkit.co/) for providing free raw video data
- [darknet](https://pjreddie.com/darknet/) for providing most C++ compatible with OpenCV pretrained models
## Technoligies:
- [OpenCv](https://opencv.org/) - install guide [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
- [Torch](https://pytorch.org) - install guide [here](https://pytorch.org/cppdocs/installing.html)
- [fmt](https://fmt.dev) 


# Quickstart

## Download fmt dependencies
<tab><tab>sudo apt install libfmt-dev libjansson-dev libxss-dev libnotify-dev libgtkmm-2.4-dev

## Yolo

For yolo detection to work you need some yolo files.

<tab><tab>cd \<your project folder\> \
mkdir -p resources/yolo\
cd resources/yolo\
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

Copy the text found in the following url into your coco.names file: https://github.com/pjreddie/darknet/blob/master/data/coco.names

#### Yolov4
<tab><tab>wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights\
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

#### Yolov3
<tab><tab>wget https://pjreddie.com/media/files/yolov3.weights\
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

## Darknet
<tab><tab>cd \<your project folder\> \
mkdir -p resources/darknet && cd resources/darknet\
<tab><tab>wget http://pjreddie.com/media/files/darknet19_448.weights\
wget https://github.com/pjreddie/darknet/blob/master/cfg/darknet19_448.cfg

## Torch
<tab><tab>cd \<your project folder\> \
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip\
unzip libtorch-shared-with-deps-latest.zip\
rm libtorch-shared-with-deps-latest.zip