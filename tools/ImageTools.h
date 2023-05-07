//
// Created by mallumo on 7.5.2023.
//

#ifndef YOLOV8_IMAGETOOLS_H
#define YOLOV8_IMAGETOOLS_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <X11/Intrinsic.h>
#include "../yolo/YoloDet.h"
#include "../yolo/YoloPose.h"

class ImageTools {
public :
    static void show(cv::Mat &image);

    static void draw(std::vector<YoloDet::Detection> &detections, cv::Mat &image);

    static cv::Mat imageFromPath(const std::string &imagePath);

    static void draw(std::vector<YoloPose::Person> &detections, cv::Mat &image);

};


#endif //YOLOV8_IMAGETOOLS_H
