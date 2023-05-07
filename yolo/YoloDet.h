//
// Created by mallumo on 7.5.2023.
//

#ifndef YOLOV8_YOLODET_H
#define YOLOV8_YOLODET_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

class YoloDet {


private:
    cv::dnn::Net net;
    const cv::Size modelShape = cv::Size(640, 640);

    int classesCount = 80;
    float x_factor = 1.0f;
    float y_factor = 1.0f;

    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};


public:
    struct Detection {
        int class_id{0};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };

    void init(const std::string &modelPath);

    std::vector<Detection> detect(cv::Mat &mat);
};


#endif //YOLOV8_YOLODET_H
