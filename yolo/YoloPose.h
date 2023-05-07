//
// Created by mallumo on 7.5.2023.
//

#ifndef YOLOV8_YOLOPOSE_H
#define YOLOV8_YOLOPOSE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

/*

{
    'names': ['person'],
    'boxes': tensor([[x1, y1, x2, y2, conf, cls_idx]]),
    'kp': tensor([[x1_kpt_0, y1_kpt_0, score_0], ... [x1_kpt_n, y1_kpt_n, score_n]])
}

 */

class YoloPose {
private:
    cv::dnn::Net net;
    const cv::Size modelShape = cv::Size(640, 640);

    const float modelScoreThreshold{0.70};
    const float modelNMSThreshold{0.50};

public:
    struct Keypoint {
        Keypoint(float x, float y, float score);

        cv::Point2d position{};
        float conf{0.0};
    };

    struct Person {
        Person(cv::Rect2i _box, float _score, std::vector<Keypoint> &_kp);

        cv::Rect2i box{};
        float score{0.0};
        std::vector<Keypoint> kp{};
    };

    void init(const std::string &modelPath);

    std::vector<Person> detect(cv::Mat &mat);
};

inline static float clamp(float val, float min, float max);

#endif //YOLOV8_YOLOPOSE_H
