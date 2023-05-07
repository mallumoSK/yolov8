//
// Created by mallumo on 7.5.2023.
//

#include "ImageTools.h"

void ImageTools::show(cv::Mat &image) {
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);
}

void ImageTools::draw(std::vector<YoloPose::Person> &detections, cv::Mat &image) {
    auto textColor = cv::Scalar(255, 255, 255);
    auto boxColor = cv::Scalar(0,  0,255);

    for (YoloPose::Person &item: detections) {
        cv::rectangle(image, item.box, boxColor, 1);

        std::string infoString = std::to_string(item.score);
        cv::Size textSize = cv::getTextSize(infoString, cv::QT_FONT_NORMAL, 1, 1, nullptr);
        cv::Rect textBox(item.box.x, item.box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(item.box.x + 5, item.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1,
                    0);
        for (YoloPose::Keypoint kp:item.kp) {
            cv::circle(image, kp.position, 3, boxColor, cv::FILLED);
        }
    }
}

void ImageTools::draw(std::vector<YoloDet::Detection> &detections, cv::Mat &image) {
    auto textColor = cv::Scalar(255, 255, 255);
    auto boxColor = cv::Scalar(0,  0,255);

    for (YoloDet::Detection &item: detections) {
        cv::Rect box = item.box;

        cv::rectangle(image, box, boxColor, 1);

        std::string infoString =
                "[" + std::to_string( item.class_id) + "] " + std::to_string(item.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(infoString, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1,
                    0);

    }
}


cv::Mat ImageTools::imageFromPath(const std::string &imagePath) {
    return cv::imread(imagePath);
}


