//
// Created by mallumo on 7.5.2023.
//

#include "YoloDet.h"

void YoloDet::init(const std::string &modelPath) {
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<YoloDet::Detection> YoloDet::detect(cv::Mat &mat) {
    static cv::Mat blob;
    static std::vector<cv::Mat> outputs;

    cv::dnn::blobFromImage(mat, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    auto data = (float *) outputs[0].data;

    std::vector<int> class_ids{};
    std::vector<float> confidences{};
    std::vector<cv::Rect> boxes{};

    for (int i = 0; i < rows; ++i) {
        float *classes_scores = data + 4;

        cv::Mat scores(1, classesCount, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id);

        if (maxClassScore > modelScoreThreshold) {
            confidences.push_back(float(maxClassScore));
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.emplace_back(left, top, width, height);
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};

    for (int idx: nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detections.push_back(result);
    }

    return detections;
}