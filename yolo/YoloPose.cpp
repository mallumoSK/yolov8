//
// Created by mallumo on 7.5.2023.
//

#include "YoloPose.h"

void YoloPose::init(const std::string &modelPath) {
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<YoloPose::Person> YoloPose::detect(cv::Mat &mat) {
    static cv::Mat blob;
    static std::vector<cv::Mat> outputs;

    cv::dnn::blobFromImage(mat, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    const int channels = outputs[0].size[2];
    const int anchors = outputs[0].size[1];
    outputs[0] = outputs[0].reshape(1, anchors);
    cv::Mat output = outputs[0].t();


    std::vector<cv::Rect> bboxList;
    std::vector<float> scoreList;
    std::vector<int> indicesList;
    std::vector<std::vector<Keypoint>> kpList;

    for (int i = 0; i < channels; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bbox_ptr = row_ptr;
        auto score_ptr = row_ptr + 4;
        auto kp_ptr = row_ptr + 5;

        float score = *score_ptr;
        if (score > modelScoreThreshold) {
            float x = *bbox_ptr++;
            float y = *bbox_ptr++;
            float w = *bbox_ptr++;
            float h = *bbox_ptr;

            float x0 = clamp((x - 0.5f * w) * 1.0F, 0.f, float(modelShape.width));
            float y0 = clamp((y - 0.5f * h) * 1.0F, 0.f, float(modelShape.height));
            float x1 = clamp((x + 0.5f * w) * 1.0F, 0.f, float(modelShape.width));
            float y1 = clamp((y + 0.5f * h) * 1.0F, 0.f, float(modelShape.height));

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            std::vector<Keypoint> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kp_ptr + 3 * k));
                float kps_y = (*(kp_ptr + 3 * k + 1));
                float kps_s = *(kp_ptr + 3 * k + 2);
                kps_x = clamp(kps_x, 0.f, float(modelShape.width));
                kps_y = clamp(kps_y, 0.f, float(modelShape.height));

                kps.emplace_back(kps_x, kps_y, kps_s);
            }

            bboxList.push_back(bbox);
            scoreList.push_back(score);
            kpList.push_back(kps);
        }
    }

    cv::dnn::NMSBoxes(
            bboxList,
            scoreList,
            modelScoreThreshold,
            modelNMSThreshold,
            indicesList
    );

    std::vector<YoloPose::Person> result{};
    for (auto &i: indicesList) {
        result.emplace_back(bboxList[i], scoreList[i], kpList[i]);
    }

    return result;
}

YoloPose::Keypoint::Keypoint(float x, float y, float score) {
    this->position = cv::Point2d(x, y);
    this->conf = score;
}

YoloPose::Person::Person(cv::Rect2i _box, float _score, std::vector<Keypoint> &_kp) {
    this->box = _box;
    this->score = _score;
    this->kp = _kp;
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}