#include "main.h"



int main() {
    auto image = ImageTools::imageFromPath("../__sampledata/bus_640.jpg");

//    yoloDetection(image);
    yoloPose(image);
    return 0;
}

void yoloPose(cv::Mat &image) {
    auto yolo = YoloPose();
    yolo.init("../__sampledata/yolov8n-pose.onnx");

    auto result = yolo.detect(image);

    ImageTools::draw(result, image);
    ImageTools::show(image);
}

void yoloDetection(cv::Mat &image) {
    auto yolo = YoloDet();
    yolo.init("../__sampledata/yolov8n.onnx");

    auto result = yolo.detect(image);

    ImageTools::draw(result, image);
    ImageTools::show(image);
}





