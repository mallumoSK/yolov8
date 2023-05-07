//
// Created by mallumo on 6.5.2023.
//

/*
 * https://github.com/triple-Mu/YOLOv8-TensorRT/tree/main/csrc/pose/normal
 *
 * https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/csrc/pose/normal/include/yolov8-pose.hpp
 *
 *
 * https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-CPP-Inference
 *
 * yolo export model=yolov8n-pose.pt  format=onnx opset=12  //  1, 56, 8400
 * yolo export model=yolov8n.pt  format=onnx opset=12       //  1, 84, 8400
 *
 * in:  [1, 3, 640, 640]
 * out: [1, 56, 8400] / [1, 84, 8400]
 */

#ifndef YOLOV8_MAIN_H
#define YOLOV8_MAIN_H

#include "yolo/YoloDet.h"
#include "yolo/YoloPose.h"
#include "tools/ImageTools.h"


void yoloDetection(cv::Mat &image);
void yoloPose(cv::Mat &mat);


#endif //YOLOV8_MAIN_H
