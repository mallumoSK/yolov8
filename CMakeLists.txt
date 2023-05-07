cmake_minimum_required(VERSION 3.25)
project(yolov8)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(PROJECT_SOURCES
        main.cpp main.h
        yolo/YoloDet.cpp yolo/YoloDet.h
        yolo/YoloPose.cpp yolo/YoloPose.h
        tools/ImageTools.cpp tools/ImageTools.h
        )

add_executable(yolov8 ${PROJECT_SOURCES})

target_link_libraries(yolov8 ${OpenCV_LIBS})
