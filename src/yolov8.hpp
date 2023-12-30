#ifndef YOLOV8_HPP
#define YOLOV8_HPP

#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <random>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

struct Config
{
    float confThreshold;
    float nmsThreshold;
    float scoreThreshold;
    int inpWidth;
    int inpHeight;
    std::string onnx_path;
};

struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

const std::vector<std::string> coconame = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

class YOLOV8
{
  public:
    YOLOV8(Config config);
    ~YOLOV8();
    std::vector<Detection> detect(cv::Mat &frame);
    void draw_detections(cv::Mat &frame, std::vector<Detection> &detections);
    void draw_detection(cv::Mat &frame, Detection &detection);

    float confThreshold;
    float nmsThreshold;
    float scoreThreshold;
    int inpWidth;
    int inpHeight;
    float rx; // the width ratio of original image and resized image
    float ry; // the height ratio of original image and resized image
    std::string onnx_path;
    Resize resize;
    ov::Tensor input_tensor;
    ov::InferRequest infer_request;
    ov::CompiledModel compiled_model;
    void initialmodel();
    void preprocess_img(cv::Mat &frame);
    std::vector<Detection> postprocess(float *detections, ov::Shape &output_shape);
};

#endif
