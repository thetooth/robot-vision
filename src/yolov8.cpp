#include "yolov8.hpp"

#include <iostream>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;
using namespace dnn;

YOLOV8::YOLOV8(Config config)
{
    confThreshold = config.confThreshold;
    nmsThreshold = config.nmsThreshold;
    scoreThreshold = config.scoreThreshold;
    inpWidth = config.inpWidth;
    inpHeight = config.inpHeight;
    onnx_path = config.onnx_path;
    initialmodel();
}

YOLOV8::~YOLOV8()
{
}

std::vector<Detection> YOLOV8::detect(Mat &frame)
{
    preprocess_img(frame);
    infer_request.infer();
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();
    return postprocess(detections, output_shape);
}

void YOLOV8::initialmodel()
{
    ov::Core core;
    auto availableDevices = core.get_available_devices();
    for (auto &&device : availableDevices)
    {
        spdlog::info("Device: {}", device);
    }

    std::shared_ptr<ov::Model> model = core.read_model(onnx_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::RGB)
        .set_shape({1, 640, 640, 3});
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255, 255, 255}); // .scale({ 112, 112, 112 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    compiled_model = core.compile_model(model, "MULTI:GPU,GNA");
    infer_request = compiled_model.create_infer_request();
}

void YOLOV8::preprocess_img(Mat &frame)
{
    try
    {
        float width = frame.cols;
        float height = frame.rows;
        cv::Size new_shape = cv::Size(inpWidth, inpHeight);
        float r = float(new_shape.width / max(width, height));
        int new_unpadW = int(round(width * r));
        int new_unpadH = int(round(height * r));

        cv::resize(frame, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);
        resize.resized_image = resize.resized_image;
        resize.dw = new_shape.width - new_unpadW;
        resize.dh = new_shape.height - new_unpadH;
        cv::Scalar color = cv::Scalar(100, 100, 100);
        cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT,
                           color);

        rx = (float)frame.cols / (float)(resize.resized_image.cols - resize.dw);
        ry = (float)frame.rows / (float)(resize.resized_image.rows - resize.dh);
        float *input_data = (float *)resize.resized_image.data;
        input_tensor =
            ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "unknown exception" << std::endl;
    }
}

std::vector<Detection> YOLOV8::postprocess(float *detections, ov::Shape &output_shape)
{
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    int out_rows = output_shape[1];
    int out_cols = output_shape[2];
    const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)detections);

    for (int i = 0; i < det_output.cols; ++i)
    {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 84);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > 0.3)
        {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nms_result);

    std::vector<Detection> output;
    for (size_t i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    return output;
}

void YOLOV8::draw_detections(cv::Mat &frame, std::vector<Detection> &detections)
{
    for (size_t i = 0; i < detections.size(); i++)
    {
        draw_detection(frame, detections[i]);
    }
}

void YOLOV8::draw_detection(cv::Mat &frame, Detection &detection)
{
    auto box = detection.box;
    auto classId = detection.class_id;
    // if (classId != 0) continue;
    auto confidence = detection.confidence;

    box.x = rx * box.x;
    box.y = ry * box.y;
    box.width = rx * box.width;
    box.height = ry * box.height;

    float xmax = box.x + box.width;
    float ymax = box.y + box.height;

    // detection box
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> dis(100, 255);
    // cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), color, 3);

    // Detection box text
    std::string classString = coconame[classId] + ' ' + std::to_string(confidence).substr(0, 4);
    cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
    cv::rectangle(frame, textBox, color, cv::FILLED);
    cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0),
                2, 0);

    // cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0),
    // cv::FILLED); cv::putText(frame, coconame[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX,
    // 0.5, cv::Scalar(0, 0, 0));
}
