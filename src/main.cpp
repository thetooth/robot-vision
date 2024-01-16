#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/objdetect/aruco_detector.hpp>

#include <openvino/openvino.hpp>

#include <chrono>

#include "camera.hpp"
#include "nats.h"
#include "nc.hpp"
#include "spdlog/spdlog.h"
#include "yolov8.hpp"

using json = nlohmann::json;
using namespace std::chrono_literals;

int main()
{
    // Communications
    natsConnection *nc = nullptr;
    auto ncStatus = natsConnection_ConnectTo(&nc, "nats://192.168.0.107:4222");
    if (ncStatus != NATS_OK)
    {
        spdlog::error("NATS connection failure: {}", natsStatus_GetText(ncStatus));
        return 1;
    }

    // AI Core
    Config config = {0.2, 0.4, 0.4, 640, 640, "yolov8n_quantized.xml"};
    YOLOV8 inference(config);

    // Camera
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    inputVideo.set(cv::CAP_PROP_FPS, 60);
    // inputVideo.set(cv::CAP_PROP_CONVERT_RGB, false);
    // cv::Mat cameraMatrix, distCoeffs;
    // float markerLength = 0.05;

    // // Camera calibration
    // if (!Camera::readCameraParameters("camera.yml", cameraMatrix, distCoeffs))
    // {
    //     spdlog::error("Invalid camera.yml file");
    //     return 1;
    // }

    // // Aruco fiducial marker
    // cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    // cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    // cv::aruco::ArucoDetector markerDetector(dictionary, detectorParams);

    // // Set coordinate system
    // cv::Mat objPoints(4, 1, CV_32FC3);
    // objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    // objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    // objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    // objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

    cv::namedWindow("out", cv::WINDOW_NORMAL);
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    auto start = 0ns;
    auto end = 0ns;

    while (inputVideo.grab())
    {
        start = std::chrono::high_resolution_clock::now().time_since_epoch();

        cv::Mat image, out;
        inputVideo.retrieve(image);
        // cv::resize(image, out, cv::Size(640, 640));
        image.copyTo(out);

        auto detections = inference.detect(out);
        for (auto &&detection : detections)
        {
            if (detection.confidence < 0.75)
            {
                continue;
            }
            if (coconame[detection.class_id] != "cup")
            {
                continue;
            }
            auto box = detection.box;
            NC::Pose pose;
            pose.x = -box.x / 2 + box.width / 2;
            pose.y = box.y + box.height / 2;

            json j = {{"command", "goto"}, {"pose", pose}};
            std::string msg = j.dump();

            // spdlog::info("Sending: {}", msg);

            natsStatus pubStatus = natsConnection_PublishString(nc, "motion.command", msg.c_str());
            if (pubStatus != NATS_OK)
            {
                spdlog::error("NATS publish failure: {}", natsStatus_GetText(pubStatus));
            }
            inference.draw_detection(out, detection);
        }

        // std::vector<int> markerIds;
        // std::vector<std::vector<cv::Point2f>> markerCorners;
        // markerDetector.detectMarkers(out, markerCorners, markerIds);
        // // If at least one marker detected
        // if (markerIds.size() > 0)
        // {
        //     cv::aruco::drawDetectedMarkers(out, markerCorners, markerIds);
        //     int nMarkers = markerCorners.size();
        //     std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        //     // Calculate pose for each marker
        //     for (int i = 0; i < nMarkers; i++)
        //     {
        //         solvePnP(objPoints, markerCorners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
        //     }
        //     // Draw axis for each marker
        //     for (unsigned int i = 0; i < markerIds.size(); i++)
        //     {
        //         cv::drawFrameAxes(out, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
        //     }
        // }

        end = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cv::rectangle(out, cv::Point(0, 0), cv::Point(100, 50), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(out, std::to_string(elapsed.count()) + " ms", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 2);

        cv::imshow("out", out);
        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
}
