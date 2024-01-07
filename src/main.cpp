#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <openvino/openvino.hpp>

#include <chrono>

#include "camera.hpp"
#include "flatten.hpp"
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
    // Config config = {0.2, 0.0, 0.0, 640, 640, "best.onnx"};
    // YOLOV8 inference(config);

    // Camera
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    inputVideo.set(cv::CAP_PROP_FPS, 30);
    // inputVideo.set(cv::CAP_PROP_CONVERT_RGB, false);
    cv::Mat cameraMatrix, distCoeffs;
    float markerLength = 37.0;
    float markerSeparation = 4.0;
    int markersX = 5;
    int markersY = 7;

    // Camera calibration
    if (!Camera::readCameraParameters("camera.yml", cameraMatrix, distCoeffs))
    {
        spdlog::error("Invalid camera.yml file");
        return 1;
    }

    // Aruco fiducial marker
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector markerDetector(dictionary, detectorParams);
    cv::aruco::GridBoard board(cv::Size(markersX, markersY), float(markerLength), float(markerSeparation), dictionary);

    // Generate markers
    for (int i = 1; i <= 6; i++)
    {
        cv::Mat markerImage;
        auto id = 200 + (i * 8);
        cv::aruco::generateImageMarker(dictionary, id, 200, markerImage, 1);
        cv::imwrite(fmt::format("marker{}.png", id), markerImage);
    }

    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

    cv::namedWindow("out", cv::WINDOW_NORMAL);
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    auto start = 0ns;
    auto cameraTime = 0ns;
    auto inferenceTime = 0ns;
    auto markerTime = 0ns;
    auto arucoTime = 0ns;

    while (inputVideo.grab())
    {
        start = std::chrono::high_resolution_clock::now().time_since_epoch();

        cv::Mat image, out;
        inputVideo.retrieve(image);
        // cv::resize(image, out, cv::Size(640, 640));
        image.copyTo(out);
        cameraTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start;

        // auto detections = inference.detect(out);
        // inferenceTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start - cameraTime;

        // for (auto &&detection : detections)
        // {
        //     if (detection.confidence < 0.75)
        //     {
        //         // continue;
        //     }
        //     // if (coconame[detection.class_id] != "cup")
        //     // {
        //     //     continue;
        //     // }
        //     auto box = detection.box;
        //     NC::Pose pose;
        //     pose.x = -box.x / 2 + box.width / 2;
        //     pose.y = box.y + box.height / 2;

        //     json j = {{"command", "goto"}, {"pose", pose}};
        //     std::string msg = j.dump();

        //     // spdlog::info("Sending: {}", msg);

        //     natsStatus pubStatus = natsConnection_PublishString(nc, "motion.command", msg.c_str());
        //     if (pubStatus != NATS_OK)
        //     {
        //         spdlog::error("NATS publish failure: {}", natsStatus_GetText(pubStatus));
        //     }
        //     inference.draw_detection(out, detection);
        // }
        markerTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start - cameraTime - inferenceTime;

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::Mat gray;
        cv::threshold(out, gray, 100, 255, cv::THRESH_BINARY);
        markerDetector.detectMarkers(gray, markerCorners, markerIds);
        // If at least one marker detected
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(out, markerCorners, markerIds);
            // int nMarkers = markerCorners.size();
            // std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

            // Filter board markers
            std::vector<int> boardIds;
            std::vector<std::vector<cv::Point2f>> boardCorners;
            for (size_t i = 0; i < markerIds.size(); i++)
            {
                if (markerIds[i] < 200)
                {
                    boardIds.push_back(markerIds[i]);
                    boardCorners.push_back(markerCorners[i]);
                }
            }

            cv::Vec3d rvec, tvec;
            if (boardIds.size() > 0)
            {
                // Get object and image points for the solvePnP function
                cv::Mat objPoints, imgPoints;
                board.matchImagePoints(boardCorners, boardIds, objPoints, imgPoints);
                // Find pose
                cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
                // If at least one board marker detected
                auto markersOfBoardDetected = (int)objPoints.total() / 4;
                if (markersOfBoardDetected > 0)
                {
                    cv::drawFrameAxes(out, cameraMatrix, distCoeffs, rvec, tvec, 100);
                }
            }

            // // Calculate pose for each marker
            // for (int i = 0; i < nMarkers; i++)
            // {
            //     solvePnP(objPoints, markerCorners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
            // }

            // Draw axis for each marker
            for (unsigned int i = 0; i < markerIds.size(); i++)
            {
                if (markerIds[i] == 248)
                {
                    cv::Mat R;
                    cv::Rodrigues(rvec, R);
                    R = R.t();
                    cv::Mat tvec2 = -R * cv::Mat(tvec);
                    // cv::Rodrigues(R, rvec);

                    cv::Mat loc = tvec2 * cv::Mat(markerCorners[i]);

                    NC::Pose pose;
                    // pose.x = tvec2.at<double>(0) - markerLength / 2;
                    // pose.y = tvec2.at<double>(1) + markerLength / 2;
                    pose.x = loc.at<double>(0);
                    pose.y = loc.at<double>(1);

                    spdlog::info("Marker {} at ({}, {})", markerIds[i], pose.x, pose.y);

                    json j = {{"command", "goto"}, {"pose", pose}};
                    std::string msg = j.dump();

                    natsStatus pubStatus = natsConnection_PublishString(nc, "motion.command", msg.c_str());
                    if (pubStatus != NATS_OK)
                    {
                        spdlog::error("NATS publish failure: {}", natsStatus_GetText(pubStatus));
                    }
                }
            }

            // Flatten image
            // if (nMarkers >= 4)
            // {
            //     Camera::flatten(out, markerIds, markerCorners);
            // }
        }
        arucoTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start - cameraTime - inferenceTime -
                    markerTime;

        auto str = fmt::format("Camera: {}ms Inf: {}ms Marker: {}ms Aruco: {}ms",
                               std::chrono::duration_cast<std::chrono::milliseconds>(cameraTime).count(),
                               std::chrono::duration_cast<std::chrono::milliseconds>(inferenceTime).count(),
                               std::chrono::duration_cast<std::chrono::milliseconds>(markerTime).count(),
                               std::chrono::duration_cast<std::chrono::milliseconds>(arucoTime).count());

        cv::rectangle(out, cv::Point(0, 0), cv::Point(640, 20), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(out, str, cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

        cv::imshow("out", out);
        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
}
