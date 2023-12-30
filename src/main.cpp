#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <openvino/openvino.hpp>

#include "camera.hpp"
#include "nats.h"
#include "nc.hpp"
#include "spdlog/spdlog.h"
#include "yolov8.hpp"

using json = nlohmann::json;

int main()
{
    // Communications
    natsConnection *nc = nullptr;
    auto ncStatus = natsConnection_ConnectTo(&nc, NATS_DEFAULT_URL);
    if (ncStatus != NATS_OK)
    {
        spdlog::error("NATS connection failure: {}", natsStatus_GetText(ncStatus));
        return 1;
    }

    // AI Core
    Config config = {0.2, 0.4, 0.4, 640, 640, "yolov8s.onnx"};
    YOLOV8 inference(config);

    // Camera
    cv::VideoCapture inputVideo;
    inputVideo.open(2);
    inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cv::Mat cameraMatrix, distCoeffs;
    float markerLength = 0.05;

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

    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

    while (inputVideo.grab())
    {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        // cv::resize(image, imageCopy, cv::Size(640, 640));
        image.copyTo(imageCopy);

        auto detections = inference.detect(imageCopy);
        for (auto &&detection : detections)
        {
            if (detection.confidence < 0.75 || coconame[detection.class_id] != "cup")
            {
                continue;
            }
            auto box = detection.box;
            NC::Pose pose;
            pose.x = -box.x + box.width / 2;
            pose.y = box.y + box.height / 2;

            json j = {{"command", "goto"}, {"pose", pose}};
            std::string msg = j.dump();
            natsStatus pubStatus = natsConnection_PublishString(nc, "motion.command", msg.c_str());
            if (pubStatus != NATS_OK)
            {
                spdlog::error("NATS publish failure: {}", natsStatus_GetText(pubStatus));
            }
            inference.draw_detection(imageCopy, detection);
        }

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        markerDetector.detectMarkers(imageCopy, markerCorners, markerIds);
        // If at least one marker detected
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
            int nMarkers = markerCorners.size();
            std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
            // Calculate pose for each marker
            for (int i = 0; i < nMarkers; i++)
            {
                solvePnP(objPoints, markerCorners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
            }
            // Draw axis for each marker
            for (unsigned int i = 0; i < markerIds.size(); i++)
            {
                cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
            }
        }

        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
}
