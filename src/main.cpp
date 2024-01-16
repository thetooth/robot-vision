#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

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
    if (!Camera::readCameraParameters("camera2.yml", cameraMatrix, distCoeffs))
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
    float boardLength = 37;
    cv::Mat boardPoints(4, 1, CV_32FC3);
    boardPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-boardLength / 2.f, boardLength / 2.f, 0);
    boardPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(boardLength / 2.f, boardLength / 2.f, 0);
    boardPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(boardLength / 2.f, -boardLength / 2.f, 0);
    boardPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-boardLength / 2.f, -boardLength / 2.f, 0);
    float targetLength = 53;
    cv::Mat targetPoints(4, 1, CV_32FC3);
    targetPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-targetLength / 2.f, targetLength / 2.f, 0);
    targetPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(targetLength / 2.f, targetLength / 2.f, 0);
    targetPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(targetLength / 2.f, -targetLength / 2.f, 0);
    targetPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-targetLength / 2.f, -targetLength / 2.f, 0);

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
        // cv::resize(image, out, cv::Size(640, 480));
        image.copyTo(out);
        cameraTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start;
        markerTime = std::chrono::high_resolution_clock::now().time_since_epoch() - start - cameraTime - inferenceTime;

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        // cv::Mat out;
        // cv::threshold(out, out, 100, 255, cv::THRESH_TRUNC);
        markerDetector.detectMarkers(out, markerCorners, markerIds);
        // If at least one marker detected
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(out, markerCorners, markerIds);
            int nMarkers = markerCorners.size();
            std::vector<cv::Mat> rvecs(nMarkers), tvecs(nMarkers);
            cv::Mat refRvec, refTvec, targetRvec, targetTvec;
            std::vector<cv::Point2f> refCorners, targetCorners;
            bool haveRef = false, haveTarget = false;

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

            if (boardIds.size() > 0)
            {
                // Get object and image points for the solvePnP function
                cv::Mat objPoints, imgPoints;
                board.matchImagePoints(boardCorners, boardIds, objPoints, imgPoints);
                cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, refRvec, refTvec);
                haveRef = true;
            }

            for (unsigned int i = 0; i < markerIds.size(); i++)
            {
                if (markerIds[i] == 248)
                {
                    cv::solvePnP(targetPoints, markerCorners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i]);
                }
                else
                {
                    cv::solvePnP(boardPoints, markerCorners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i]);
                }
                // cv::solvePnPRefineLM(objPoints, markerCorners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i]);

                // if (markerIds[i] == 8)
                // {
                //     refRvec = rvecs[i];
                //     refTvec = tvecs[i];
                //     refCorners = markerCorners[i];
                //     haveRef = true;
                // }
                if (markerIds[i] == 248)
                {
                    targetRvec = rvecs[i];
                    targetTvec = tvecs[i];
                    targetCorners = markerCorners[i];
                    haveTarget = true;
                }
            }
            if (markerIds.size() > 1 && haveRef && haveTarget)
            {
                // Get rotation matrix from rotation vectors
                cv::Mat R0;
                cv::Rodrigues(refRvec, R0);
                cv::Mat R248;
                cv::Rodrigues(targetRvec, R248);

                // Calculate the relative translation and rotation between the markers
                cv::Mat relativeTvec = R0.t() * (targetTvec - refTvec);
                cv::Mat relativeRvec;
                cv::Rodrigues(R0.t() * R248, relativeRvec);

                // spdlog::info("Relative position: ({}, {}, {})", relativeTvec.at<double>(0),
                // relativeTvec.at<double>(1),
                //  relativeTvec.at<double>(2));

                NC::Pose pose;
                pose.x = relativeTvec.at<double>(0) - markerLength;
                pose.y = -relativeTvec.at<double>(1);
                pose.z = relativeTvec.at<double>(2);

                spdlog::info("Relative position: ({}, {}, {})", pose.x, pose.y, pose.z);

                json j = {{"command", "goto"}, {"pose", pose}};
                std::string msg = j.dump();

                natsStatus pubStatus = natsConnection_PublishString(nc, "motion.command", msg.c_str());
                if (pubStatus != NATS_OK)
                {
                    spdlog::error("NATS publish failure: {}", natsStatus_GetText(pubStatus));
                }
            }
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
