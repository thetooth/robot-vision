#include <stdio.h>
#include <string>
#include <vector>

#include <limits>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/opencv.hpp>

namespace Camera
{
    std::vector<cv::Point3f> getCornersInCameraWorld(double side, cv::Vec3d rvec, cv::Vec3d tvec);
    void flatten(cv::Mat &out, std::vector<int> markerIds, std::vector<std::vector<cv::Point2f>> markerCorners);
} // namespace Camera
