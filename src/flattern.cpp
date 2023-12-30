#include <stdio.h>
#include <string>
#include <vector>

#include <aruco/aruco.h>
#include <limits>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace aruco;

Scalar red(0, 0, 255);
Scalar green(0, 200, 0);
Scalar blue(255, 0, 0);
Scalar purple(255, 0, 255);

void drawMarkers()
{
}

void getPlanePoints(const InputArray &imagePoints, const Mat rvec, const Mat tvec, const InputArray &cameraMatrix,
                    const InputArray &distortionMatrix, vector<Point3f> &planePoints)
{
    Mat tvec_t;
    Mat rotMatInv;
    vector<Point2f> worldPoints2f;

    transpose(tvec, tvec_t);
    Rodrigues(rvec, rotMatInv);
    transpose(rotMatInv, rotMatInv); // Actual inversion of rotation matrix

    undistortPoints(imagePoints, worldPoints2f, cameraMatrix, distortionMatrix);

    Mat_<float> T = rotMatInv * -tvec_t;

    for (auto _point : worldPoints2f)
    {
        Mat_<float> point = Mat::ones(3, 1, DataType<float>::type);
        point(0) = _point.x;
        point(1) = _point.y;

        Mat_<float> d = rotMatInv * point;

        auto scaling = -T(2) / d(2);                                              // -Tz / dz;
        planePoints.push_back({scaling * d(0) + T(0), scaling * d(1) + T(1), 0}); // scaling * dx + Tx,
    }
}

void getAABB(const vector<Point3f> &points, vector<Point3f> &aabb)
{
    aabb.clear();
    aabb.push_back({numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max()});
    aabb.push_back({numeric_limits<float>::lowest(), numeric_limits<float>::lowest(), numeric_limits<float>::lowest()});
    Point3f &mins = aabb[0];
    Point3f &maxes = aabb[1];

    for (auto i : points)
    {
        mins.x = i.x < mins.x ? i.x : mins.x;
        mins.y = i.y < mins.y ? i.y : mins.y;
        mins.z = i.z < mins.z ? i.z : mins.z;
        maxes.x = i.x > maxes.x ? i.x : maxes.x;
        maxes.y = i.y > maxes.y ? i.y : maxes.y;
        maxes.z = i.z > maxes.z ? i.z : maxes.z;
    }
}

void test()
{
    CameraParameters camP;
    MarkerMap mMap;
    MarkerDetector mDet;
    MarkerMapPoseTracker poseTracker;
    Mat img;
    VideoCapture vCap;
    bool singleImage = false;

    // Matrix to flip along x axis
    Vec<float, 3> flipX(3.14159, 0, 0);
    Mat flipXmat;
    Rodrigues(flipX, flipXmat);

    vCap.open(0);
    cv::namedWindow("in", 1);
    cv::namedWindow("out", 1);

    mMap.readFromFile("../test_data/map.yml");
    camP.readFromXMLFile("../test_data/teck.yml");
    mDet.setDictionary("ARUCO");
    poseTracker.setParams(camP, mMap);

    if (singleImage)
        img = imread("../test_data/board1.jpg");

    while (true)
    {

        if (!singleImage)
            vCap >> img;

        vector<aruco::Marker> detected = mDet.detect(img);

#if 0
        for (auto marker: detected)
            marker.draw(img, Scalar(0,0,255), 2);
#endif

        if (poseTracker.isValid())
        {
            if (poseTracker.estimatePose(detected))
            {
                // aruco::CvDrawingUtils::draw3dAxis(img,
                // camP,poseTracker.getRvec(),poseTracker.getTvec(),mMap[0].getMarkerSize()*2);
                const Mat &tvec = poseTracker.getTvec();
                const Mat &rvec = poseTracker.getRvec();
                Mat vtvec;
                transpose(tvec, vtvec);
                cout << "tvec " << tvec << " rvec " << rvec << endl;
                cout << endl;

                {
                    vector<Point2f> imagePoints = {
                        {0, 0},
                        {(float)img.cols, 0},
                        {(float)img.cols, (float)img.rows},
                        {0, (float)img.rows},
                    };
                    vector<Point3f> planePoints;

                    getPlanePoints(imagePoints, rvec, tvec, camP.CameraMatrix, camP.Distorsion, planePoints);

                    for (auto i : imagePoints)
                        cout << "IP " << i << endl;

                    for (auto i : planePoints)
                        cout << "PP " << i << endl;

                    transform(planePoints, planePoints, flipXmat);
#if 0
                    // check reverse projection
                    vector <Point2f> imagePointsAgain;

                    projectPoints(planePoints, rvec, tvec, camP.CameraMatrix, camP.Distorsion, imagePointsAgain);

                    line(img, imagePointsAgain[0], imagePointsAgain[1], red,    2);
                    line(img, imagePointsAgain[1], imagePointsAgain[2], green,  2);
                    line(img, imagePointsAgain[2], imagePointsAgain[3], blue,   2);
                    line(img, imagePointsAgain[3], imagePointsAgain[0], purple, 2);
#endif

                    vector<Point3f> aabb;
                    getAABB(planePoints, aabb);
                    for (auto i : aabb)
                        cout << "aabb " << i << endl;

                    auto w = aabb[1].x - aabb[0].x;
                    auto h = aabb[1].y - aabb[0].y;
                    auto hScale = img.cols / w;
                    auto vScale = img.rows / h;

                    vector<Point2f> imagePoints3;

                    for (auto &i : planePoints)
                        imagePoints3.push_back({i.x, i.y});

                    auto scale = w > h ? hScale : vScale;
                    Point2f offset = {aabb[0].x, aabb[0].y};
                    offset *= -scale;

                    cout << "offset " << offset << endl;

                    for (auto &i : imagePoints3)
                    {
                        i *= scale;
                        i += offset;
                    }

                    for (auto i : imagePoints3)
                        cout << "scaled " << i << endl;

#if 0
                    line(img, imagePoints3[0], imagePoints3[1], red,    2);
                    line(img, imagePoints3[1], imagePoints3[2], green,  2);
                    line(img, imagePoints3[2], imagePoints3[3], blue,   2);
                    line(img, imagePoints3[3], imagePoints3[0], purple, 2);
#endif

                    Mat pxform = getPerspectiveTransform(imagePoints, imagePoints3);
                    Mat outImg;
                    warpPerspective(img, outImg, pxform, Size(img.cols, img.rows));
                    imshow("out", outImg);

#if 1
                    // Show Z-axis rotation
                    Mat rotMat;
                    Mat matR, matQ;
                    Rodrigues(rvec, rotMat);
                    /*
                    Vec3d euler = RQDecomp3x3(rotMat, matR, matQ);
                    cout << "euler " << euler << endl;
                    */

                    Mat rot = Mat(rotMat, Rect(0, 0, 2, 2));
                    cout << "rot " << rot << endl;
                    Mat q(1, 2, DataType<float>::type);
                    q.at<float>(1) = 100; // Point at 0, 100

                    q *= rot;

                    cout << "Q " << q << endl;
                    Point2f center(img.cols / 2, img.rows / 2);

                    line(img, center, Point2f(center.x + q.at<float>(0), center.y + q.at<float>(1)), green, 2);

#endif
                }

            } // if pose detected
            // break;

            cv::imshow("in", img);

            char key = cv::waitKey(singleImage ? 0 : 1);
            if (key == 27)
                break;
            if (key == 32)
            {
                cv::waitKey(0);
            }

        } // pose trackeris valid
    }
}

int main(int argc, char **argv)
{
    test();
    return 0;
}
