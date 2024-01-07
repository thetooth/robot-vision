
#include "flatten.hpp"

std::vector<cv::Point3f> Camera::getCornersInCameraWorld(double side, cv::Vec3d rvec, cv::Vec3d tvec)
{
    double half_side = side / 2;

    // compute rot_mat
    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);

    // transpose of rot_mat for easy columns extraction
    cv::Mat rot_mat_t = rot_mat.t();

    // the two E-O and F-O vectors
    double *tmp = rot_mat_t.ptr<double>(0);
    cv::Point3f camWorldE(tmp[0] * half_side, tmp[1] * half_side, tmp[2] * half_side);

    tmp = rot_mat_t.ptr<double>(1);
    cv::Point3f camWorldF(tmp[0] * half_side, tmp[1] * half_side, tmp[2] * half_side);

    // convert tvec to point
    cv::Point3f tvec_3f(tvec[0], tvec[1], tvec[2]);

    // return vector:
    std::vector<cv::Point3f> ret(4, tvec_3f);

    ret[0] += camWorldE + camWorldF;
    ret[1] += -camWorldE + camWorldF;
    ret[2] += -camWorldE - camWorldF;
    ret[3] += camWorldE - camWorldF;

    return ret;
}

void Camera::flatten(cv::Mat &out, std::vector<int> markerIds, std::vector<std::vector<cv::Point2f>> markerCorners)
{
    std::array<cv::Point2f, 4> srcCorners; // corner that we want
    std::array<cv::Point2f, 4> srcCornersSmall;
    std::array<cv::Point2f, 4> dstCorners; // destination corner

    // id  8 14 18 47
    for (size_t i = 0; i < markerIds.size(); i++)
    {
        // first corner
        if (markerIds[i] == 8)
        {
            srcCorners[0] = markerCorners[i][0]; // get the first point
            srcCornersSmall[0] = markerCorners[i][2];
        }
        // second corner
        else if (markerIds[i] == 16)
        {
            srcCorners[1] = markerCorners[i][1]; // get the second point
            srcCornersSmall[1] = markerCorners[i][3];
        }
        // third corner
        else if (markerIds[i] == 32)
        {
            srcCorners[2] = markerCorners[i][2]; // get the thirt point
            srcCornersSmall[2] = markerCorners[i][0];
        }
        // fourth corner
        else if (markerIds[i] == 24)
        {
            srcCorners[3] = markerCorners[i][3]; // get the fourth point
            srcCornersSmall[3] = markerCorners[i][1];
        }
    }

    float scale = 0.5;
    dstCorners[0] = cv::Point2f(0.0f, 0.0f);
    dstCorners[1] = cv::Point2f(640.0f * scale, 0.0f);
    dstCorners[2] = cv::Point2f(640.0f * scale, 640.0f * scale);
    dstCorners[3] = cv::Point2f(0.0f, 640.0f * scale);

    // get perspectivetransform
    cv::Mat M = getPerspectiveTransform(srcCorners, dstCorners);
    cv::warpPerspective(out, out, M, cv::Size(640, 480));
}