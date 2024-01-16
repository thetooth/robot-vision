#include <ctime>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

namespace Camera
{
    inline static bool readCameraParameters(std::string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;
        fs["camera_matrix"] >> camMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
        return true;
    }

    inline static bool saveCameraParams(const std::string &filename, cv::Size imageSize, float aspectRatio, int flags,
                                        const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, double totalAvgErr)
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened())
            return false;

        time_t tt;
        time(&tt);
        struct tm *t2 = localtime(&tt);
        char buf[1024];
        strftime(buf, sizeof(buf) - 1, "%c", t2);

        fs << "calibration_time" << buf;
        fs << "image_width" << imageSize.width;
        fs << "image_height" << imageSize.height;

        if (flags & cv::CALIB_FIX_ASPECT_RATIO)
            fs << "aspectRatio" << aspectRatio;

        if (flags != 0)
        {
            sprintf(buf, "flags: %s%s%s%s", flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                    flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                    flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                    flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        }
        fs << "flags" << flags;
        fs << "camera_matrix" << cameraMatrix;
        fs << "distortion_coefficients" << distCoeffs;
        fs << "avg_reprojection_error" << totalAvgErr;
        return true;
    }

    inline static cv::Mat getRelativeMarkerPosition(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                                    const std::vector<cv::Point2f> &imagePoints0,
                                                    const std::vector<cv::Point2f> &imagePoints248,
                                                    const cv::Mat &rvec0, const cv::Mat &tvec0)
    {

        // Define 3D coordinates of markers in the physical world
        std::vector<cv::Point3f> objectPoints;
        objectPoints.push_back(cv::Point3f(0.0, 0.0, 0.0)); // Marker ID 0, the fixed marker
        objectPoints.push_back(cv::Point3f(1.0, 0.0, 0.0)); // Marker ID 248, the relative marker

        // Convert rotation vector of the fixed marker to rotation matrix
        cv::Mat R0;
        cv::Rodrigues(rvec0, R0);

        // Project the 3D points of the markers into image space for both markers
        std::vector<cv::Point2f> projectedPoints0, projectedPoints248;
        cv::projectPoints(objectPoints, rvec0, tvec0, cameraMatrix, distCoeffs, projectedPoints0);

        // Solve PnP for the relative marker (ID 248)
        cv::Mat rvec248, tvec248;
        cv::solvePnP(objectPoints, imagePoints248, cameraMatrix, distCoeffs, rvec248, tvec248);

        // Convert the rotation vector to a rotation matrix
        cv::Mat R248;
        cv::Rodrigues(rvec248, R248);

        // Calculate the relative translation and rotation between the markers
        cv::Mat relativeTvec = R0.t() * (tvec248 - tvec0);
        cv::Mat relativeRvec;
        cv::Rodrigues(R0.t() * R248, relativeRvec);

        return relativeTvec;
    }

    inline static cv::Mat inversePerspective(const cv::Vec3d rvec, const cv::Vec3d tvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        R = R.t();
        cv::Mat invTvec = -R * tvec;
        cv::Mat invRvec;
        cv::Rodrigues(R, invRvec);
        return invRvec;
    }

    inline static std::pair<cv::Mat, cv::Mat> relativePosition(const cv::Vec3d rvec1, const cv::Vec3d tvec1,
                                                               const cv::Vec3d rvec2, const cv::Vec3d tvec2)
    {
        // cv::Mat rvec1_reshape = rvec1.reshape(3, 1);
        // cv::Mat tvec1_reshape = tvec1.reshape(3, 1);
        // cv::Mat rvec2_reshape = rvec2.reshape(3, 1);
        // cv::Mat tvec2_reshape = tvec2.reshape(3, 1);

        // Inverse the second marker, the right one in the image
        cv::Mat invRvec = inversePerspective(rvec2, tvec2);
        cv::Mat invTvec = (-invRvec).t() * tvec2;

        cv::Mat orgRvec = inversePerspective(invRvec, invTvec);

        cv::Mat composedRvec, composedTvec;
        cv::composeRT(rvec1, tvec1, invRvec, -invRvec * tvec2, composedRvec, composedTvec);

        composedRvec = composedRvec.reshape(3, 1);
        composedTvec = composedTvec.reshape(3, 1);

        return std::make_pair(composedRvec, composedTvec);
    }

} // namespace Camera
