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

    inline static cv::Mat inversePerspective(const cv::Mat &rvec, const cv::Mat &tvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        R = R.t();
        cv::Mat invTvec = -R * tvec;
        cv::Mat invRvec;
        cv::Rodrigues(R, invRvec);
        return invRvec;
    }

    inline static std::pair<cv::Mat, cv::Mat> relativePosition(const cv::Mat &rvec1, const cv::Mat &tvec1,
                                                               const cv::Mat &rvec2, const cv::Mat &tvec2)
    {
        cv::Mat rvec1_reshape = rvec1.reshape(3, 1);
        cv::Mat tvec1_reshape = tvec1.reshape(3, 1);
        cv::Mat rvec2_reshape = rvec2.reshape(3, 1);
        cv::Mat tvec2_reshape = tvec2.reshape(3, 1);

        // Inverse the second marker, the right one in the image
        cv::Mat invRvec = inversePerspective(rvec2_reshape, tvec2_reshape);

        cv::Mat orgRvec = inversePerspective(invRvec, -invRvec * tvec2_reshape);

        cv::Mat composedRvec, composedTvec;
        cv::composeRT(rvec1_reshape, tvec1_reshape, invRvec, -invRvec * tvec2_reshape, composedRvec, composedTvec);

        composedRvec = composedRvec.reshape(3, 1);
        composedTvec = composedTvec.reshape(3, 1);

        return std::make_pair(composedRvec, composedTvec);
    }

} // namespace Camera
