/**
 * PnPProblem.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include <iostream>
#include <sstream>

#include "PnPProblem.h"

#include <opencv2/calib3d/calib3d.hpp>

void PnPProblem::setProjectionMatrix(const cv::Mat &rotation_matrix, const cv::Mat &translation_matrix) {

    _projection_matrix.at<double>(0, 0) = rotation_matrix.at<double>(0, 0);
    _projection_matrix.at<double>(0, 1) = rotation_matrix.at<double>(0, 1);
    _projection_matrix.at<double>(0, 2) = rotation_matrix.at<double>(0, 2);
    _projection_matrix.at<double>(1, 0) = rotation_matrix.at<double>(1, 0);
    _projection_matrix.at<double>(1, 1) = rotation_matrix.at<double>(1, 1);
    _projection_matrix.at<double>(1, 2) = rotation_matrix.at<double>(1, 2);
    _projection_matrix.at<double>(2, 0) = rotation_matrix.at<double>(2, 0);
    _projection_matrix.at<double>(2, 1) = rotation_matrix.at<double>(2, 1);
    _projection_matrix.at<double>(2, 2) = rotation_matrix.at<double>(2, 2);
    _projection_matrix.at<double>(0, 3) = translation_matrix.at<double>(0);
    _projection_matrix.at<double>(1, 3) = translation_matrix.at<double>(1);
    _projection_matrix.at<double>(2, 3) = translation_matrix.at<double>(2);
    _projection_matrix.at<double>(3, 0) = 0;
    _projection_matrix.at<double>(3, 1) = 0;
    _projection_matrix.at<double>(3, 2) = 0;
    _projection_matrix.at<double>(3, 3) = 1;
}


void PnPProblem::estimatePoseRANSAC(const std::vector<cv::Point3f> &list_3d_points,
                                    const std::vector<cv::Point2f> &list_2d_points, int flags, bool use_extrinsic_guess,
                                    int iterations_count, float reprojection_error, double confidence) {

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat inliers_points;

    cv::solvePnPRansac(list_3d_points, list_2d_points, _camera_matrix, dist_coeffs, rvec, tvec, use_extrinsic_guess,
                       iterations_count, reprojection_error, confidence, inliers_points, flags);

    Rodrigues(rvec, _rotation_matrix);
    _translation_matrix = tvec;

    if (!inliers_points.empty()) {
        _inliers_points.clear();

        for (int index = 0; index < inliers_points.rows; ++index) {
            int n = inliers_points.at<int>(index);
            cv::Point2f point2d = list_2d_points[n];
            _inliers_points.push_back(point2d);
        }
    }

    this->setProjectionMatrix(_rotation_matrix, _translation_matrix);
}

void PnPProblem::setOpticalCenter(double cx, double cy) {
    _camera_matrix.at<double>(0, 2) = cx;
    _camera_matrix.at<double>(1, 2) = cy;

}

void PnPProblem::setCameraParameter(double cx, double cy, double fx, double fy) {
    _camera_matrix.at<double>(0, 2) = cx;
    _camera_matrix.at<double>(1, 2) = cy;
    _camera_matrix.at<double>(0, 0) = fx;
    _camera_matrix.at<double>(1, 1) = fy;
    _camera_matrix.at<double>(2, 2) = 1;

}
