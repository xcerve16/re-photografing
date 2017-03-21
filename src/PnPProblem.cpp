/*
 * PnPProblem.cpp
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#include <iostream>
#include <sstream>

#include "PnPProblem.h"

#include <opencv2/calib3d/calib3d.hpp>

/* Functions headers */
cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2);

double DOT(cv::Point3f v1, cv::Point3f v2);

cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2);

cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin);


/* Functions for Möller–Trumbore intersection algorithm */

cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2) {
    cv::Point3f tmp_p;
    tmp_p.x = v1.y * v2.z - v1.z * v2.y;
    tmp_p.y = v1.z * v2.x - v1.x * v2.z;
    tmp_p.z = v1.x * v2.y - v1.y * v2.x;
    return tmp_p;
}

double DOT(cv::Point3f v1, cv::Point3f v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2) {
    cv::Point3f tmp_p;
    tmp_p.x = v1.x - v2.x;
    tmp_p.y = v1.y - v2.y;
    tmp_p.z = v1.z - v2.z;
    return tmp_p;
}

/* End functions for Möller–Trumbore intersection algorithm
 *  */

// Function to get the nearest 3D point to the Ray origin
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin) {
    cv::Point3f p1 = points_list[0];
    cv::Point3f p2 = points_list[1];

    double d1 = std::sqrt(std::pow(p1.x - origin.x, 2) + std::pow(p1.y - origin.y, 2) + std::pow(p1.z - origin.z, 2));
    double d2 = std::sqrt(std::pow(p2.x - origin.x, 2) + std::pow(p2.y - origin.y, 2) + std::pow(p2.z - origin.z, 2));

    if (d1 < d2) {
        return p1;
    } else {
        return p2;
    }
}

// Custom constructor given the intrinsic camera parameters

PnPProblem::PnPProblem(const double params[]) {
    _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    _A_matrix.at<double>(0, 0) = params[0];       //      [ fx   0  cx ]
    _A_matrix.at<double>(1, 1) = params[1];       //      [  0  fy  cy ]
    _A_matrix.at<double>(0, 2) = params[2];       //      [  0   0   1 ]
    _A_matrix.at<double>(1, 2) = params[3];
    _A_matrix.at<double>(2, 2) = 1;
    _R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    _T_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
    _P_matrix = cv::Mat::zeros(4, 4, CV_64FC1);   // rotation-translation matrix

}

void PnPProblem::set_P_matrix(const cv::Mat &R_matrix, const cv::Mat &t_matrix) {
    // Rotation-Translation Matrix Definition
    _P_matrix.at<double>(0, 0) = R_matrix.at<double>(0, 0);
    _P_matrix.at<double>(0, 1) = R_matrix.at<double>(0, 1);
    _P_matrix.at<double>(0, 2) = R_matrix.at<double>(0, 2);
    _P_matrix.at<double>(1, 0) = R_matrix.at<double>(1, 0);
    _P_matrix.at<double>(1, 1) = R_matrix.at<double>(1, 1);
    _P_matrix.at<double>(1, 2) = R_matrix.at<double>(1, 2);
    _P_matrix.at<double>(2, 0) = R_matrix.at<double>(2, 0);
    _P_matrix.at<double>(2, 1) = R_matrix.at<double>(2, 1);
    _P_matrix.at<double>(2, 2) = R_matrix.at<double>(2, 2);
    _P_matrix.at<double>(0, 3) = t_matrix.at<double>(0);
    _P_matrix.at<double>(1, 3) = t_matrix.at<double>(1);
    _P_matrix.at<double>(2, 3) = t_matrix.at<double>(2);
    _P_matrix.at<double>(3, 0) = 0;
    _P_matrix.at<double>(3, 1) = 0;
    _P_matrix.at<double>(3, 2) = 0;
    _P_matrix.at<double>(3, 3) = 1;
}


// Estimate the pose given a list of 2D/3D correspondences and the method to use
bool
PnPProblem::estimatePose(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d,
                         int flags) {
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

    bool useExtrinsicGuess = false;

    // Pose estimation
    bool correspondence = cv::solvePnP(list_points3d, list_points2d, _A_matrix, distCoeffs, rvec, tvec,
                                       useExtrinsicGuess, flags);

    // Transforms Rotation Vector to Matrix
    Rodrigues(rvec, _R_matrix);
    _T_matrix = tvec;

    // Set projection matrix
    this->set_P_matrix(_R_matrix, _T_matrix);

    return correspondence;
}

// Estimate the pose given a list of 2D/3D correspondences with RANSAC and the method to use

void PnPProblem::estimatePoseRANSAC(const std::vector<cv::Point3f> &list_points3d, // list with model 3D coordinates
                                    const std::vector<cv::Point2f> &list_points2d,     // list with scene 2D coordinates
                                    int flags, cv::Mat &inliers, int iterationsCount,  // PnP method; inliers container
                                    float reprojectionError, double confidence)    // Ransac parameters
{
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    // output translation vector

    bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as
    // initial approximations of the rotation and translation vectors

    cv::solvePnPRansac(list_points3d, list_points2d, _A_matrix, distCoeffs, rvec, tvec,
                       useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                       inliers, flags);

    Rodrigues(rvec, _R_matrix);      // converts Rotation Vector to Matrix
    _T_matrix = tvec;       // set translation matrix

    this->set_P_matrix(_R_matrix, _T_matrix); // set rotation-translation matrix

}

void PnPProblem::mySolvePnPRansac(const std::vector<cv::Point3f> &list_points3d,
                                  const std::vector<cv::Point2f> &list_points2d,
                                  cv::Mat rvect, cv::Mat tvect) {

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    solvePnPRansac(list_points3d, list_points2d, _A_matrix, distCoeffs, rvect, tvect);

}

void PnPProblem::myProjectPoints(std::vector<cv::Point3f> list3DPoint, cv::Mat rvect, cv::Mat tvect,
                                 std::vector<cv::Point2f> list2DPoint) {
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    projectPoints(list3DPoint, rvect, tvect, _A_matrix, distCoeffs, list2DPoint);
}

// Backproject a 3D point to 2D using the estimated pose parameters

cv::Point2f PnPProblem::backproject3DPoint(const cv::Point3f &point3d) {
    // 3D point vector [x y z 1]'
    cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
    point3d_vec.at<double>(0) = point3d.x;
    point3d_vec.at<double>(1) = point3d.y;
    point3d_vec.at<double>(2) = point3d.z;
    point3d_vec.at<double>(3) = 1;

    // 2D point vector [u v 1]'
    cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
    point2d_vec = _A_matrix * _P_matrix * point3d_vec;

    // Normalization of [u v]'
    cv::Point2f point2d;
    point2d.x = (float) (point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
    point2d.y = (float) (point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

    return point2d;
}

void PnPProblem::setMatrixParam(double fx, double fy, double cx, double cy) {

    /*double fx = width * f / sx;
    double fy = height * f / sy;
    double cx = width / 2;
    double cy = height / 2;*/

    _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    _A_matrix.at<double>(0, 0) = fx;       //      [ fx   0  cx ]
    _A_matrix.at<double>(1, 1) = fy;       //      [  0  fy  cy ]
    _A_matrix.at<double>(0, 2) = cx;       //      [  0   0   1 ]
    _A_matrix.at<double>(1, 2) = cy;
    _A_matrix.at<double>(2, 2) = 1;
    _R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    _T_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
    _P_matrix = cv::Mat::zeros(4, 4, CV_64FC1);   // rotation-translation matrix

}


PnPProblem::PnPProblem() {

}

void PnPProblem::setOpticalCenter(double x, double y) {
    _A_matrix.at<double>(0, 2) = x;
    _A_matrix.at<double>(1, 2) = y;
}
