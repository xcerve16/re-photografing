/**
 * PnPProblem.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef PNPPROBLEM_H_
#define PNPPROBLEM_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ModelRegistration.h"

/**
 * PnP metoda
 * Dle tutorialu dostupneho na http://docs.opencv.org/3.1.0/dc/d2c/tutorial_real_time_pose.html
 */
class PnPProblem {

private:

    cv::Mat _camera_matrix;

    cv::Mat _rotation_matrix;

    cv::Mat _translation_matrix;

    cv::Mat _projection_matrix;

    std::vector<cv::Point2f> _inliers_points;

public:
    PnPProblem() {
        _camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
        _rotation_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
        _translation_matrix = cv::Mat::zeros(3, 1, CV_64FC1);
        _projection_matrix = cv::Mat::zeros(4, 4, CV_64FC1);
    };

    ~PnPProblem() {}

    void estimatePoseRANSAC(const std::vector<cv::Point3f> &list_3d_points,
                            const std::vector<cv::Point2f> &list_2d_points, int flags, bool use_extrinsic_guess,
                            int iterations_count, float reprojection_error, double confidence);

    cv::Mat getCameraMatrix() const { return _camera_matrix; }

    cv::Mat getRotationMatrix() const { return _rotation_matrix; }

    cv::Mat getTranslationMatrix() const { return _translation_matrix; }

    cv::Mat getProjectionMatrix() const { return _projection_matrix; }

    std::vector<cv::Point2f> getInliersPoints() const { return _inliers_points; }

    void setProjectionMatrix(const cv::Mat &rotation_matrix, const cv::Mat &translation_matrix);

    void setCameraMatrix(const cv::Mat &camera_matrix) { _camera_matrix = camera_matrix; };

    void setOpticalCenter(double cx, double cy);

    void setCameraParameter(double cx, double cy, double fx, double fy);
};


#endif
