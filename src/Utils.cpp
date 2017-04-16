/**
 * Utils.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include <iostream>

#include "PnPProblem.h"
#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


int lineType = 8;
int radius = 4;


void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color) {
    for (size_t i = 0; i < list_points.size(); i++) {
        cv::Point2f point_2d = list_points[i];
        cv::circle(image, point_2d, radius, color, -1, lineType);
    }
}

void draw2DPoint(cv::Mat image, cv::Point2f &points, cv::Scalar color) {
    cv::circle(image, points, radius, color, -1, lineType);
}


cv::Mat rot2euler(const cv::Mat &rotationMatrix) {

    cv::Mat euler(3, 1, CV_64F);

    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);

    double x, y, z;


    if (m10 > 0.998) {
        x = 0;
        y = CV_PI / 2;
        z = atan2(m02, m22);
    } else if (m10 < -0.998) {
        x = 0;
        y = -CV_PI / 2;
        z = atan2(m02, m22);
    } else {
        x = atan2(-m12, m11);
        y = asin(m10);
        z = atan2(-m20, m00);
    }

    euler.at<double>(0) = x;
    euler.at<double>(1) = y;
    euler.at<double>(2) = z;

    return euler;
}

cv::Mat euler2rot(const cv::Mat &euler) {

    cv::Mat rotationMatrix(3, 3, CV_64F);

    double x = euler.at<double>(0);
    double y = euler.at<double>(1);
    double z = euler.at<double>(2);

    double ch = cos(z);
    double sh = sin(z);
    double ca = cos(y);
    double sa = sin(y);
    double cb = cos(x);
    double sb = sin(x);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh * sb - ch * sa * cb;
    m02 = ch * sa * sb + sh * cb;
    m10 = sa;
    m11 = ca * cb;
    m12 = -ca * sb;
    m20 = -sh * ca;
    m21 = sh * sa * cb + ch * sb;
    m22 = -sh * sa * sb + ch * cb;

    rotationMatrix.at<double>(0, 0) = m00;
    rotationMatrix.at<double>(0, 1) = m01;
    rotationMatrix.at<double>(0, 2) = m02;
    rotationMatrix.at<double>(1, 0) = m10;
    rotationMatrix.at<double>(1, 1) = m11;
    rotationMatrix.at<double>(1, 2) = m12;
    rotationMatrix.at<double>(2, 0) = m20;
    rotationMatrix.at<double>(2, 1) = m21;
    rotationMatrix.at<double>(2, 2) = m22;

    return rotationMatrix;
}


