/*
 * ModelRegistration.cpp
 *
 *  Created on: Apr 18, 2014
 *      Author: edgar
 */

#include "ModelRegistration.h"

ModelRegistration::ModelRegistration() {
    n_registrations_ = 0;
    max_registrations_ = 0;
}

void ModelRegistration::registerPoint(const cv::Point2f &point2d, const cv::Point3f &point3d) {
    // add correspondence at the end of the vector
    list_points2d_.push_back(point2d);
    list_points3d_.push_back(point3d);
    n_registrations_++;
}

void ModelRegistration::register2DPoint(const cv::Point2f &point2d) {
    // add correspondence at the end of the vector
    list_points2d_.push_back(point2d);
    n_registrations_++;
}

void ModelRegistration::register3DPoint(const cv::Point3f &point3d) {
    // add correspondence at the end of the vector
    list_points3d_.push_back(point3d);
}

void ModelRegistration::reset() {
    n_registrations_ = 0;
    max_registrations_ = 0;
    list_points2d_.clear();
    list_points3d_.clear();
}

void ModelRegistration::set_points3d(int i, cv::Point3f point3_) {
    list_points3d_[i] = point3_;
}
