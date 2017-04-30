/**
 * ModelRegistration.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include "ModelRegistration.h"

ModelRegistration::ModelRegistration() {
    _count_registrations = 0;
    _max_registrations = 0;
    _index_registration = 0;
}

void ModelRegistration::register2DPoint(const cv::Point2f &point2d) {
    _list_2D_points.push_back(point2d);
    _count_registrations++;
}

void ModelRegistration::register3DPoint(const cv::Point3f &point3d) {
    _list_3D_points.push_back(point3d);
}

void ModelRegistration::setIndexRegistration(int i) {
    _index_registration = i;
}

void ModelRegistration::incRegistrationIndex() {
    _index_registration++;
}



