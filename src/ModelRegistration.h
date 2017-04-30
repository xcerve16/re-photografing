/**
 * ModelRegistration.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef MODELREGISTRATION_H_
#define MODELREGISTRATION_H_

#include <iostream>
#include <opencv2/core/core.hpp>

class ModelRegistration {

private:

    int _count_registrations;

    int _max_registrations;

    int _index_registration;

    std::vector<cv::Point2f> _list_2D_points;

    std::vector<cv::Point3f> _list_3D_points;

public:

    ModelRegistration();

    ~ModelRegistration() {}

    void setRegistrationMax(int n) { _max_registrations = n; }

    int getRegistrationMax() const { return _max_registrations; }

    int getRegistrationCount() const { return _count_registrations; }

    int getIndexRegistration() const { return _index_registration; }

    bool isRegistration() const { return (_count_registrations < _max_registrations); }

    void register2DPoint(const cv::Point2f &point2d);

    void register3DPoint(const cv::Point3f &point3d);

    std::vector<cv::Point2f> getList2DPoints() const { return _list_2D_points; }

    std::vector<cv::Point3f> getList3DPoints() const { return _list_3D_points; }

    void setIndexRegistration(int i);

    void incRegistrationIndex();
};

#endif