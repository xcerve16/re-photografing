/**
 * CameraCalibrator.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include <opencv/cv.hpp>
#include <iostream>

#include "CameraCalibrator.h"

int CameraCalibrator::addChessboardPoints(const vector<string> &list_files_names, Size &border_size) {

    vector<Point2f> image_corners;
    vector<Point3f> object_corners;

    for (int i = 0; i < border_size.height; i++) {
        for (int j = 0; j < border_size.width; j++) {
            object_corners.push_back(Point3f(i, j, 0.0f));
        }
    }

    Mat image_gray;
    int successes = 0;
    for (int i = 0; i < list_files_names.size(); i++) {
        image_gray = imread(list_files_names[i], IMREAD_GRAYSCALE);
        if (findChessboardCorners(image_gray, border_size, image_corners)) {
            cornerSubPix(image_gray, image_corners, Size(5, 5), Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
            if (image_corners.size() == border_size.area()) {
                addPoints(image_corners, object_corners);
                successes++;
            }
        }
    }
    return successes;
}

void CameraCalibrator::addPoints(const vector<cv::Point2f> &image_corners,
                                 const vector<cv::Point3f> &object_corners) {
    _image_points.push_back(image_corners);
    _object_points.push_back(object_corners);
}

double CameraCalibrator::calibrate(Size image_size) {
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat dist_coeffs;
    return calibrateCamera(_object_points, _image_points, image_size, _camera_matrix, dist_coeffs, rvecs, tvecs);
}


