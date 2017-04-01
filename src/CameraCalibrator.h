/**
 * CameraCalibrator.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef REPHOTOGRAFING_CALIBRATIONCAMERA_H
#define REPHOTOGRAFING_CALIBRATIONCAMERA_H

#include <opencv2/core/mat.hpp>

using namespace cv;
using namespace std;

/**
 * Kalibrace fotoaparatu
 * Dle tutorialu dostupneho na https://www.packtpub.com/books/content/learn-computer-vision-applications-open-cv
 */

class CameraCalibrator {

private:

    vector<std::vector<Point3f>> _object_points;

    vector<std::vector<Point2f>> _image_points;

    cv::Mat _camera_matrix;

public:

    CameraCalibrator() { _camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1); };

    int addChessboardPoints(const vector<string> &list_files_names, Size &border_size);

    void addPoints(const vector<cv::Point2f> &image_corners, const vector<cv::Point3f> &object_corners);

    double calibrate(Size image_size);

    cv::Mat getCameraMatrix() { return _camera_matrix; }
};


#endif
