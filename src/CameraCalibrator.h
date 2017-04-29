/**
 * CameraCalibrator.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef REPHOTOGRAFING_CALIBRATIONCAMERA_H
#define REPHOTOGRAFING_CALIBRATIONCAMERA_H

#include <opencv2/core/mat.hpp>

/**
 * Kalibrace fotoaparatu
 * Dle tutorialu dostupneho na https://www.packtpub.com/books/content/learn-computer-vision-applications-open-cv
 */

class CameraCalibrator {

private:

    std::vector<std::vector<cv::Point3f> > _object_points;

    std::vector<std::vector<cv::Point2f> > _image_points;

    cv::Mat _camera_matrix;

public:

    CameraCalibrator() { _camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1); };

    ~CameraCalibrator() {};

    int addChessboardPoints(const std::vector<std::string> &list_files_names, cv::Size &border_size);

    void addPoints(const std::vector<cv::Point2f> &image_corners, const std::vector<cv::Point3f> &object_corners);

    double calibrate(cv::Size image_size);

    cv::Mat getCameraMatrix() { return _camera_matrix; }
};


#endif
