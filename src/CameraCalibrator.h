//
// Created by acervenka2 on 18.12.2016.
//

#ifndef REPHOTOGRAFING_CALIBRATIONCAMERA_H
#define REPHOTOGRAFING_CALIBRATIONCAMERA_H

#include <opencv2/core/mat.hpp>

using namespace cv;
using namespace std;

class CameraCalibrator {

    vector<std::vector<Point3f>> objectPoints;
    vector<std::vector<Point2f>> imagePoints;
    Mat cameraMatrix;
    int flag;
    Mat map1, map2;
    bool mustInitUndistort;

public:
    CameraCalibrator() : flag(0), mustInitUndistort(true) {};

    int addChessboardPoints(const vector<string> &filelist, Size &boardSize);

    void addPoints(const vector<cv::Point2f> &imageCorners,
                   const vector<cv::Point3f> &objectCorners);

    double calibrate(Size imageSize);

    Mat getCameraMatrix() { return cameraMatrix; }
};


#endif //REPHOTOGRAFING_CALIBRATIONCAMERA_H
