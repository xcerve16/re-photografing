//
// Created by acervenka2 on 18.12.2016.
//

#include <opencv/cv.hpp>

#include "CameraCalibrator.h"

int CameraCalibrator::addChessboardPoints(const vector<string> &filelist, Size &boardSize) {

    vector<Point2f> imageCorners;
    vector<Point3f> objectCorners;

    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectCorners.push_back(Point3f(i, j, 0.0f));
        }
    }

    Mat image;
    int successes = 0;

    for (int i = 0; i < filelist.size(); i++) {
        image = imread(filelist[i], 0);
        bool found = findChessboardCorners(image, boardSize, imageCorners);
        cornerSubPix(image, imageCorners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
        if (imageCorners.size() == boardSize.area()) {
            addPoints(imageCorners, objectCorners);
            successes++;
        }
    }
    return successes;
}

void CameraCalibrator::addPoints(const vector<cv::Point2f> &imageCorners,
                                 const vector<cv::Point3f> &objectCorners) {
    imagePoints.push_back(imageCorners);
    objectPoints.push_back(objectCorners);
}

double CameraCalibrator::calibrate(Size &imageSize) {
    mustInitUndistort = true;
    return calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flag);
}

Mat CameraCalibrator::myremap(const Mat &image) {
    Mat undistorted;
    if (mustInitUndistort) {
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), image.size(), CV_64FC2, map1, map2);
        mustInitUndistort = false;
    }

    remap(image, undistorted, map1, map2, INTER_LINEAR);
    return undistorted;
}

void CameraCalibrator::cleanVectors() {
    rvecs.clear();
    tvecs.clear();
}
