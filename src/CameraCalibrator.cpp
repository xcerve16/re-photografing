//
// Created by acervenka2 on 18.12.2016.
//

#include <opencv/cv.hpp>
#include <iostream>

#include "CameraCalibrator.h"
#include "Utils.h"

int CameraCalibrator::addChessboardPoints(const vector<string> &filelist, Size &boardSize) {

    vector<Point2f> imageCorners;
    vector<Point2f> imageCorners1;
    vector<Point3f> objectCorners;

    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectCorners.push_back(Point3f(i, j, 0.0f));
        }
    }

    Mat image, image_gray;
    int successes = 0;
    for (int i = 0; i < filelist.size(); i++) {
        image = imread(filelist[i]);
        cvtColor( image, image_gray, cv::COLOR_RGB2GRAY);
        if (findChessboardCorners(image_gray, boardSize, imageCorners )) {
            cornerSubPix(image_gray, imageCorners, Size(5, 5), Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
            if (imageCorners.size() == boardSize.area()) {
                addPoints(imageCorners, objectCorners);
                successes++;
                /*drawChessboardCorners(image, boardSize, imageCorners, true);
                draw2DPoints(image, imageCorners1, Scalar(255, 255, 255));
                imshow("CALIBRACE", image);
                waitKey(5000);*/
            }
        }
    }
    return successes;
}

void CameraCalibrator::addPoints(const vector<cv::Point2f> &imageCorners,
                                 const vector<cv::Point3f> &objectCorners) {
    imagePoints.push_back(imageCorners);
    objectPoints.push_back(objectCorners);
}

double CameraCalibrator::calibrate(Size imageSize) {
    mustInitUndistort = true;
    std::vector<cv::Mat> rvecs, tvecs;
    return calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flag);
}

Mat CameraCalibrator::myremap(const Mat &image) {
    Mat undistorted;
    if (mustInitUndistort) {
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), image.size(), CV_32FC1, map1, map2);
        mustInitUndistort = false;
    }

    remap(image, undistorted, map1, map2, INTER_LINEAR);
    return undistorted;
}
