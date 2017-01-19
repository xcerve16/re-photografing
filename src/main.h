//
// Created by acervenka2 on 17.01.2017.
//

#ifndef REPHOTOGRAFING_MAIN_H
#define REPHOTOGRAFING_MAIN_H

#include <iostream>
#include <pthread.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iomanip>
#include "opencv2/xfeatures2d.hpp"

#include "PnPProblem.h"
#include "CameraCalibrator.h"
#include "Utils.h"
#include "MyRobustMatcher.h"
#include "Line2D.h"
#include "Model.h"
#include "MSAC.h"
#include "RobustMatcher.h"


#ifdef WIN32

#include <windows.h>

#endif
#ifdef linux
#include <stdio.h>
#endif

#define USE_PPHT
#define MAX_NUM_LINES    200

const double PI = 3.141592653589793238463;

using namespace cv;
using namespace std;
using namespace xfeatures2d;

bool end_registration = false;
int const number_registration = 8;

// Color
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);
Scalar yellow(0, 255, 255);
Scalar white(255, 255, 255);

ModelRegistration registration;
CameraCalibrator cameraCalibrator;
MyRobustMatcher robustMatcher;
RobustMatcher rmatcher;
PnPProblem pnp_detection;
KalmanFilter kalmanFilter;
MSAC msac;


// Robust Matcher parameters
double confidenceLevel = 0.98;
int numKeyPoints = 1000;
float ratioTest = 0.70f;
double max_dist = 0;
double min_dist = 100;

// SIFT parameters
int numOfPoints = 1000;

// MSAC parameters
int mode = MODE_NIETO;
int numVps = 3;
bool verbose = false;

// Window's names
string WIN_USER_SELECT_POINT = "WIN_USER_SELECT_POINT";
string WIN_REF_IMAGE_FOR_USER = "WIN_REF_IMAGE_FOR_USER";
string WIN_REF_IMAGE_WITH_HOUGH_LINES = "WIN_REF_IMAGE_WITH_HOUGH_LINES";
string WIN_REF_IMAGE_WITH_VANISH_POINTS = "WIN_REF_IMAGE_WITH_VANISH_POINTS";
string WIN_REAL_TIME_DEMO = "WIN_REAL_TIME_DEMO";

// File's path
String path_to_first_image = "resource/image/test1.jpg";
String path_to_second_image = "resource/image/test2.jpg";
String path_to_ref_image = "resource/image/refVelehrad.jpg";
string video_read_path = "resource/video/video2.mp4";
string yml_read_path = "result.yml";
string ply_read_path = "resource/data/box.ply";

// RANSAC parameters
int iterationsCount = 500;
float reprojectionError = 2.0;
double confidence = 0.95;

// Kalman Filter parameters
int minInliersKalman = 30;
int nStates = 18;            // the number of states
int nMeasurements = 6;       // the number of measured states
int nInputs = 0;             // the number of control actions
double dt = 0.125;           // time between measurements (1/FPS)

// PnP parameters
int pnpMethod = SOLVEPNP_ITERATIVE;


struct arg_struct {
    Mat frame;
    vector<Mat> frames;
    Mat detection_model;
    vector<Point3f> list_points3D_model;
    vector<Point2f> list_points2D_scene_match;
};


void *robust_matcher(void *arg);

void *fast_robust_matcher(void *arg);

static void onMouseModelRegistration(int event, int x, int y, int, void *);

vector<Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg);

void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

void updateKalmanFilter(KalmanFilter &KF, Mat &measurements, Mat &translation_estimated, Mat &rotation_estimated);

void fillMeasurements(Mat &measurements, const Mat &translation_measured, const Mat &rotation_measured);

double convert_radian_to_degree(double input) {
    return (input * 180) / PI;
}

#endif //REPHOTOGRAFING_MAIN_H
