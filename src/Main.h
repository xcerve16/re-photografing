/**
 * Main.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

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
#include "Line.h"
#include "MSAC.h"
#include "RobustMatcher.h"


#ifdef WIN32

#include <windows.h>
#include <opencv/cv.hpp>

#endif
#ifdef linux
#include <stdio.h>
#endif

#define USE_PPHT
#define MAX_NUM_LINES    200

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
RobustMatcher robustMatcher;
PnPProblem pnp_registration;
PnPProblem pnp_detection;
KalmanFilter kalmanFilter;
MSAC msac;


// Robust Matcher parameters
double confidenceLevel = 0.99;
float ratioTest = 0.70f;
double min_dist = 100;

// SIFT parameters
int numKeyPoints = 1000;

// MSAC parameters
int mode = MODE_NIETO;
int numVps = 3;
bool verbose = false;

// Window's names
const string WIN_USER_SELECT_POINT = "WIN_USER_SELECT_POINT";
const string WIN_REF_IMAGE_FOR_USER = "WIN_REF_IMAGE_FOR_USER";
const string WIN_REAL_TIME_DEMO = "WIN_REAL_TIME_DEMO";

// Grand hotel
/*const string path_to_first_image = "resource/image/grand_hotel (2).jpg";
const string path_to_second_image = "resource/image/grand_hotel (4).jpg";
const string path_to_ref_image = "resource/reference/ref_grand_hotel.jpg";
const string video_read_path = "resource/video/grand_hotel.mp4";
const string path_rephotography = "resource/results/exp_grand_hotel.jpg";*/

// Biskupsky palac
/*const string path_to_first_image = "resource/image/rsz_biskupsky_palac_2.jpg";
const string path_to_second_image = "resource/image/rsz_biskupsky_palac_3.jpg";
const string path_to_ref_image = "resource/image/GPS/Biskupsky_dvur (1).jpg";
const string video_read_path = "resource/video/biskupsky_palac.mp4";
const string path_rephotography = "resource/results/exp_biskupsky_palac.jpg";*/

// Ulice Ceska
/*const string path_to_first_image = "resource/image/ulice_ceska (2).jpg";
const string path_to_second_image = "resource/image/ulice_ceska (4).jpg";
const string path_to_ref_image = "resource/reference/ref_ulice_ceska.jpg";
const string video_read_path = "resource/video/ulice_ceska.3gp";
const string path_rephotography = "resource/results/exp_ulice_ceska.jpg";*/

// Moravske zemske divadlo
/*const string path_to_first_image = "resource/image/rsz_moravske_zemske_divadlo_3.jpg";
const string path_to_second_image = "resource/image/rsz_moravske_zemske_divadlo_4.jpg";
const string path_to_ref_image = "resource/reference/rsz_ref_moravske_zemske_divadlo.jpg";
const string video_read_path = "resource/video/moravske_zemske_divadlo.mp4";
const string path_rephotography = "resource/results/exp_moravske_zemske_divadlo.jpg";*/

// Kasna parmas
/*const string path_to_first_image = "resource/image/kasna_parmas (10).jpg";
const string path_to_second_image = "resource/image/kasna_parmas (8).jpg";
const string path_to_ref_image = "resource/reference/rsz_ref_kasna_parmas.jpg";
const string video_read_path = "resource/video/kasna_parmas (1).3gp";
const string path_rephotography = "resource/results/exp_kasna_parmas.jpg";*/

// Kostel svateho Jakuba
/*const string path_to_first_image = "resource/image/rsz_kostel_svateho_jakuba_1.jpg";
const string path_to_second_image = "resource/image/rsz_kostel_svateho_jakuba_2.jpg";
const string path_to_ref_image = "resource/image/ref_kostel_svateho_jakuba.jpg";
const string video_read_path = "resource/video/kostel_svateho_jakuba.mp4";
const string path_rephotography = "resource/results/exp_kostel_svateho_jakuba.jpg";*/

// Galerie
/*const string path_to_first_image = "resource/image/GPS/Galerie (1).jpg";
const string path_to_second_image = "resource/image/GPS/Galerie (2).jpg";
const string path_to_ref_image = "resource/image/GPS/Galerie (7).jpg";
const string video_read_path = "resource/video/galerie (2).3gp";
const string path_rephotography = "resource/results/exp_galerie.jpg";*/

// Janackovo_divadlo
/*const string path_to_first_image = "resource/image/janackovo_divadlo (7).jpg";
const string path_to_second_image = "resource/image/janackovo_divadlo (6).jpg";
const string path_to_ref_image = "resource/image/GPS/Janackovo_divadlo (4).jpg";
const string video_read_path = "resource/video/janackovo_divadlo (2).3gp";
const string path_rephotography = "resource/results/exp_janackovo_divadlo.jpg";*/

// Bazilika Velehrad
/*const string path_to_first_image = "resource/image/bazilika_velehrad (2).jpg";
const string path_to_second_image = "resource/image/bazilika_velehrad (3).jpg";
const string path_to_ref_image = "resource/reference/ref_bazilika_velehrad.jpg";
const string video_read_path = "resource/video/bazilika_velehrad.mp4";
const string path_rephotography = "resource/results/exp_bazilika_velehrad.jpg";*/

// Cerveny kostel
const string path_to_first_image = "resource/image/cerveny_kostel (10).jpg";
const string path_to_second_image = "resource/image/cerveny_kostel (11).jpg";
const string path_to_ref_image = "resource/image/GPS/Cerveny_kostel (3).jpg";
const string video_read_path = "resource/video/cerveny_kostel_2.3gp";
const string path_rephotography = "resource/results/exp_cerveny_kostel.jpg";

// ERROR message
const string ERROR_READ_IMAGE = "Could not open or find the image";
const string ERROR_WRITE_IMAGE = "Could not open the camera device";
const string ERROR_OPEN_CAMERA = "Could not open the camera device";


// RANSAC parameters
bool useExtrinsicGuess = false;
int iterationsCount = 1000;
float reprojectionError = 10.0;
double confidence = 0.95;
int pnp_method = SOLVEPNP_ITERATIVE;

// Kalman Filter parameters
int minInliersKalman = 10;
int nStates = 18;
int nMeasurements = 6;
int nInputs = 0;
double dt = 0.125;


struct matcher_struct {
    int num;
    Mat last_current_frame;
    Mat current_frame;
    vector<Point3f> list_3D_points;
    Mat measurements;
};

void *robust_matcher(void *arg);

void *fast_robust_matcher(void *arg);

static void onMouseModelRegistration(int event, int x, int y, int, void *);

vector<Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg);

bool getRobustEstimation(Mat current_frame_vis, vector<Point3f> list_3D_points, Mat measurements);

bool getLightweightEstimation(Mat last_current_frame_vis, Mat current_frame_vis,
                              vector<Point3f> list_3D_points, Mat measurements);

void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

void updateKalmanFilter(KalmanFilter &KF, Mat &measurements, Mat &translation_estimated, Mat &rotation_estimated);

void fillMeasurements(Mat &measurements, const Mat &translation_measured, const Mat &rotation_measured);

Mat loadImage(const string basic_string);

#endif
