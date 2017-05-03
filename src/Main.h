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
#include <opencv2/video/tracking.hpp>
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

bool end_registration = false;
int const number_registration = 8;
int index_of_registration = 0;

// Color
cv::Scalar red(0, 0, 255);
cv::Scalar green(0, 255, 0);
cv::Scalar blue(255, 0, 0);

CameraCalibrator cameraCalibrator;
RobustMatcher robustMatcher;
PnPProblem pnp_registration;
PnPProblem pnp_detection;
cv::KalmanFilter kalmanFilter;
MSAC msac;


// Robust cv::Matcher parameters
double confidenceLevel = 0.999;
float ratioTest = 0.70f;
double max_distance = 2;

// SIFT parameters
int numKeyPoints = 1000;

// MSAC parameters
int mode = MODE_NIETO;
int numVps = 3;
bool verbose = false;

// Window's names
const std::string WIN_USER_SELECT_POINT = "WIN_USER_SELECT_POINT";
const std::string WIN_REF_IMAGE_FOR_USER = "WIN_REF_IMAGE_FOR_USER";
const std::string WIN_REAL_TIME_DEMO = "WIN_REAL_TIME_DEMO";

// Grand hotel
/*const std::string path_to_first_image = "resource/image/grand_hotel (2).jpg";
const std::string path_to_second_image = "resource/image/grand_hotel (4).jpg";
const std::string path_to_ref_image = "resource/reference/ref_grand_hotel.jpg";
const std::string video_read_path = "resource/video/grand_hotel.mp4";
const std::string path_rephotography = "resource/results/exp_grand_hotel.jpg";*/

// Biskupsky palac
const std::string path_to_first_image = "resource/image/GPS/Biskupsky_dvur (1).jpg";
const std::string path_to_second_image = "resource/image/GPS/Biskupsky_dvur (2).jpg";
const std::string path_to_third_image = "resource/image/GPS/Biskupsky_dvur (3).jpg";

const std::string path_to_ref_image = "resource/reference/ref_biskupsky_palac.jpg";
const std::string video_read_path = "resource/video/biskupsky_palac.mp4";
//const std::string path_rephotography = "resource/results/exp_biskupsky_palac.jpg";*/

// Ulice Ceska
/*const std::string path_to_first_image = "resource/image/ulice_ceska (2).jpg";
const std::string path_to_second_image = "resource/image/ulice_ceska (4).jpg";
const std::string path_to_ref_image = "resource/reference/ref_ulice_ceska.jpg";
const std::string video_read_path = "resource/video/ulice_ceska.3gp";
const std::string path_rephotography = "resource/results/exp_ulice_ceska.jpg";*/

// Moravske zemske divadlo
/*const std::string path_to_first_image = "resource/image/rsz_moravske_zemske_divadlo_3.jpg";
const std::string path_to_second_image = "resource/image/rsz_moravske_zemske_divadlo_4.jpg";
const std::string path_to_ref_image = "resource/reference/rsz_ref_moravske_zemske_divadlo.jpg";
const std::string video_read_path = "resource/video/moravske_zemske_divadlo.mp4";
const std::string path_rephotography = "resource/results/exp_moravske_zemske_divadlo.jpg";*/

// Kasna parmas
/*const std::string path_to_first_image = "resource/image/kasna_parmas (10).jpg";
const std::string path_to_second_image = "resource/image/kasna_parmas (8).jpg";
const std::string path_to_ref_image = "resource/reference/rsz_ref_kasna_parmas.jpg";
const std::string video_read_path = "resource/video/kasna_parmas (1).3gp";
const std::string path_rephotography = "resource/results/exp_kasna_parmas.jpg";*/

// Kostel svateho Jakuba
/*const std::string path_to_first_image = "resource/image/rsz_kostel_svateho_jakuba_1.jpg";
const std::string path_to_second_image = "resource/image/rsz_kostel_svateho_jakuba_2.jpg";
const std::string path_to_ref_image = "resource/image/ref_kostel_svateho_jakuba.jpg";
const std::string video_read_path = "resource/video/kostel_svateho_jakuba.mp4";
const std::string path_rephotography = "resource/results/exp_kostel_svateho_jakuba.jpg";*/

// Galerie
/*const std::string path_to_first_image = "resource/image/galerie (8).jpg";
const std::string path_to_second_image = "resource/image/galerie (9).jpg";
const std::string path_to_ref_image = "resource/reference/ref_galerie.jpg";
const std::string video_read_path = "resource/video/galerie (2).3gp";
const std::string path_rephotography = "resource/results/exp_galerie.jpg";*/

// Janackovo_divadlo
/*const std::string path_to_first_image = "resource/image/janackovo_divadlo (7).jpg";
const std::string path_to_second_image = "resource/image/janackovo_divadlo (6).jpg";
const std::string path_to_ref_image = "resource/reference/rsz_ref_janackovo_divadlo.jpg";
const std::string video_read_path = "resource/video/janackovo_divadlo (2).3gp";
const std::string path_rephotography = "resource/results/exp_janackovo_divadlo.jpg";*/

// Bazilika Velehrad
/*const std::string path_to_first_image = "resource/image/bazilika_velehrad (2).jpg";
const std::string path_to_second_image = "resource/image/bazilika_velehrad (3).jpg";
const std::string path_to_ref_image = "resource/reference/ref_bazilika_velehrad.jpg";
const std::string video_read_path = "resource/video/bazilika_velehrad.mp4";
const std::string path_rephotography = "resource/results/exp_bazilika_velehrad.jpg";*/

// Cerveny kostel
/*const std::string path_to_first_image = "resource/image/cerveny_kostel (12).jpg";
const std::string path_to_second_image = "resource/image/cerveny_kostel (13).jpg";
const std::string path_to_ref_image = "resource/reference/ref_cerveny_kostel.jpg";
const std::string video_read_path = "resource/video/cerveny_kostel_2.3gp";
const std::string path_rephotography = "resource/results/exp_cerveny_kostel.jpg";*/

// ERROR message
const std::string ERROR_READ_IMAGE = "Could not open or find the image";
//const std::string ERROR_WRITE_IMAGE = "Could not open the camera device";
const std::string ERROR_OPEN_CAMERA = "Could not open the camera device";


// RANSAC parameters
bool useExtrinsicGuess = false;
int iterationsCount = 50;
float reprojectionError = 10.0;
double confidence = 0.95;
int pnp_method = cv::SOLVEPNP_ITERATIVE;

// Kalman Filter parameters
int minInliersKalman = 10;
int nStates = 18;
int nMeasurements = 6;
int nInputs = 0;
double dt = 0.125;


struct robust_matcher_struct {
    cv::Mat current_frame;
    std::vector<cv::Point3f> list_3D_points;
    cv::Mat measurements;
    int directory;
};

struct fast_robust_matcher_struct {
    cv::Mat last_current_frame;
    std::vector<cv::Point3f> list_3D_points;
    cv::Mat current_frame;
    int directory;
};

void *robust_matcher(void *arg);

void *fast_robust_matcher(void *arg);

bool getRobustEstimation(cv::Mat current_frame_vis, std::vector<cv::Point3f> list_3D_points, cv::Mat measurements,
                         int &directory);

bool getLightweightEstimation(cv::Mat last_current_frame_vis, std::vector<cv::Point3f> list_3D_points,
                              cv::Mat current_frame_vis, int &directory);

void updateKalmanFilter(cv::KalmanFilter &KF, cv::Mat &measurements, cv::Mat &translation_estimated,
                        cv::Mat &rotation_estimated);

void fillMeasurements(cv::Mat &measurements, const cv::Mat &translation_measured, const cv::Mat &rotation_measured);

pthread_t fast_robust_matcher_t, robust_matcher_t;

robust_matcher_struct robust_matcher_arg_struct;

fast_robust_matcher_struct fast_robust_matcher_arg_struct;

int getDirectory(int x, int y);

class Main {

private:
    cv::Mat first_image;

    cv::Mat second_image;

    cv::Mat ref_image;

    ModelRegistration registration;

    std::vector<cv::Point2f> detection_points_first_image;

    std::vector<cv::Point3f> list_3D_points;

    std::vector<int> index_points;

    std::vector<cv::Mat> current_frames;

    cv::Mat measurements;

    int start;

    int end;

    void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

    cv::Mat loadImage(const std::string path_to_file);

    std::vector<cv::Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg);


public:
    Main() {};

    void initReconstruction(cv::Mat first_frame, cv::Mat second_frame, cv::Mat reference_frame, cv::Point2f cf,
                            cv::Point2f ff, cv::Point2f cc, cv::Point2f fc);

    cv::Mat processReconstruction();

    ModelRegistration getModelRegistration() { return registration; }

    cv::Point2f nextPoint();

    cv::Point2f registrationPoints(float x, float y);

    void initNavigation();

    int processNavigation(cv::Mat current_frame, int count_frames);


};


#endif
