// C++
#include <iostream>
#include <time.h>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "RobustMatcher.h"
#include "ModelRegistration.h"
#include "Utils.h"

/**  GLOBAL VARIABLES  **/

using namespace cv;
using namespace std;

string video_read_path = "data/box.mp4";       // recorded video
string yml_read_path = "data/cookies_ORB.yml"; // 3dpts + descriptors
string ply_read_path = "data/box.ply";         // mesh

// Intrinsic camera parameters: UVC WEBCAM
double f = 55;                           // focal length in mm
double sx = 22.3, sy = 14.9;             // sensor size
double width = 640, height = 480;        // image size

double params_WEBCAM[] = {width * f / sx,   // fx
                          height * f / sy,  // fy
                          width / 2,      // cx
                          height / 2};    // cy

// Some basic colors
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);
Scalar yellow(0, 255, 255);


// Robust Matcher parameters
int numKeyPoints = 2000;      // number of detected keypoints
float ratioTest = 0.70f;          // ratio test
bool fast_match = true;       // fastRobustMatch() or robustMatch()

// RANSAC parameters
int iterationsCount = 500;      // number of Ransac iterations.
float reprojectionError = 2.0;  // maximum allowed distance to consider it an inlier.
double confidence = 0.95;        // ransac successful confidence.

// Kalman Filter parameters
int minInliersKalman = 30;    // Kalman threshold updating

// PnP parameters
int pnpMethod = SOLVEPNP_ITERATIVE;


/**  Functions headers  **/
void help();

void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

void updateKalmanFilter(KalmanFilter &KF, Mat &measurements,
                        Mat &translation_estimated, Mat &rotation_estimated);

void fillMeasurements(Mat &measurements,
                      const Mat &translation_measured, const Mat &rotation_measured);


/**  Main program  **/
int main(int argc, char *argv[]) {

    CommandLineParser parser(argc, (const char *const *) argv, keys);

    PnPProblem pnp_detection(params_WEBCAM);
    PnPProblem pnp_detection_est(params_WEBCAM);

    Model model;
    model.load(yml_read_path);

    Mesh mesh;
    mesh.load(ply_read_path);


    KalmanFilter KF;
    int nStates = 18;            // the number of states
    int nMeasurements = 6;       // the number of measured states
    int nInputs = 0;             // the number of control actions
    double dt = 0.125;           // time between measurements (1/FPS)

    initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);
    Mat measurements(nMeasurements, 1, CV_64F);
    measurements.setTo(Scalar(0));
    bool good_measurement = false;

    vector<Point3f> list_points3d_model = model.get_points3d();
    Mat descriptors_model = model.get_descriptors();

    namedWindow("REAL TIME DEMO", WINDOW_KEEPRATIO);


    VideoCapture cap;
    cap.open(video_read_path);

    if (!cap.isOpened()) {
        cout << "Could not open the camera device" << endl;
        return -1;
    }

    time_t start, end;
    double fps, sec;
    int counter = 0;

    time(&start);

    Mat frame, frame_vis;

    while (cap.read(frame) && waitKey(30) != 27) {

        frame_vis = frame.clone();


        /*************************************************************
         *                  * Robust matcher *
         *************************************************************/

        RobustMatcher rmatcher;

        Ptr<FeatureDetector> orb = ORB::create();

        rmatcher.setFeatureDetector(orb);
        rmatcher.setDescriptorExtractor(orb);

        Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
        Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

        Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
        rmatcher.setDescriptorMatcher(matcher);
        rmatcher.setRatio(ratioTest);

        vector<DMatch> good_matches;
        vector<KeyPoint> keypoints_scene;
        rmatcher.robustMatch(frame, good_matches, keypoints_scene, descriptors_model);

        /*************************************************************
         *                  * 2D/3D correspondences *
         *************************************************************/

        vector<Point3f> list_points3d_model_match;
        vector<Point2f> list_points2d_scene_match;
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_points3d_model[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }

        draw2DPoints(frame_vis, list_points2d_scene_match, red);
        Mat inliers_idx;
        vector<Point2f> list_points2d_inliers;

        /*************************************************************
         *                  * Estimate pose *
         *************************************************************/

        if (good_matches.size() > 0) {


            pnp_detection.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnpMethod,
                                             inliers_idx, iterationsCount, reprojectionError, confidence);


            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
                int n = inliers_idx.at<int>(inliers_index);
                Point2f point2d = list_points2d_scene_match[n];
                list_points2d_inliers.push_back(point2d);
            }

            draw2DPoints(frame_vis, list_points2d_inliers, blue);


        /*************************************************************
         *                  * Kalman filter *
        *************************************************************/

            good_measurement = false;

            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman) {

                // Get the measured translation
                Mat translation_measured(3, 1, CV_64F);
                translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                Mat rotation_measured(3, 3, CV_64F);
                rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);

                good_measurement = true;

            }

            // Instantiate estimated translation and rotation
            Mat translation_estimated(3, 1, CV_64F);
            Mat rotation_estimated(3, 3, CV_64F);

            // update the Kalman filter with good measurements
            updateKalmanFilter(KF, measurements,
                               translation_estimated, rotation_estimated);


            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

        }


        if (good_measurement) {
            drawObjectMesh(frame_vis, &mesh, &pnp_detection, green);  // draw current pose
        } else {
            drawObjectMesh(frame_vis, &mesh, &pnp_detection_est, yellow); // draw estimated pose
        }

        float l = 5;
        vector<Point2f> pose_points2d;
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, 0, 0)));  // axis center
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(l, 0, 0)));  // axis x
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, l, 0)));  // axis y
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, 0, l)));  // axis z
        draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes

        // FRAME RATE

        // see how much time has elapsed
        time(&end);

        // calculate current FPS
        ++counter;
        sec = difftime(end, start);

        fps = counter / sec;

        drawFPS(frame_vis, fps, yellow); // frame ratio
        double detection_ratio = ((double) inliers_idx.rows / (double) good_matches.size()) * 100;
        drawConfidence(frame_vis, detection_ratio, yellow);


        // -- Step X: Draw some debugging text

        // Draw some debug text
        int inliers_int = inliers_idx.rows;
        int outliers_int = (int) good_matches.size() - inliers_int;
        string inliers_str = IntToString(inliers_int);
        string outliers_str = IntToString(outliers_int);
        string n = IntToString((int) good_matches.size());
        string text = "Found " + inliers_str + " of " + n + " matches";
        string text2 = "Inliers: " + inliers_str + " - Outliers: " + outliers_str;

        drawText(frame_vis, text, green);
        drawText2(frame_vis, text2, red);

        imshow("REAL TIME DEMO", frame_vis);
    }

    destroyWindow("REAL TIME DEMO");
}

/**********************************************************************************************************/
void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt) {

    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter

    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));       // set process noise
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));   // set measurement noise
    setIdentity(KF.errorCovPost, Scalar::all(1));             // error covariance


    /** DYNAMIC MODEL **/

    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

    // position
    KF.transitionMatrix.at<double>(0, 3) = dt;
    KF.transitionMatrix.at<double>(1, 4) = dt;
    KF.transitionMatrix.at<double>(2, 5) = dt;
    KF.transitionMatrix.at<double>(3, 6) = dt;
    KF.transitionMatrix.at<double>(4, 7) = dt;
    KF.transitionMatrix.at<double>(5, 8) = dt;
    KF.transitionMatrix.at<double>(0, 6) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(1, 7) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(2, 8) = 0.5 * pow(dt, 2);

    // orientation
    KF.transitionMatrix.at<double>(9, 12) = dt;
    KF.transitionMatrix.at<double>(10, 13) = dt;
    KF.transitionMatrix.at<double>(11, 14) = dt;
    KF.transitionMatrix.at<double>(12, 15) = dt;
    KF.transitionMatrix.at<double>(13, 16) = dt;
    KF.transitionMatrix.at<double>(14, 17) = dt;
    KF.transitionMatrix.at<double>(9, 15) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(10, 16) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(11, 17) = 0.5 * pow(dt, 2);


    /** MEASUREMENT MODEL **/

    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    KF.measurementMatrix.at<double>(0, 0) = 1;  // x
    KF.measurementMatrix.at<double>(1, 1) = 1;  // y
    KF.measurementMatrix.at<double>(2, 2) = 1;  // z
    KF.measurementMatrix.at<double>(3, 9) = 1;  // roll
    KF.measurementMatrix.at<double>(4, 10) = 1; // pitch
    KF.measurementMatrix.at<double>(5, 11) = 1; // yaw

}

/**********************************************************************************************************/
void updateKalmanFilter(KalmanFilter &KF, Mat &measurement,
                        Mat &translation_estimated, Mat &rotation_estimated) {

    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();

    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);

}

/**********************************************************************************************************/
void fillMeasurements(Mat &measurements,
                      const Mat &translation_measured, const Mat &rotation_measured) {
    // Convert rotation matrix to euler angles
    Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = rot2euler(rotation_measured);

    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}
