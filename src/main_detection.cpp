// C++
#include <iostream>
#include <time.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iomanip>
#include <pthread.h>
// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "Utils.h"
#include "RobustMatcher.h"
#include "MyRobustMatcher.h"
#include "CameraCalibrator.h"

/**  GLOBAL VARIABLES  **/

using namespace cv;
using namespace std;

const double PI = 3.141592653589793238463;

string video_read_path = "resource/video/video2.mp4";
string yml_read_path = "result.yml";
string ply_read_path = "resource/data/box.ply";

// Color
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);
Scalar yellow(0, 255, 255);
Scalar white(255, 255, 255);

CameraCalibrator cameraCalibrator;
RobustMatcher rmatcher;
PnPProblem pnp_detection;

// Robust Matcher parameters
int numKeyPoints = 1000;
float ratioTest = 0.70f;
double max_dist = 0;
double min_dist = 100;

// RANSAC parameters
int iterationsCount = 500;
float reprojectionError = 2.0;
double confidence = 0.95;

// Kalman Filter parameters
int minInliersKalman = 30;

// PnP parameters
int pnpMethod = SOLVEPNP_ITERATIVE;

// Window's names
string WIN_REAL_TIME_DEMO = "WIN_REAL_TIME_DEMO";


void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

void updateKalmanFilter(KalmanFilter &KF, Mat &measurements, Mat &translation_estimated, Mat &rotation_estimated);

void fillMeasurements(Mat &measurements, const Mat &translation_measured, const Mat &rotation_measured);

double convert_radian_to_degree(double input) {
    double degrees = (input * 180) / PI;
    return degrees;
}

struct arg_struct {
    Mat frame;
    vector<Mat> frames;
    Mat detection_model;
    vector<Point3f> list_points3D_model;
    vector<Point2f> list_points2D_scene_match;
};

void *fast_robust_matcher(void *arg) {

    struct arg_struct *arg_struct = (struct arg_struct *) arg;
    vector<Mat> frames = arg_struct->frames;
    for (int i = 0; i < frames.size(); i++) {
        Mat frame_vis = arg_struct->frames[i];
        Mat detection_model = arg_struct->detection_model;
        vector<Point3f> list_points3D_model = arg_struct->list_points3D_model;

        vector<Point3f> list_points3D_model_match;
        vector<Point2f> list_points2D_scene_match;
        vector<KeyPoint> keypoints_scene;

        vector<DMatch> good_matches;


        rmatcher.fastRobustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_points3D_model[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3D_model_match.push_back(point3d_model);
            list_points2D_scene_match.push_back(point2d_scene);
        }
        arg_struct->list_points2D_scene_match = list_points2D_scene_match;
        arg_struct->frame = frame_vis;
        arg_struct->detection_model = detection_model;
        arg_struct->list_points3D_model;
        draw2DPoints(frame_vis, list_points2D_scene_match, green);

        namedWindow("Matcher");
        imshow("Matcher", frame_vis);

    }
    pthread_exit(NULL);
}


void *robust_matcher(void *arg) {

    struct arg_struct *arg_struct1 = (struct arg_struct *) arg;

    Mat frame_vis = arg_struct1->frame;
    Mat detection_model = arg_struct1->detection_model;
    vector<Point3f> list_points3D_model = arg_struct1->list_points3D_model;

    vector<Point3f> list_points3D_model_match;
    vector<Point2f> list_points2D_scene_match;
    vector<KeyPoint> keypoints_scene;

    vector<DMatch> good_matches;


    rmatcher.robustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        Point3f point3d_model = list_points3D_model[good_matches[match_index].trainIdx];
        Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
        list_points3D_model_match.push_back(point3d_model);
        list_points2D_scene_match.push_back(point2d_scene);
    }

    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;

    if (good_matches.size() > 0) {
        pnp_detection.estimatePoseRANSAC(list_points3D_model_match, list_points2D_scene_match, pnpMethod,
                                         inliers_idx, iterationsCount, reprojectionError, confidence);
        for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
            int n = inliers_idx.at<int>(inliers_index);
            Point2f point2d = list_points2D_scene_match[n];
            list_points2d_inliers.push_back(point2d);
        }
    }

    arg_struct1->list_points2D_scene_match = list_points2D_scene_match;
    draw2DPoints(frame_vis, list_points2D_scene_match, blue);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {


    Mat image;
    vector<string> fileList;
    for (int i = 1; i <= 20; i++) {
        stringstream str;
        str << "resource/image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        fileList.push_back(str.str());
        image = imread(str.str(), 0);
    }

    Size boardSize(6, 4);
    cameraCalibrator.addChessboardPoints(fileList, boardSize);
    cameraCalibrator.calibrate((Size &) image.size);

    double cameraParams[4];
    cameraParams[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    cameraParams[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    cameraParams[2] = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
    cameraParams[3] = cameraCalibrator.getCameraMatrix().at<double>(1, 2);


    pnp_detection.setMatrixParam(cameraParams);
    PnPProblem pnp_detection_est(cameraParams);

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

    vector<Point3f> list_points3D_model = model.get_points3d();
    Mat descriptors_model = model.get_descriptors();

    //namedWindow("REAL TIME DEMO", WINDOW_KEEPRATIO);

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

    Mat current_frame = imread("resource/image/test0.jpg");
    Mat first_frame = imread("resource/image/test1.jpg");

    Mat detection_model;

    vector<DMatch> matches;


    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    Ptr<SURF> detector = SURF::create();
    FlannBasedMatcher matcher;

    vector<KeyPoint> keyPoints2;
    Ptr<FeatureDetector> orb = ORB::create(numKeyPoints);

    Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

    Ptr<DescriptorMatcher> descriptorMatcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    rmatcher.setDescriptorMatcher(descriptorMatcher);
    rmatcher.setRatio(ratioTest);

    vector<KeyPoint> keypoints_scene;


    /*************************************************************
     *           * Robust Camera Pose Estimation *
     *************************************************************/

    detector->setHessianThreshold(numKeyPoints);

    detector->detectAndCompute(current_frame, Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(first_frame, Mat(), keypoints_2, descriptors_2);


    matcher.match(descriptors_1, descriptors_2, matches);

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 0.02)) { good_matches.push_back(matches[i]); }
    }

    Mat img_matches;
    drawMatches(current_frame, keypoints_1, first_frame, keypoints_2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



    /*************************************************************
     *           * Real-time Camera Pose Estimation *
     *************************************************************/

    orb->detect(current_frame, keyPoints2);
    orb->compute(current_frame, keyPoints2, detection_model);

    rmatcher.setFeatureDetector(orb);
    rmatcher.setDescriptorExtractor(orb);

    vector<Point3f> list_points3d_model_match;
    vector<Point2f> list_points2d_scene_match;

    bool isFirstImage = true;

    vector<Point2f> featuresPrevious;
    vector<Point2f> featuresCurrent;
    vector<Point2f> featuresNextPos;
    vector<uchar> featuresFound;
    Mat cImage, lastImgRef, err;

    while (cap.read(frame) && waitKey(30) != 27) {

        frame_vis = frame.clone();

        good_matches.clear();
        rmatcher.fastRobustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

        /*list_points3d_model_match.clear();
        list_points2d_scene_match.clear();
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_points3D_model[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }

        draw2DPoints(frame_vis, list_points2d_scene_match, blue);*/

        rmatcher.robustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

        list_points3d_model_match.clear();
        list_points2d_scene_match.clear();
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_points3D_model[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }

        Mat inliers_idx;
        vector<Point2f> list_points2d_inliers;

        if (matches.size() > 0) {
            pnp_detection.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnpMethod,
                                             inliers_idx, iterationsCount, reprojectionError, confidence);
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
                int n = inliers_idx.at<int>(inliers_index);
                Point2f point2d = list_points2d_scene_match[n];
                list_points2d_inliers.push_back(point2d);
            }
        }

        //draw2DPoints(frame_vis, list_points2d_inliers, blue);


        /*************************************************************
         *                  * Interleaved Scheme *
        *************************************************************/

        double cx = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
        double cy = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

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

        /*************************************************************
         *                   * Lucas-Kanade method *
         *************************************************************/

        cvtColor(frame_vis, cImage, CV_BGR2GRAY);
        cImage.convertTo(frame_vis, CV_8U);
        featuresPrevious = std::move(featuresCurrent);
        goodFeaturesToTrack(frame_vis, featuresCurrent, 30, 0.01, 30);

        if (!isFirstImage) {
            calcOpticalFlowPyrLK(lastImgRef, frame_vis, featuresPrevious, featuresNextPos, featuresFound, err);
            for (size_t i = 0; i < featuresNextPos.size(); i++) {
                if (featuresFound[i]) {
                    line(frame_vis, featuresPrevious[i], featuresNextPos[i], Scalar(0, 0, 255), 5);
                }
            }
        }

        lastImgRef = frame_vis.clone();
        isFirstImage = false;

        /*************************************************************
         *                   * Kalman filter *
         *************************************************************/

        Mat translation_estimated(3, 1, CV_64F);
        Mat rotation_estimated(3, 3, CV_64F);

        // update the Kalman filter with good measurements
        updateKalmanFilter(KF, measurements, translation_estimated, rotation_estimated);
        pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

        float l = 5;
        vector<Point2f> pose_points2d;
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(5, 5, 5)));  // axis center
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(l, 0, 0)));  // axis x
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, l, 0)));  // axis y
        pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0, 0, l)));  // axis z
        draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes

        time(&end);
        ++counter;
        sec = difftime(end, start);
        fps = counter / sec;

        drawFPS(frame_vis, fps, yellow); // frame ratio
        double detection_ratio = ((double) inliers_idx.rows / (double) good_matches.size()) * 100;
        drawConfidence(frame_vis, detection_ratio, yellow);

        int inliers_int = inliers_idx.rows;


        /*

        Mat r, t;
        vector<Point2f> ll;
        vector<Point2f> kk;
         Mat essential = findEssentialMat(list_points2d_inliers, list_points2d_scene_match, cameraCalibrator.getCameraMatrix());
         correctMatches(essential, list_points2d_inliers, list_points2d_scene_match, list_points2d_inliers, list_points2d_scene_match);
         recoverPose(essential, list_points2d_inliers, list_points2d_scene_match, r, t, 1, Point2f(cx, cy));

         double xAngle = atan2f(r.at<float>(2, 1), r.at<float>(2, 2));
         double yAngle = atan2f(-r.at<float>(2, 0), sqrtf(r.at<float>(2, 1) * r.at<float>(2, 1) + r.at<float>(2, 2) * r.at<float>(2, 2)));
         double zAngle = atan2f(r.at<float>(1, 0), r.at<float>(0, 0));

         xAngle = (int) convert_radian_to_degree(xAngle);
         yAngle = (int) convert_radian_to_degree(yAngle);
         zAngle = (int) convert_radian_to_degree(zAngle);

         cout << "xAngle: " << xAngle << "%" << endl;
         cout << "yAngle: " << yAngle << "%" << endl;
         cout << "zAngle: " << zAngle << "%" << endl;*/

        imshow(WIN_REAL_TIME_DEMO, frame_vis);
    }
    destroyWindow(WIN_REAL_TIME_DEMO);
}

/**********************************************************************************************************/
void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt) {

    KF.init(nStates, nMeasurements, nInputs, CV_64F);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(KF.errorCovPost, Scalar::all(1));

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
    KF.transitionMatrix.at<double>(0, 6) = 0.5 * dt * dt;
    KF.transitionMatrix.at<double>(1, 7) = 0.5 * dt * dt;
    KF.transitionMatrix.at<double>(2, 8) = 0.5 * dt * dt;

    // orientation
    KF.transitionMatrix.at<double>(9, 12) = dt;
    KF.transitionMatrix.at<double>(10, 13) = dt;
    KF.transitionMatrix.at<double>(11, 14) = dt;
    KF.transitionMatrix.at<double>(12, 15) = dt;
    KF.transitionMatrix.at<double>(13, 16) = dt;
    KF.transitionMatrix.at<double>(14, 17) = dt;
    KF.transitionMatrix.at<double>(9, 15) = 0.5 * dt * dt;
    KF.transitionMatrix.at<double>(10, 16) = 0.5 * dt * dt;
    KF.transitionMatrix.at<double>(11, 17) = 0.5 * dt * dt;


    /** Mereny model **/

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

void updateKalmanFilter(KalmanFilter &KF, Mat &measurement, Mat &translation_estimated, Mat &rotation_estimated) {

    Mat prediction = KF.predict();

    Mat estimated = KF.correct(measurement);

    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);


    Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    rotation_estimated = euler2rot(eulers_estimated);

}

void fillMeasurements(Mat &measurements, const Mat &translation_measured, const Mat &rotation_measured) {

    Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = rot2euler(rotation_measured);

    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}
