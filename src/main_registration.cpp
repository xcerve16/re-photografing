//
// Created by acervenka2 on 15.12.2016.
//

#include <iostream>

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

Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);
Scalar white(255, 255, 255);

ModelRegistration registration;
CameraCalibrator cameraCalibrator;
MyRobustMatcher robustMatcher;
Model model;
MSAC msac;

// Robust matcher parameters
double confidenceLevel = 0.98;
double minDistance = 1.0;
double ratioTest = 0.65f;

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


String path_to_first_image = "resource/image/test1.jpg";
String path_to_second_image = "resource/image/test2.jpg";
String path_to_ref_image = "resource/image/refVelehrad.jpg";

static void onMouseModelRegistration(int event, int x, int y, int, void *) {

    if (event == EVENT_LBUTTONUP) {

        Point2f point_2d = Point2f((float) x, (float) y);
        bool is_registrable = registration.is_registrable();

        if (is_registrable) {
            registration.register2DPoint(point_2d);

            if (registration.getNumRegistration() == registration.getNumMax()) {
                end_registration = true;
            }
        }
    }
}

double convert_radian_to_degree(double input) {
    double degrees = (input * 180) / PI;
    return degrees;
}

vector<Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg) {
    cv::Mat imgCanny;
    cv::Canny(imgGRAY, imgCanny, 180, 120, 3);
    vector<vector<cv::Point> > lineSegments;
    vector<cv::Point> aux;
#ifndef USE_PPHT
    vector<Vec2f> lines;
    cv::HoughLines( imgCanny, lines, 1, CV_PI/180, 200);

    for(size_t i=0; i< lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];

        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;

        Point pt1, pt2;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        aux.clear();
        aux.push_back(pt1);
        aux.push_back(pt2);
        lineSegments.push_back(aux);

        line(outputImg, pt1, pt2, CV_RGB(0, 0, 0), 1, 8);

    }
#else
    vector<Vec4i> lines;
    int houghThreshold = 70;
    if (imgGRAY.cols * imgGRAY.rows < 400 * 400)
        houghThreshold = 100;

    cv::HoughLinesP(imgCanny, lines, 1, CV_PI / 180, houghThreshold, 10, 10);

    while (lines.size() > MAX_NUM_LINES) {
        lines.clear();
        houghThreshold += 10;
        cv::HoughLinesP(imgCanny, lines, 1, CV_PI / 180, houghThreshold, 10, 10);
    }
    for (size_t i = 0; i < lines.size(); i++) {
        Point pt1, pt2;
        pt1.x = lines[i][0];
        pt1.y = lines[i][1];
        pt2.x = lines[i][2];
        pt2.y = lines[i][3];
        line(outputImg, pt1, pt2, CV_RGB(0, 0, 0), 2);

        aux.clear();
        aux.push_back(pt1);
        aux.push_back(pt2);
        lineSegments.push_back(aux);
    }

#endif

    // Multiple vanishing points
    std::vector<cv::Mat> vps;            // vector of vps: vps[vpNum], with vpNum=0...numDetectedVps
    std::vector<std::vector<int> > CS;    // index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

    // Call msac function for multiple vanishing point estimation
    msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps);
    for (int v = 0; v < vps.size(); v++) {
        printf("VP %d (%.3f, %.3f, %.3f)", v, vps[v].at<float>(0, 0), vps[v].at<float>(1, 0), vps[v].at<float>(2, 0));
        fflush(stdout);
        double vpNorm = cv::norm(vps[v]);
        if (fabs(vpNorm - 1) < 0.001) {
            printf("(INFINITE)");
            fflush(stdout);
        }
        printf("\n");
    }

    // Draw line segments according to their cluster
    msac.drawCS(outputImg, lineSegmentsClusters, vps);
    return vps;
}


int main(int argc, char *argv[]) {

    /*************************************************************
     *                   * Kalibrace *
     *************************************************************/

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

    /*************************************************************
     *                * Robust matcher *
     *************************************************************/

    Mat first_image = imread(path_to_first_image);
    Mat second_image = imread(path_to_second_image);

    robustMatcher.setConfidenceLevel(confidenceLevel);
    robustMatcher.setMinDistanceToEpipolar(minDistance);
    robustMatcher.setRatio(ratioTest);

    Ptr<FeatureDetector> featureDetector = SURF::create(numOfPoints);
    robustMatcher.setFeatureDetector(featureDetector);

    vector<DMatch> matches;
    vector<KeyPoint> key_points_first_image, key_points_second_image;
    Mat fundamental = robustMatcher.match(first_image, second_image, matches, key_points_first_image,
                                          key_points_second_image);

    vector<Point2f> detection_points_first_image, detection_points_second_image;

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = key_points_first_image[it->queryIdx].pt.x;
        float y = key_points_first_image[it->queryIdx].pt.y;
        detection_points_first_image.push_back(Point2f(x, y));
        //circle(first_image, Point((int) x, (int) y), 3, white, 3);

        x = key_points_second_image[it->trainIdx].pt.x;
        y = key_points_second_image[it->trainIdx].pt.y;
        //circle(second_image, Point((int) x, (int) y), 3, white, 3);
        detection_points_second_image.push_back(Point2f(x, y));
    }

    vector<Vec3f> lines1, lines2;
    computeCorrespondEpilines(Mat(detection_points_first_image), 1, fundamental, lines1);

    /*for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it) {
        line(second_image, Point(0, -(*it)[2] / (*it)[1]),
             Point(second_image.cols, (int) (-((*it)[2] + (*it)[0] * second_image.cols) / (*it)[1])), white);
    }*/

    computeCorrespondEpilines(Mat(detection_points_second_image), 2, fundamental, lines2);

    /*for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it) {
        line(first_image, Point(0, -(*it)[2] / (*it)[1]),
             Point(first_image.cols, (int) (-((*it)[2] + (*it)[0] * first_image.cols) / (*it)[1])), white);
    }*/

    /*namedWindow("Right Image (RANSAC)");
    imshow("Right Image (RANSAC)", image1);
    namedWindow("Left Image (RANSAC)");
    imshow("Left Image (RANSAC)", image2);*/

    /*************************************************************
     *                   * Triangulation *
     *************************************************************/

    cameraCalibrator.calibrate((Size &) first_image.size);
    Mat rotation_vector_first_image = cameraCalibrator.getRotationVector().data()[0];
    Mat translation_vector_first_image = cameraCalibrator.getTransportVector().data()[0];
    Rodrigues(rotation_vector_first_image, rotation_vector_first_image);

    cameraCalibrator.calibrate((Size &) second_image.size);
    Mat rotation_vector_second_image = cameraCalibrator.getRotationVector().data()[1];
    Mat translation_vector_second_image = cameraCalibrator.getTransportVector().data()[1];
    Rodrigues(rotation_vector_second_image, rotation_vector_second_image);

    Mat result_3D_points(1, 100, CV_64FC4);
    Mat rotation_translation_vector_first_image, rotation_translation_vector_second_image;

    hconcat(rotation_vector_first_image, translation_vector_first_image, rotation_translation_vector_first_image);
    hconcat(rotation_vector_second_image, translation_vector_second_image, rotation_translation_vector_second_image);

    Mat camera_matrix_a = cameraCalibrator.getCameraMatrix() * rotation_translation_vector_first_image;
    Mat camera_matrix_b = cameraCalibrator.getCameraMatrix() * rotation_translation_vector_second_image;
    triangulatePoints(camera_matrix_a, camera_matrix_b, detection_points_first_image, detection_points_second_image,
                      result_3D_points);

    double w = result_3D_points.at<double>(3, 0);
    double x = result_3D_points.at<double>(0, 0) / w;
    double y = result_3D_points.at<double>(1, 0) / w;
    double z = result_3D_points.at<double>(2, 0) / w;

    /*cout << "Value of X: " << x << endl;
    cout << "Value of Y: " << y << endl;
    cout << "Value of Z: " << z << endl;*/

    Mat triangulation_3D_points;
    transpose(result_3D_points, triangulation_3D_points);
    convertPointsFromHomogeneous(triangulation_3D_points, triangulation_3D_points);

    vector<Point3f> list_3D_points_after_triangulation;
    vector<Point2f> list_2D_points_after_triangulation;
    for (int i = 0; i < triangulation_3D_points.rows; i++) {
        list_3D_points_after_triangulation.push_back(
                Point3f(triangulation_3D_points.at<float>(i, 0), triangulation_3D_points.at<float>(i, 1),
                        triangulation_3D_points.at<float>(i, 2)));
    }

    projectPoints(list_3D_points_after_triangulation, rotation_vector_first_image, translation_vector_first_image,
                  cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(),
                  list_2D_points_after_triangulation);


    draw2DPoints(first_image, list_2D_points_after_triangulation, blue);
    Mat frame_with_triangulation = first_image.clone();

    /*************************************************************
     *                    * Registration *                       *
     *************************************************************/

    Mat ref_image = imread(path_to_ref_image);

    namedWindow(WIN_REF_IMAGE_FOR_USER);
    namedWindow(WIN_USER_SELECT_POINT, WINDOW_KEEPRATIO);
    setMouseCallback(WIN_USER_SELECT_POINT, onMouseModelRegistration, 0);

    Mat clone_of_ref_image = ref_image.clone();

    if (!ref_image.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    vector<Point2f> list;
    int i = 0;
    while (registration.get_points3d().size() < number_registration) {
        if (list_3D_points_after_triangulation[i].x < 100.0 && list_3D_points_after_triangulation[i].x > -100.0 &&
            list_3D_points_after_triangulation[i].y < 100.0 &&
            list_3D_points_after_triangulation[i].y > -100.0) {
            registration.register3DPoint(list_3D_points_after_triangulation[i]);
            list.push_back(list_2D_points_after_triangulation[i]);
        }
        i++;
    }

    registration.setNumMax(number_registration);
    vector<Point2f> list_points2d;
    vector<Point3f> list_points3d;
    while (waitKey(30) < 0) {

        clone_of_ref_image = ref_image.clone();
        list_points2d = registration.get_points2d();
        list_points3d = registration.get_points3d();
        if (!end_registration) {
            drawCounter(clone_of_ref_image, registration.getNumRegistration(), registration.getNumMax(), red);
            draw2DPoint(frame_with_triangulation, list[registration.getNumRegistration()], green);
            Point3f point3f = registration.get_points3d()[registration.getNumRegistration()];
            drawQuestion(clone_of_ref_image, point3f, red);
        } else {
            drawText(clone_of_ref_image, "END REGISTRATION", green);
            drawCounter(clone_of_ref_image, registration.getNumRegistration(), registration.getNumMax(), green);
            break;
        }

        draw2DPoints(clone_of_ref_image, list_points2d, blue);
        imshow(WIN_USER_SELECT_POINT, clone_of_ref_image);
        imshow(WIN_REF_IMAGE_FOR_USER, frame_with_triangulation);
    }


    /*************************************************************
     *                   * Point Estimation *
     *************************************************************/

    double cx, cy;

    double camera_parameters[4];
    camera_parameters[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    camera_parameters[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    camera_parameters[2] = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
    camera_parameters[3] = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

    PnPProblem pnp_registration(camera_parameters);

    vector<KeyPoint> key_points_ref_image;
    Mat descriptors_of_ref_image;

    robustMatcher.getExtractor()->detect(ref_image, key_points_ref_image);
    robustMatcher.getDetector()->compute(ref_image, key_points_ref_image, descriptors_of_ref_image);

    for (int i = 0; i < descriptors_of_ref_image.rows; i++) {
        model.add_descriptor(descriptors_of_ref_image.row(i));
    }

    vector<Point3f> list_3D_points_after_registration = registration.get_points3d();
    vector<Point2f> registration_2DPoint = registration.get_points2d();

    for (int i = 0; i < list_3D_points_after_registration.size(); i++) {
        model.add_correspondence(registration_2DPoint[i], list_3D_points_after_registration[i]);
    }

    /*************************************************************
     *                   * Vanish point *
     *************************************************************/

    vector<Mat> vanish_point;
    Mat output_ref_image, gray_ref_image;
    Size size_ref_image;

    int width = ref_image.cols;
    int height = ref_image.rows;
    size_ref_image = Size(width, height);

    msac.init(mode, size_ref_image, verbose);
    resize(ref_image, ref_image, size_ref_image);

    if (ref_image.channels() == 3) {
        cvtColor(ref_image, gray_ref_image, CV_BGR2GRAY);
        ref_image.copyTo(output_ref_image);
    } else {
        ref_image.copyTo(gray_ref_image);
        cvtColor(ref_image, output_ref_image, CV_GRAY2BGR);
    }

    vanish_point = processImage(msac, numVps, gray_ref_image, output_ref_image);
    imshow(WIN_REF_IMAGE_WITH_HOUGH_LINES, output_ref_image);

    vector<Point3f> vanish_point_3d;
    vector<Point2f> vanish_point_2d;
    for (int i = 0; i < vanish_point.size(); i++) {
        vanish_point_2d.push_back(Point2f(vanish_point[i].at<float>(0, 0), vanish_point[i].at<float>(1, 0)));
    }

    Point2f A = vanish_point_2d[0];
    Point2f B = vanish_point_2d[1];
    Point2f C = vanish_point_2d[2];

    Point2f center, tmp;
    center.x = (A.x + B.x) / 2;
    center.y = (A.y + B.y) / 2;
    tmp.x = C.x;
    tmp.y = C.y;
    line(ref_image, center, tmp, green, 1);
    Line2D primka1(center.x, center.y, tmp.x, tmp.y);

    center.x = (A.x + C.x) / 2;
    center.y = (A.y + C.y) / 2;
    tmp.x = B.x;
    tmp.y = B.y;
    line(ref_image, center, tmp, green, 1);
    Line2D primka2(center.x, center.y, tmp.x, tmp.y);

    center.x = (C.x + B.x) / 2;
    center.y = (C.y + B.y) / 2;
    tmp.x = A.x;
    tmp.y = A.y;
    line(ref_image, center, tmp, green, 1);
    line(ref_image, center, tmp, green, 1);
    Line2D primka3(center.x, center.y, tmp.x, tmp.y);

    double prusecik_x, prusecik_y;
    if (primka1.getIntersection(primka2, prusecik_x, prusecik_y)) {
        printf("Prusecik [%f; %f]\n", prusecik_x, prusecik_y);
    }

    cx = prusecik_x;
    cy = prusecik_y;

    namedWindow(WIN_REF_IMAGE_WITH_VANISH_POINTS);
    imshow(WIN_REF_IMAGE_WITH_VANISH_POINTS, ref_image);


    for (int i = 0; i < list_3D_points_after_registration.size(); i++) {
        cout << list_3D_points_after_registration[i] << endl;
    }

    /*************************************************************
     *                   * Save registration *
     *************************************************************/

    Mat r, t;
    vector<Point2f> ll;
    vector<Point2f> kk;

    Mat essential = findEssentialMat(key_points_first_image, key_points_second_image, cameraCalibrator.getCameraMatrix());
    correctMatches(essential, key_points_first_image, key_points_second_image, key_points_first_image, key_points_second_image);
    recoverPose(essential, key_points_first_image, key_points_second_image, r, t, 1, Point2f(cx, cy));

    double xAngle = atan2f(r.at<float>(2, 1), r.at<float>(2, 2));
    double yAngle = atan2f(-r.at<float>(2, 0),
                           sqrtf(r.at<float>(2, 1) * r.at<float>(2, 1) + r.at<float>(2, 2) * r.at<float>(2, 2)));
    double zAngle = atan2f(r.at<float>(1, 0), r.at<float>(0, 0));

    xAngle = (int) convert_radian_to_degree(xAngle);
    yAngle = (int) convert_radian_to_degree(yAngle);
    zAngle = (int) convert_radian_to_degree(zAngle);

    cout << "x angle: " << xAngle << "%" << endl;
    cout << "y angle: " << yAngle << "%" << endl;
    cout << "z angle: " << zAngle << "%" << endl;

    /*
     * [ fx   0  cx ]
     * [  0  fy  cy ]
     * [  0   0   1 ]
     */
    Mat camera_matrix_ref = cv::Mat::zeros(3, 3, CV_64FC1);
    camera_matrix_ref.at<double>(0, 0) = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    camera_matrix_ref.at<double>(1, 1) = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    camera_matrix_ref.at<double>(0, 2) = cx;
    camera_matrix_ref.at<double>(1, 2) = cy;
    camera_matrix_ref.at<double>(2, 2) = 1;

    model.set_camera_matrix(camera_matrix_ref);
    model.save("result.yml");

    waitKey(0);
    return 0;
}


