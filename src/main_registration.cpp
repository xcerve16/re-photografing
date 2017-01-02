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
Mesh mesh;
MSAC msac;

String frame1 = "resource/image/test1.jpg";
String frame2 = "resource/image/test2.jpg";
String ref = "resource/image/refVelehrad.jpg";

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

vector<Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg) {
    cv::Mat imgCanny;

    // Canny
    cv::Canny(imgGRAY, imgCanny, 180, 120, 3);

    // Hough
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
        /*circle(outputImg, pt1, 2, CV_RGB(255,255,255), CV_FILLED);
        circle(outputImg, pt1, 3, CV_RGB(0,0,0),1);
        circle(outputImg, pt2, 2, CV_RGB(255,255,255), CV_FILLED);
        circle(outputImg, pt2, 3, CV_RGB(0,0,0),1);*/

        // Store into vector of pairs of Points for msac
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

    Mat image1 = imread(frame1);
    Mat image2 = imread(frame2);

    cameraCalibrator.cleanVectors();
    cameraCalibrator.calibrate((Size &) image1.size);

    robustMatcher.setConfidenceLevel(0.98);
    robustMatcher.setMinDistanceToEpipolar(1.0);
    robustMatcher.setRatio(0.65f);

    Ptr<FeatureDetector> pfd = SURF::create(10);
    robustMatcher.setFeatureDetector(pfd);

    vector<DMatch> matches;
    vector<KeyPoint> keyPoints1, keyPoints2;
    Mat fundamental = robustMatcher.match(image1, image2, matches, keyPoints1, keyPoints2);

    vector<Point2f> points1, points2;

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = keyPoints1[it->queryIdx].pt.x;
        float y = keyPoints1[it->queryIdx].pt.y;
        points1.push_back(Point2f(x, y));
        circle(image1, Point((int) x, (int) y), 3, white, 3);

        x = keyPoints2[it->trainIdx].pt.x;
        y = keyPoints2[it->trainIdx].pt.y;
        circle(image2, Point((int) x, (int) y), 3, white, 3);
        points2.push_back(Point2f(x, y));
    }

    vector<Vec3f> lines1;
    computeCorrespondEpilines(Mat(points1), 1, fundamental, lines1);

    for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it) {
        line(image2, Point(0, -(*it)[2] / (*it)[1]),
             Point(image2.cols, (int) (-((*it)[2] + (*it)[0] * image2.cols) / (*it)[1])), white);
    }

    vector<Vec3f> lines2;
    computeCorrespondEpilines(Mat(points2), 2, fundamental, lines2);

    for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it) {
        line(image1, Point(0, -(*it)[2] / (*it)[1]),
             Point(image1.cols, (int) (-((*it)[2] + (*it)[0] * image1.cols) / (*it)[1])), white);
    }

    /*namedWindow("Right Image (RANSAC)");
    imshow("Right Image (RANSAC)", image1);
    namedWindow("Left Image (RANSAC)");
    imshow("Left Image (RANSAC)", image2);*/

    /*************************************************************
     *                   * Triangulation *
     *************************************************************/
    cameraCalibrator.cleanVectors();
    cameraCalibrator.calibrate((Size &) image1.size);
    Mat rotation_vector_a = cameraCalibrator.getRotationVector().data()[0];
    Mat translation_vector_a = cameraCalibrator.getTransportVector().data()[0];
    Rodrigues(rotation_vector_a, rotation_vector_a);

    cameraCalibrator.cleanVectors();
    cameraCalibrator.calibrate((Size &) image2.size);
    Mat rotation_vector_b = cameraCalibrator.getRotationVector().data()[1];
    Mat translation_vector_b = cameraCalibrator.getTransportVector().data()[1];
    Rodrigues(rotation_vector_b, rotation_vector_b);

    Mat pnts3D(1, 100, CV_64FC4);

    Mat rt_a;
    Mat rt_d;

    hconcat(rotation_vector_a, translation_vector_a, rt_a);
    hconcat(rotation_vector_b, translation_vector_b, rt_d);

    Mat camera_matrix_a = cameraCalibrator.getCameraMatrix() * rt_a;
    Mat camera_matrix_b = cameraCalibrator.getCameraMatrix() * rt_d;
    triangulatePoints(camera_matrix_a, camera_matrix_b, points1, points2, pnts3D);

    double w = pnts3D.at<double>(3, 0);
    double x = pnts3D.at<double>(0, 0) / w;
    double y = pnts3D.at<double>(1, 0) / w;
    double z = pnts3D.at<double>(2, 0) / w;

    cout << "Value of X: " << x << endl;
    cout << "Value of Y: " << y << endl;
    cout << "Value of Z: " << z << endl;

    Mat triangulatedPoints3D;
    transpose(pnts3D, triangulatedPoints3D);
    convertPointsFromHomogeneous(triangulatedPoints3D, triangulatedPoints3D);

    vector<Point3f> list3DPoint;
    vector<Point2f> list2DPoint;
    for (int i = 0; i < triangulatedPoints3D.rows; i++) {
        list3DPoint.push_back(Point3f(triangulatedPoints3D.at<float>(i, 0), triangulatedPoints3D.at<float>(i, 1),
                                      triangulatedPoints3D.at<float>(i, 2)));
    }

    projectPoints(list3DPoint, rotation_vector_b, translation_vector_b, cameraCalibrator.getCameraMatrix(),
                  cameraCalibrator.getDistCoeffs(), list2DPoint);


    Mat xx = imread(frame1);

    draw2DPoints(xx, list2DPoint, blue);

    /*namedWindow("Triangulation 1");
    imshow("Triangulation 1", xx);*/

    projectPoints(list3DPoint, rotation_vector_a, translation_vector_a, cameraCalibrator.getCameraMatrix(),
                  cameraCalibrator.getDistCoeffs(), list2DPoint);

    Mat yy = imread(frame2);

    draw2DPoints(yy, list2DPoint, blue);

    /*namedWindow("Triangulation 2");
    imshow("Triangulation 2", yy);*/


    /*************************************************************
     *                    * Registration *                       *
     *************************************************************/

    Mat image3 = imread(ref);


    namedWindow("MODEL REGISTRATION", WINDOW_KEEPRATIO);
    setMouseCallback("MODEL REGISTRATION", onMouseModelRegistration, 0);

    Mat img_vis = image3.clone();

    if (!image3.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }


    vector<Point2f> list;
    int i = 0;
    while (registration.get_points3d().size() < number_registration) {
        if (list3DPoint[i].x < 100.0 && list3DPoint[i].x > -100.0 && list3DPoint[i].y < 100.0 &&
            list3DPoint[i].y > -100.0 && list3DPoint[i].z < 100.0 && list3DPoint[i].z > -100.0) {
            registration.register3DPoint(list3DPoint[i]);
            list.push_back(list2DPoint[i]);
        }
        i++;
    }


    registration.setNumMax(number_registration);
    vector<Point2f> list_points2d;
    vector<Point3f> list_points3d;
    while (waitKey(30) < 0) {

        img_vis = image3.clone();
        list_points2d = registration.get_points2d();
        list_points3d = registration.get_points3d();
        if (!end_registration) {
            drawCounter(img_vis, registration.getNumRegistration(), registration.getNumMax(), red);
            draw2DPoint(yy, list[registration.getNumRegistration()], green);
            Point3f point3f = registration.get_points3d()[registration.getNumRegistration()];
            drawQuestion(img_vis, point3f, red);
        } else {
            drawText(img_vis, "END REGISTRATION", green);
            drawCounter(img_vis, registration.getNumRegistration(), registration.getNumMax(), green);
            break;
        }

        draw2DPoints(img_vis, list_points2d, blue);
        imshow("MODEL REGISTRATION", img_vis);
        imshow("MODEL REGISTRATION 1", yy);
    }


    /*************************************************************
     *                   * Point Estimation *
     *************************************************************/

    /*line(img_vis, list_points2d[0], list_points2d[1], blue, 5);
    line(img_vis, list_points2d[1], list_points2d[2], blue, 5);
    line(img_vis, list_points2d[2], list_points2d[3], blue, 5);
    line(img_vis, list_points2d[3], list_points2d[0], blue, 5);

    /* namedWindow("Point estimation");
     imshow("Point estimation", img_vis);*/

    double cx, cy;

    double parameters[4];
    parameters[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    parameters[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    parameters[2] = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
    parameters[3] = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

    PnPProblem pnp_registration(parameters);

    vector<KeyPoint> keyPoints3;
    Mat descriptors;

    robustMatcher.getExtractor()->detect(image3, keyPoints3);
    robustMatcher.getDetector()->compute(image3, keyPoints3, descriptors);

    for (int i = 0; i < descriptors.rows; i++) {
        model.add_descriptor(descriptors.row(i));
    }

    vector<Point3f> registration_3DPoint = registration.get_points3d();
    vector<Point2f> registration_2DPoint = registration.get_points2d();
    
    for (int i = 0; i < list3DPoint.size(); i++) {
        model.add_correspondence(registration_2DPoint[i], registration_3DPoint[i]);
    }


    /*************************************************************
     *                   * Vanish point *
     *************************************************************/

    vector<Mat> vanish_point;

    // Images
    cv::Mat inputImg = imread("resource/image/refVelehrad.jpg"), imgGRAY;
    cv::Mat outputImg;

    int mode = MODE_NIETO;
    int numVps = 3;
    bool verbose = false;

    cv::Size procSize;

    int width = inputImg.cols;
    int height = inputImg.rows;
    procSize = cv::Size(width, height);

    MSAC msac;
    msac.init(mode, procSize, verbose);
    cv::resize(inputImg, inputImg, procSize);
    
    if (inputImg.channels() == 3) {
        cv::cvtColor(inputImg, imgGRAY, CV_BGR2GRAY);
        inputImg.copyTo(outputImg);
    } else {
        inputImg.copyTo(imgGRAY);
        cv::cvtColor(inputImg, outputImg, CV_GRAY2BGR);
    }

    vanish_point = processImage(msac, numVps, imgGRAY, outputImg);


    // View
    imshow("Output", outputImg);


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
    line(image3, center, tmp, green, 1);
    Line2D primka1(center.x, center.y, tmp.x, tmp.y);

    center.x = (A.x + C.x) / 2;
    center.y = (A.y + C.y) / 2;
    tmp.x = B.x;
    tmp.y = B.y;
    line(image3, center, tmp, green, 1);
    Line2D primka2(center.x, center.y, tmp.x, tmp.y);

    center.x = (C.x + B.x) / 2;
    center.y = (C.y + B.y) / 2;
    tmp.x = A.x;
    tmp.y = A.y;
    line(image3, center, tmp, green, 1);
    line(image3, center, tmp, green, 1);
    Line2D primka3(center.x, center.y, tmp.x, tmp.y);


    double prusecik_x, prusecik_y;
    if (primka1.getIntersection(primka2, prusecik_x, prusecik_y)) {
        printf("Prusecik [%f; %f]\n", prusecik_x, prusecik_y);
    }

    cx = prusecik_x;
    cy = prusecik_y;


    /*************************************************************
     *                   * Save registration *
     *************************************************************/


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

    namedWindow("Final");
    imshow("Final", image3);

    waitKey(0);
    return 0;
}


