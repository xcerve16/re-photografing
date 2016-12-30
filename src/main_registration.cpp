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
#include "MyMatcher.h"
#include "Line2D.h"

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

String frame1 = "image/test1.jpg";
String frame2 = "image/test2.jpg";
String ref = ;

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


int main(int argc, char *argv[]) {

    /*************************************************************
     *                   * Kalibrace *
     *************************************************************/

    Mat image;
    vector<string> fileList;
    for (int i = 1; i <= 20; i++) {
        stringstream str;
        str << "image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
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

    /*namedWindow("Right Image Epilines (RANSAC)");
    imshow("Right Image Epilines (RANSAC)", image1);
    namedWindow("Left Image Epilines (RANSAC)");
    imshow("Left Image Epilines (RANSAC)", image2);*/

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
        Point3f point3f = Point3f(triangulatedPoints3D.at<float>(i, 0), triangulatedPoints3D.at<float>(i, 1),
                                  triangulatedPoints3D.at<float>(i, 2));
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

    while (waitKey(30) < 0) {

        img_vis = image3.clone();
        vector<Point2f> list_points2d = registration.get_points2d();
        vector<Point3f> list_points3d = registration.get_points3d();
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

    vector<Point2f> list_points2d = registration.get_points2d();

    /*************************************************************
     *                   * Point Estimation *
     *************************************************************/

    line(img_vis, list_points2d[0], list_points2d[1], blue, 5);
    line(img_vis, list_points2d[1], list_points2d[2], blue, 5);
    line(img_vis, list_points2d[2], list_points2d[3], blue, 5);
    line(img_vis, list_points2d[3], list_points2d[0], blue, 5);

    /* namedWindow("Point estimation");
     imshow("Point estimation", img_vis);*/

    double cx,cy;

    double parameters[4];
    parameters[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    parameters[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    parameters[2] = img_vis.cols / 2; // nutno jeste vypocitat
    parameters[3] = img_vis.rows / 2; // nutno jeste vypocitat

    PnPProblem pnp_registration(parameters);

    vector<KeyPoint> keyPoints3;
    Mat descriptors;

    robustMatcher.getExtractor()->detect(image3, keyPoints3);
    robustMatcher.getDetector()->compute(image3, keyPoints3, descriptors);


    Mat rvect, tvect;
    Mat out;

    vector<Point3f> list3D = registration.get_points3d();
    vector<Point2f> list2D = registration.get_points2d();


    pnp_registration.mySolvePnPRansac(list3D, list2D, rvect, tvect);

    solvePnPRansac(list3D, list2D, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(), rvect, tvect);


    vector<Point3f> nose_end_point3D;
    vector<Point2f> nose_end_point2D;
    vector<Point2f> pom;
    nose_end_point3D.push_back(Point3d(0, 0, 0));

    projectPoints(nose_end_point3D, rvect, tvect, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(),
                  nose_end_point2D);

    draw2DPoints(image3, nose_end_point2D, blue);
    for (int i = 0; i < list2D.size(); i++) {
        for (int j = 0; j < nose_end_point2D.size(); j++) {
            line(image3, nose_end_point2D[j], list2D[i], blue, 3);
        }
    }

    pom.push_back(nose_end_point2D[0]);
    nose_end_point3D.clear();
    nose_end_point2D.clear();


    nose_end_point3D.push_back(Point3d(25, 0, 0));
    projectPoints(nose_end_point3D, rvect, tvect, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(),
                  nose_end_point2D);

    draw2DPoints(image3, nose_end_point2D, blue);
    for (int i = 0; i < list2D.size(); i++) {
        for (int j = 0; j < nose_end_point2D.size(); j++) {
            line(image3, nose_end_point2D[j], list2D[i], blue, 3);
        }

    }

    pom.push_back(nose_end_point2D[0]);
    nose_end_point3D.clear();
    nose_end_point2D.clear();

    nose_end_point3D.push_back(Point3d(0, 25, 0));

    projectPoints(nose_end_point3D, rvect, tvect, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(),
                  nose_end_point2D);

    draw2DPoints(image3, nose_end_point2D, blue);
    for (int i = 0; i < list2D.size(); i++) {
        for (int j = 0; j < nose_end_point2D.size(); j++) {
            line(image3, nose_end_point2D[j], list2D[i], blue, 3);
        }

    }

    pom.push_back(nose_end_point2D[0]);

    Point2f A = pom[0];
    Point2f B = pom[1];
    Point2f C = pom[2];

    line(image3, A, B, red, 1);
    line(image3, B, C, red, 1);
    line(image3, C, A, red, 1);

    Point2f center, tmp;
    center.x = (A.x + B.x) / 2;
    center.y = (A.y + B.y) / 2;
    tmp.x = C.x;
    tmp.y = C.y;
    line(image3, center, tmp, green, 5);
    Line2D primka1(center.x , center.y , tmp.x, tmp.y);

    center.x = (A.x + C.x) / 2;
    center.y = (A.y + C.y) / 2;
    tmp.x = B.x;
    tmp.y = B.y;
    line(image3, center, tmp, green, 5);
    Line2D primka2(center.x , center.y , tmp.x, tmp.y);

    center.x = (C.x + B.x) / 2;
    center.y = (C.y + B.y) / 2;
    tmp.x = A.x;
    tmp.y = A.y;
    line(image3, center, tmp, green, 5);
    Line2D primka3(center.x , center.y , tmp.x, tmp.y);


    double prusecik_x, prusecik_y;
    if(primka1.getIntersection(primka2, prusecik_x, prusecik_y))    {
        printf("Prusecik [%f; %f]\n", prusecik_x, prusecik_y);
    }

    cx = prusecik_x;
    cy = prusecik_y;

    namedWindow("Final");
    imshow("Final", image3);

    waitKey(0);
    return 0;
}


