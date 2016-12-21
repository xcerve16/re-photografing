//
// Created by acervenka2 on 15.12.2016.
//

// C++
#include <iostream>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iomanip>
#include "opencv2/xfeatures2d.hpp"

// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "CameraCalibrator.h"
#include "Utils.h"
#include "MyMatcher.h"
#include "RobustMatcher.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

/**  GLOBAL VARIABLES  **/

string ply_read_path = "data/box.ply";          // object mesh
string write_path = "data/cookies_ORB.yml";     // output file
string yml_read_path = "data/cookies_ORB.yml"; // 3dpts + descriptors


// Boolean the know if the registration it's done
bool end_registration = false;

// Intrinsic camera parameters: UVC WEBCAM
double f = 45; // focal length in mm
double sx = 22.3, sy = 14.9;
double width = 2592, height = 1944;
double params_CANON[] = {width * f / sx,   // fx
                         height * f / sy,  // fy
                         width / 2,      // cx
                         height / 2};    // cy


int const number_registration = 8;
int pts[] = {1, 2, 3, 4, 5, 6, 7, 8}; // 3 -> 4

// Some basic colors
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar blue(255, 0, 0);

ModelRegistration registration;
Model model;
Mesh mesh;
PnPProblem pnp_registration(params_CANON);
CameraCalibrator cameraCalibrator;
MyRobustMatcher rmatcher;

static void onMouseModelRegistration(int event, int x, int y, int, void *) {
    if (event == EVENT_LBUTTONUP) {

        int n_vertex = pts[registration.getNumRegistration()];
        Point2f point_2d = Point2f((float) x, (float) y);
        Point3f point_3d = mesh.getVertex(n_vertex - 1);

        bool is_registrable = registration.is_registrable();
        if (is_registrable) {
            registration.registerPoint(point_2d, point_3d);
            if (registration.getNumRegistration() == registration.getNumMax()) end_registration = true;
        }
    }
}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        return -1;
    }


    /*************************************************************
     *                   * Kalibrace *
     *************************************************************/

    Mat image;
    vector<string> filelist;
    for (int i = 1; i <= 20; i++) {
        stringstream str;
        str << "image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        filelist.push_back(str.str());
        image = imread(str.str(), 0);
    }

    Size boardSize(6, 4);
    cameraCalibrator.addChessboardPoints(filelist, boardSize);
    cameraCalibrator.calibrate((Size &) image.size);

    /*************************************************************
     *                * Robust matcher *
     *************************************************************/

    Mat image1 = imread("image/church01.jpg", 0);
    Mat image2 = imread("image/church02.jpg", 0);
    if (!image1.data || !image2.data)
        return 0;

    cameraCalibrator.cleanVectors();
    cameraCalibrator.calibrate((Size &) image1.size);
    cameraCalibrator.calibrate((Size &) image2.size);


    rmatcher.setConfidenceLevel(0.98);
    rmatcher.setMinDistanceToEpipolar(1.0);
    rmatcher.setRatio(0.65f);

    Ptr<FeatureDetector> pfd = SURF::create(10);
    rmatcher.setFeatureDetector(pfd);

    vector<DMatch> matches;
    vector<KeyPoint> keypoints1, keypoints2;
    Mat fundemental = rmatcher.match(image1, image2, matches, keypoints1, keypoints2);

    vector<Point2f> points1, points2;

    for (vector<DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(Point2f(x, y));
        circle(image1, Point((int) x, (int) y), 3, Scalar(255, 255, 255), 3);

        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        circle(image2, Point((int) x, (int) y), 3, Scalar(255, 255, 255), 3);
        points2.push_back(Point2f(x, y));
    }

    vector<Vec3f> lines1;
    computeCorrespondEpilines(Mat(points1), 1, fundemental, lines1);

    for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it) {
        line(image2, Point(0, -(*it)[2] / (*it)[1]),
             Point(image2.cols, (int) (-((*it)[2] + (*it)[0] * image2.cols) / (*it)[1])), Scalar(255, 255, 255));
    }

    vector<Vec3f> lines2;
    computeCorrespondEpilines(Mat(points2), 2, fundemental, lines2);

    for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it) {
        line(image1, Point(0, -(*it)[2] / (*it)[1]),
             Point(image1.cols, (int) (-((*it)[2] + (*it)[0] * image1.cols) / (*it)[1])), Scalar(255, 255, 255));
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

    Mat rvec, tvec;

    solvePnPRansac(triangulatedPoints3D, points1, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(),
                   rvec, tvec);
    cout << "Rotation Vector: " << rvec << endl;
    cout << "Translation Vector: " << tvec << endl;

    /*************************************************************
     *                    * Registration *                       *
     *************************************************************/

    mesh.load(ply_read_path);

    namedWindow("MODEL REGISTRATION", WINDOW_KEEPRATIO);
    setMouseCallback("MODEL REGISTRATION", onMouseModelRegistration, 0);

    Mat img_in = imread("data/resized_IMG_3875.jpg");
    Mat img_vis = img_in.clone();

    if (!img_in.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    registration.setNumMax(number_registration);


    while (waitKey(30) < 0) {

        img_vis = img_in.clone();
        vector<Point2f> list_points2d = registration.get_points2d();
        vector<Point3f> list_points3d = registration.get_points3d();

        drawPoints(img_vis, list_points2d, list_points3d, red);

        if (!end_registration) {
            int n_vertex = pts[registration.getNumRegistration()];
            Point3f current_point3d = mesh.getVertex(n_vertex - 1);
            drawQuestion(img_vis, current_point3d, green);
            drawCounter(img_vis, registration.getNumRegistration(), registration.getNumMax(), red);
        } else {
            drawText(img_vis, "END REGISTRATION", green);
            drawCounter(img_vis, registration.getNumRegistration(), registration.getNumMax(), green);
            break;
        }
        imshow("MODEL REGISTRATION", img_vis);
    }

    vector<Point2f> list_points2d = registration.get_points2d();
    vector<Point3f> list_points3d = registration.get_points3d();

    if (pnp_registration.estimatePose(list_points3d, list_points2d, SOLVEPNP_ITERATIVE)) {
        cout << "Correspondence found" << endl;

        vector<Point2f> list_points2d_mesh = pnp_registration.verify_points(&mesh);
        draw2DPoints(img_vis, list_points2d_mesh, green);

    } else {
        cout << "Correspondence not found" << endl << endl;
    }

    vector<KeyPoint> keypoints_model;
    Mat descriptors;

    rmatcher.getExtractor()->detect(img_in, keypoints_model);
    rmatcher.getDetector()->compute(img_in, keypoints_model, descriptors);

    for (unsigned int i = 0; i < keypoints_model.size(); ++i) {
        Point2f point2d(keypoints_model[i].pt);
        Point3f point3d;
        bool on_surface = pnp_registration.backProject2DPoint(&mesh, point2d, point3d);
        if (on_surface) {
            model.add_correspondence(point2d, point3d);
            model.add_descriptor(descriptors.row(i));
            model.add_key_point(keypoints_model[i]);
        } else {
            model.add_outlier(point2d);
        }
    }

    model.save(write_path);
    img_vis = img_in.clone();

    vector<Point2f> list_points_in = model.get_points2d_in();
    vector<Point2f> list_points_out = model.get_points2d_out();

    string num = IntToString((int) list_points_in.size());
    string text = "There are " + num + " inliers";
    drawText(img_vis, text, green);

    num = IntToString((int) list_points_out.size());
    text = "There are " + num + " outliers";
    drawText2(img_vis, text, red);

    drawObjectMesh(img_vis, &mesh, &pnp_registration, blue);

    draw2DPoints(img_vis, list_points_in, green);
    draw2DPoints(img_vis, list_points_out, red);

    imshow("MODEL REGISTRATION", img_vis);

    /*************************************************************
     *                  * Pose Estimation *
     *************************************************************/

    Mat im = img_vis;

    vector<Point2f> image_points;
    vector<Point3f> model_points;
    Mat rotation_vector, translation_vector;

    for (int i = 0; i < 8; i++) {
        image_points.push_back(Point2d(list_points2d[i].x, list_points2d[i].y));
        model_points.push_back(Point3d(list_points3d[i].x, list_points3d[i].y, list_points3d[i].z));
    }

    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = Point2d(im.cols / 2, im.rows / 2);
    Mat camera_matrix = (Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    Mat dist_coeffs = Mat::zeros(4, 1, cv::DataType<double>::type);

    cout << "Camera Matrix " << endl << camera_matrix << endl;

    /*
     * Funkce pro vypocet odhadu 3D bodu
     * model_points - list 3D bodu
     * image_points - list 2D bodu
     * camera_matrix - vstupni kamera matice s optickym stredem
     * dist_coeffs - vstupni vektor zkreslenych koeficientu
     * rotation_vector_a - vystupni vektor po rotaci
     * translation_vector - vystupni vektor pro posun
     * SOLVEPNP_ITERATIVE - iterační metoda založená na Levenberg-Marquardt optimalizaci
     */
    //solvePnP(model_points, image_points, cameraCalibrator.getCameraMatrix(), cameraCalibrator.getDistCoeffs(), rotation_vector, translation_vector, SOLVEPNP_ITERATIVE );
    solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs,
                  rotation_vector, translation_vector, SOLVEPNP_ITERATIVE);

    vector<Point3d> nose_end_point3D;
    vector<Point2d> nose_end_point2D;
    nose_end_point3D.push_back(Point3d(0, 0, 1000.0));

    projectPoints(nose_end_point3D, rotation_vector, translation_vector, cameraCalibrator.getCameraMatrix(),
                  cameraCalibrator.getDistCoeffs(), nose_end_point2D);


    /*for (int i = 0; i < image_points.size(); i++) {
        circle(im, image_points[i], 3, blue, -1);
    }*/


    for (int i = 0; i < image_points.size(); i++) {
        line(im, image_points[i], nose_end_point2D[0], blue, 2);
    }

    cout << "Rotation Vector " << endl << rotation_vector << endl;
    cout << "Translation Vector" << endl << translation_vector << endl;

    cout << nose_end_point2D << endl;

    imshow("Output", im);

    waitKey(0);
    destroyWindow("MODEL REGISTRATION");

    return 0;
}


