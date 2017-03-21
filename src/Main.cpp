//
// Created by acervenka2 on 17.01.2017.
//

#include "Main.h"

int index = 0;
vector<Point2f> inliners;

bool getRobustEstimation(Mat current_frame_vis, Mat description_first_image,
                         vector<Point3f> list_3D_points_after_registration, Mat measurements, Mat refT) {

    vector<DMatch> good_matches;
    vector<KeyPoint> key_points_current_frame;
    vector<Point3f> list_points3d_model_match;
    vector<Point2f> list_points2d_scene_match;
    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;
    Mat revT;

    robustMatcher.fastRobustMatch(current_frame_vis, good_matches, key_points_current_frame, description_first_image);

    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        Point3f point3d_model = list_3D_points_after_registration[good_matches[match_index].trainIdx];
        Point2f point2d_scene = key_points_current_frame[good_matches[match_index].queryIdx].pt;
        list_points3d_model_match.push_back(point3d_model);
        list_points2d_scene_match.push_back(point2d_scene);
    }

    draw2DPoints(current_frame_vis, list_points2d_scene_match, red);

    bool good_measurement = false;

    if (good_matches.size() > 0) {

        pnp_registration.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnpMethod,
                                            inliers_idx, iterationsCount, reprojectionError, confidence);
        for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
            int n = inliers_idx.at<int>(inliers_index);
            Point2f point2d = list_points2d_scene_match[n];
            list_points2d_inliers.push_back(point2d);
        }

        Mat tvect_ref_frame = pnp_registration.get_T_matrix();
        Mat rvect_ref_frame = pnp_registration.get_R_matrix();

        Mat rvect_traspose;
        transpose(rvect_ref_frame, rvect_traspose);
        Mat pos = -rvect_traspose * tvect_ref_frame;

        Mat T(4, 4, rvect_traspose.type());
        T(cv::Range(0, 3), cv::Range(0, 3)) = rvect_traspose * 1;
        T(cv::Range(0, 3), cv::Range(3, 4)) = pos * 1;

        double *p = T.ptr<double>(3);
        p[0] = p[1] = p[2] = 0;
        p[3] = 1;


        revT = (Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        revT = T.inv() * refT;
        try {
            /*double xAngle = atan2f(revT.at<float>(2, 1), revT.at<float>(2, 2));
            double yAngle = atan2f(-revT.at<float>(2, 0), sqrtf(revT.at<float>(2, 1) * revT.at<float>(2, 1) +
                                                                revT.at<float>(2, 2) * revT.at<float>(2, 2)));
            double zAngle = atan2f(revT.at<float>(1, 0), revT.at<float>(0, 0));

            xAngle = (int) convert_radian_to_degree(xAngle);
            yAngle = (int) convert_radian_to_degree(yAngle);
            zAngle = (int) convert_radian_to_degree(zAngle);

            double posun_x = revT.at<float>(0, 3);
            double posun_y = revT.at<float>(1, 3);
            double posun_z = revT.at<float>(2, 3);

            cout << revT << endl;

            if (posun_x < -0.5) {
                cout << "Doleva " << posun_x << endl;
            } else if (posun_x > 0.5) {
                cout << "Doprava " << posun_x << endl;
            } else {
                cout << "OK " << posun_x << endl;
            }

            if (posun_y < -0.5) {
                cout << "Nahoru " << posun_y << endl;
            } else if (posun_y > 0.5) {
                cout << "Dolu " << posun_y << endl;
            } else {
                cout << "OK " << posun_y << endl;
            }


            if (posun_z < -0.5) {
                cout << "Dozadu " << posun_z << endl;
            } else if (posun_z > 0.5) {
                cout << "Dopredu " << posun_z << endl;
            } else {
                cout << "OK " << posun_z << endl;
            }*/


            if (inliers_idx.rows >= minInliersKalman) {

                Mat translation_measured(3, 1, CV_64F);
                translation_measured = pnp_registration.get_T_matrix();

                Mat rotation_measured(3, 3, CV_64F);
                rotation_measured = pnp_registration.get_R_matrix();

                good_measurement = true;
                fillMeasurements(measurements, translation_measured, rotation_measured);
            }
        } catch (Exception e) {
            cout << ERROR_COMPARE_MATRIX << endl;
        }
    }


    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);

    pnp_registration.set_P_matrix(rotation_estimated, translation_estimated);

    try {
        revT = (Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

        cout << pnp_registration.get_P_matrix() << endl;

        revT = pnp_registration.get_P_matrix().inv() * refT;


        double posun_x = revT.at<float>(0, 3);
        double posun_y = revT.at<float>(1, 3);
        double posun_z = revT.at<float>(2, 3);

        //TODO:Posun určen jen tak jednoduše

        if (posun_x < -0.5) {
            cout << "Doleva " << posun_x << endl;
        } else if (posun_x > 0.5) {
            cout << "Doprava " << posun_x << endl;
        } else {
            cout << "OK " << posun_x << endl;
        }

        if (posun_y < -0.5) {
            cout << "Nahoru " << posun_y << endl;
        } else if (posun_y > 0.5) {
            cout << "Dolu " << posun_y << endl;
        } else {
            cout << "OK " << posun_y << endl;
        }


        if (posun_z < -0.5) {
            cout << "Dozadu " << posun_z << endl;
        } else if (posun_z > 0.5) {
            cout << "Dopredu " << posun_z << endl;
        } else {
            cout << "OK " << posun_z << endl;
        }

    } catch (Exception e) {
        cout << ERROR_COMPARE_MATRIX << endl;
    }
    cout << "============================" << endl;

    //TODO: Kalman filter projekcni matice z kalmanova filtru ???

    inliners = list_points2d_inliers;

    return good_measurement;
}

bool getLightweightEstimation(Mat current_frame_vis, Mat description_first_image,
                              vector<Point3f> list_3D_points_after_registration, Mat measurements, Mat refT) {

    //TODO: Zjistit co vsechno zde má být

    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;


    bool good_measurement = false;


    if (inliers_idx.rows >= minInliersKalman) {

        Mat translation_measured(3, 1, CV_64F);
        translation_measured = pnp_registration.get_T_matrix();

        Mat rotation_measured(3, 3, CV_64F);
        rotation_measured = pnp_registration.get_R_matrix();

        fillMeasurements(measurements, translation_measured, rotation_measured);
        good_measurement = true;

    }

    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
    pnp_registration.set_P_matrix(rotation_estimated, translation_estimated);

    return good_measurement;
}


int main(int argc, char *argv[]) {

    /*double f = 2.0;
    double sx = 2.74, sy = 3.67;*/

    /*Mat tt, tref, trev, point, result_tt, result_tref, result_trev;
    tt = (Mat_<double>(3, 4)
            << 0.0856, 0.7410, -0.6661, -5.7397, -0.5719, -0.511, -0.6417, -9.1603, -0.8159, 0.4356, 0.3803, 2.5889);
    tref = (Mat_<double>(3, 4) << -0.0007, -0.0210, -0.0220, -0.0161, -0.0300, 0, 1.9161, 4.6030, 0, 0, 1.9161, 2.1821);
    trev = (Mat_<double>(3, 4)
            << 0.021, -0.0018, -0.0019, -0.0014, 0.0179, -0.0156, -0.0163, -0.0119, 0.0236, 0.0140, 0.0146, 0.0107);

    point = (Mat_<double>(4, 1) << 1, 1, 1, 1);

    result_tt = tt * point;
    result_tref = tref * point;
    result_trev = trev * point;
    cout << result_tt << endl;;
    cout << result_tref << endl;
    cout << result_trev << endl;

    return 0;*/


    /*************************************************************
     *                   * Kalibrace *
     *************************************************************/

    Mat image;
    vector<string> fileList;
    for (int i = 57; i <= 80; i++) {
        stringstream str;
        str << "resource/image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        fileList.push_back(str.str());
        image = imread(str.str(), 0);
    }

    Size boardSize(7, 9);
    cameraCalibrator.addChessboardPoints(fileList, boardSize);


    /*************************************************************
     *                * Robust matcher *
     *************************************************************/

    Mat first_image = imread(path_to_first_image);
    Mat second_image = imread(path_to_second_image);

    cameraCalibrator.calibrate((Size &) first_image.size);
    //cameraCalibrator.myremap(first_image);
    cout << cameraCalibrator.getCameraMatrix() << endl;

    /*double cx = cameraCalibrator.getCameraMatrix().at<float>(0, 2);
    double cy = cameraCalibrator.getCameraMatrix().at<float>(1, 2);
    double fx = cameraCalibrator.getCameraMatrix().at<float>(0, 0);
    double fy = cameraCalibrator.getCameraMatrix().at<float>(1, 1);*/

    //TODO: zjistit hodnoty proč jsou tak malé

    /*double cx = 13355.703125;
    double cy = 1494.664429;
    double fx = 6819.694824;
    double fy = 4906.716797;*/

    double cx = 222.674975;
    double cy = 222.675975;
    double fx = 273.9448723609939;
    double fy = 234.4292645984778;

    pnp_registration.setMatrixParam(fx, fy, cx, cy);

    if (!first_image.data) {
        cout << ERROR_READ_IMAGE << endl;
        return -1;
    }

    if (!first_image.data) {
        cout << ERROR_READ_IMAGE << endl;
        return -1;
    }

    robustMatcher.setConfidenceLevel(confidenceLevel);
    robustMatcher.setMinDistanceToEpipolar(min_dist);
    robustMatcher.setRatio(ratioTest);

    Ptr<FeatureDetector> featureDetector = SURF::create(numKeyPoints);
    robustMatcher.setFeatureDetector(featureDetector);

    vector<DMatch> matches;
    vector<KeyPoint> key_points_first_image, key_points_second_image;
    Mat descriptor_first_image;

    Mat fundamental = robustMatcher.match(first_image, second_image, matches, key_points_first_image,
                                          key_points_second_image);

    vector<Point2f> detection_points_first_image, detection_points_second_image;

    Mat img1 = first_image.clone();
    Mat img2 = second_image.clone();

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = key_points_first_image[it->queryIdx].pt.x;
        float y = key_points_first_image[it->queryIdx].pt.y;
        detection_points_first_image.push_back(Point2f(x, y));
        circle(img1, Point((int) x, (int) y), 3, white, 3);

        x = key_points_second_image[it->trainIdx].pt.x;
        y = key_points_second_image[it->trainIdx].pt.y;
        detection_points_second_image.push_back(Point2f(x, y));
        circle(img2, Point((int) x, (int) y), 3, white, 3);
    }

    resize(fundamental, fundamental, Size(3, 3));


    std::vector<uchar> inliers(detection_points_first_image.size(), 0);
    Mat essencial = pnp_registration.get_A_matrix() * fundamental * pnp_registration.get_A_matrix();

    Mat R1, R2, t;
    decomposeEssentialMat(essencial, R1, R2, t);


    /*************************************************************
     *                   * Triangulation *
     *************************************************************/

    Mat rotation_translation_vector_first_image, rotation_translation_vector_second_image, result_3D_points,
            rotation_vector_first_image, translation_vector_first_image, finded_3D_points;
    rotation_translation_vector_second_image = Mat::eye(3, 4, CV_64FC1);

    double cout_z[4] = {0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        switch (i) {
            case 0:
                rotation_vector_first_image = R1;
                translation_vector_first_image = t;
                break;
            case 1:
                rotation_vector_first_image = R2;
                translation_vector_first_image = t;
                break;
            case 2:
                rotation_vector_first_image = R1;
                translation_vector_first_image = t * -1;
                break;
            case 3:
                rotation_vector_first_image = R2;
                translation_vector_first_image = t * -1;
                break;
            default:
                break;
        }

        hconcat(rotation_vector_first_image, translation_vector_first_image, rotation_translation_vector_first_image);

        Mat camera_matrix_a = pnp_registration.get_A_matrix() * rotation_translation_vector_first_image;
        Mat camera_matrix_b = pnp_registration.get_A_matrix() * rotation_translation_vector_second_image;

        triangulatePoints(camera_matrix_a, camera_matrix_b, detection_points_first_image, detection_points_second_image,
                          finded_3D_points);

        Mat triangulation_3D_points;
        transpose(finded_3D_points, triangulation_3D_points);
        convertPointsFromHomogeneous(triangulation_3D_points, triangulation_3D_points);

        for (int j = 0; j < triangulation_3D_points.rows; j++) {
            if (triangulation_3D_points.at<float>(j, 2) > 0) {
                cout_z[i]++;
            }
        }

        if (result_3D_points.empty() or cout_z[i - 1] < cout_z[i]) {
            result_3D_points = triangulation_3D_points;
        }
    }

    vector<Point3f> list_3D_points_after_triangulation;
    vector<Point2f> list_2D_points_after_triangulation;
    for (int i = 0; i < result_3D_points.rows; i++) {
        list_3D_points_after_triangulation.push_back(
                Point3f(result_3D_points.at<float>(i, 0), result_3D_points.at<float>(i, 1),
                        result_3D_points.at<float>(i, 2)));
    }

    pnp_registration.myProjectPoints(list_3D_points_after_triangulation, rotation_vector_first_image,
                                     translation_vector_first_image, list_2D_points_after_triangulation);

    Point2f center1 = Point2f(cx, cy);
    Mat img3 = first_image.clone();
    draw2DPoints(img3, list_2D_points_after_triangulation, blue);
    draw2DPoint(img3, center1, green);
    Mat frame_with_triangulation = img3;

    /* namedWindow("Right Image (Triangulation)");
     imshow("Right Image (Triangulation)", img3);*/

    cout << list_3D_points_after_triangulation << endl;

    /*************************************************************
     *                    * Registration *                       *
     *************************************************************/

    namedWindow(WIN_REF_IMAGE_FOR_USER);
    namedWindow(WIN_USER_SELECT_POINT);

    setMouseCallback(WIN_USER_SELECT_POINT, onMouseModelRegistration, 0);

    Mat ref_image = imread(path_to_ref_image);

    if (!ref_image.data) {
        cout << ERROR_READ_IMAGE << endl;
        return -1;
    }

    registration.setNumMax(number_registration);
    vector<Point2f> list_points2d;
    vector<Point3f> list_points3d;

    int previousNumRegistration = registration.getNumRegistration();
    vector<int> index_of_points;

    while (waitKey(30) < 0) {

        Mat clone_of_ref_image = ref_image.clone();
        list_points2d = registration.get_points2d();
        list_points3d = registration.get_points3d();
        if (!end_registration) {
            drawCounter(clone_of_ref_image, registration.getNumRegistration(), registration.getNumMax(), red);
            draw2DPoint(frame_with_triangulation, detection_points_first_image[index], green);
            Point3f point3f = list_3D_points_after_triangulation[index];
            drawQuestion(clone_of_ref_image, point3f, red);

            if (previousNumRegistration != registration.getNumRegistration()) {
                index_of_points.push_back(index);
                registration.register3DPoint(point3f);
                previousNumRegistration = registration.getNumRegistration();
            }

        } else {
            Point3f point3f = list_3D_points_after_triangulation[index];
            registration.register3DPoint(point3f);
            index_of_points.push_back(index);
            draw2DPoints(clone_of_ref_image, list_points2d, blue);
            break;
        }

        draw2DPoints(clone_of_ref_image, list_points2d, blue);
        imshow(WIN_USER_SELECT_POINT, clone_of_ref_image);
        imshow(WIN_REF_IMAGE_FOR_USER, frame_with_triangulation);
    }

    destroyWindow(WIN_USER_SELECT_POINT);
    destroyWindow(WIN_REF_IMAGE_FOR_USER);

    vector<Point3f> list_3D_points_after_registration = registration.get_points3d();
    vector<Point2f> list_2D_points_after_registration = registration.get_points2d();

    /*************************************************************
     *                   * Vanish point *
     *************************************************************/

    vector<Mat> vanish_point;
    Mat output_ref_image, gray_ref_image;
    Size size_ref_image;

    size_ref_image = Size(ref_image.cols, ref_image.rows);

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

    vector<Point3f> vanish_point_3d;
    vector<Point2f> vanish_point_2d;
    Point2f center;
    for (int i = 0; i < vanish_point.size(); i++) {
        double x = vanish_point[i].at<float>(0, 0);
        double y = vanish_point[i].at<float>(1, 0);
        if (x < ref_image.cols && y < ref_image.rows)
            vanish_point_2d.push_back(Point2f(vanish_point[i].at<float>(0, 0), vanish_point[i].at<float>(1, 0)));
    }

    if (vanish_point_2d.size() == 3) {
        Point2f A = vanish_point_2d[0];
        Point2f B = vanish_point_2d[1];
        Point2f C = vanish_point_2d[2];

        Point2f tmp;
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

        primka1.getIntersection(primka2, cx, cy);

    } else if (vanish_point_2d.size() == 2) {
        Point2f A = vanish_point_2d[0];
        Point2f B = vanish_point_2d[1];
        cx = (A.x + B.x) / 2;
        cy = (A.y + B.y) / 2;
    } else if (vanish_point_2d.size() == 1) {
        Point2f A = vanish_point_2d[0];
        cx = A.x;
        cy = A.y;

    }

    /*namedWindow(WIN_REF_IMAGE_WITH_VANISH_POINTS);
    imshow(WIN_REF_IMAGE_WITH_VANISH_POINTS, ref_image);*/

    for (int i = 0; i < list_3D_points_after_registration.size(); i++) {
        cout << list_3D_points_after_registration[i] << endl;
    }

    pnp_registration.setOpticalCenter(cx, cy);

    //TODO: Zjistit zda se to obejde bez výpočtu ohniskové vzdálenosti

    /*************************************************************************
     *                     Pozice historické kamery
     ************************************************************************/

    Mat inliers_ref_frame;

    pnp_registration.estimatePoseRANSAC(list_3D_points_after_registration, list_2D_points_after_registration, pnpMethod,
                                        inliers_ref_frame, iterationsCount, reprojectionError, confidence);

    Mat tvect_ref_frame = pnp_registration.get_T_matrix();
    Mat rvect_ref_frame = pnp_registration.get_R_matrix();

    Mat rvect_traspose;
    transpose(rvect_ref_frame, rvect_traspose);
    Mat pos = -rvect_traspose * tvect_ref_frame;

    Mat T(4, 4, rvect_traspose.type());
    T(cv::Range(0, 3), cv::Range(0, 3)) = rvect_traspose;
    T(cv::Range(0, 3), cv::Range(3, 4)) = pos;

    double *p = T.ptr<double>(3);
    p[0] = p[1] = p[2] = 0;
    p[3] = 1;


    /*************************************************************************
     *  4.0 DETECTION
     ************************************************************************/

    initKalmanFilter(kalmanFilter, nStates, nMeasurements, nInputs, dt);
    Mat measurements(nMeasurements, 1, CV_64F);

    measurements.setTo(Scalar(0));

    VideoCapture cap;
    cap.open(video_read_path);

    if (!cap.isOpened()) {
        cout << ERROR_OPEN_CAMERA << endl;
        return -1;
    }

    Mat current_frame, current_frame_vis;
    Mat detection_model = fundamental;

    Ptr<SURF> detector = SURF::create();
    Ptr<FeatureDetector> orb = ORB::create(numKeyPoints);

    Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

    Ptr<DescriptorMatcher> descriptorMatcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    robustMatcher.setDescriptorMatcher(descriptorMatcher);
    robustMatcher.setRatio(ratioTest);

    bool isFirstImage = true;

    vector<Point2f> featuresPrevious;
    vector<Point2f> featuresCurrent;
    vector<Point2f> featuresNextPos;
    vector<uchar> featuresFound;
    Mat cImage, lastImgRef, err;

    pthread_t fast_robust_matcher_t, robust_matcher_t;

    arg_struct fast_robust_matcher_arg_struct, robust_matcher_arg_struct;

    //pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
    //pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) robust_matcher_arg_struct);

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    Mat img_matches;

    /*************************************************************
     *           * Real-time Camera Pose Estimation *
     *************************************************************/
    orb->detect(first_image, key_points_first_image);
    orb->compute(first_image, key_points_first_image, descriptor_first_image);

    robustMatcher.setFeatureDetector(orb);
    robustMatcher.setDescriptorExtractor(orb);

    vector<Point2f> list_2D_points_ref_image_resize_for_video;
    vector<Point2f> list_2D_points_first_image_resize_for_video;


    std::vector<cv::KeyPoint> keypoints1;

    Ptr<SurfFeatureDetector> surf = SurfFeatureDetector::create(numKeyPoints);
    surf->detect(first_image, keypoints1);

    Ptr<SurfDescriptorExtractor> surfDesc = SurfDescriptorExtractor::create();

    cv::Mat descriptors1;
    surfDesc->compute(first_image, keypoints1, descriptors1);

    int testPass = 0;
    int testNotPass = 0;

    namedWindow(WIN_REAL_TIME_DEMO);

    while (cap.read(current_frame) && waitKey(30) != 27) {

        current_frame_vis = current_frame.clone();

        /*if (isFirstImage) {
            while (!getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points_after_registration,
                                       list_2D_points_after_registration, fx, Point2f(cx, cy), start, measurements));
            isFirstImage = false;
        } else {
            //pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
            //pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) &robust_matcher_arg_struct);
        }*/


        bool result = getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points_after_registration,
                                          measurements, T);

        if (result) {
            testPass++;
        } else {
            testNotPass++;
        }

        //TODO: Zjistit jak na vlakna
        //current_frame_vis = getInliersPoints(first_image.clone(), current_frame_vis.clone(), keypoints1, descriptors1);

        /*************************************************************
         *                   * Lucas-Kanade method *
         *************************************************************/
        Mat cImage, current_frame_clone = current_frame.clone();
        cvtColor(current_frame_clone, cImage, CV_BGR2GRAY);
        cImage.convertTo(current_frame_clone, CV_8U);
        featuresPrevious = featuresCurrent;
        goodFeaturesToTrack(current_frame_clone, featuresCurrent, 30, 0.01, 30);

        if (!isFirstImage) {
            calcOpticalFlowPyrLK(lastImgRef, current_frame_clone, featuresPrevious, featuresNextPos, featuresFound,
                                 err);
            for (size_t i = 0; i < featuresNextPos.size(); i++) {
                if (featuresFound[i]) {
                    draw2DPoint(current_frame_vis, featuresPrevious[i], blue);
                    draw2DPoint(current_frame_vis, featuresNextPos[i], yellow);
                    line(current_frame_vis, featuresPrevious[i], featuresNextPos[i], white, 5);
                }
            }
        }

        //TODO: Zjistit návaznost na ostatni algoritmy

        lastImgRef = current_frame_clone.clone();
        isFirstImage = false;
        imshow(WIN_REAL_TIME_DEMO, current_frame_vis);
    }

    cout << "Test prosel: " << testPass << endl;
    cout << "Test neprosel: " << testNotPass << endl;

    waitKey(0);
    return 0;
}

static void onMouseModelRegistration(int event, int x, int y, int, void *) {

    if (event == EVENT_LBUTTONUP) {

        Point2f point_2d = Point2f((float) x, (float) y);
        bool is_registrable = registration.is_registrable();

        if (is_registrable) {
            registration.register2DPoint(point_2d);
            index++;
            if (registration.getNumRegistration() == registration.getNumMax()) {
                end_registration = true;
            }
        }
    } else if (event == EVENT_RBUTTONUP) {
        index++;
    }
}


void *fast_robust_matcher(void *arg) {

    pthread_exit(NULL);
}


void *robust_matcher(void *arg) {
    struct arg_struct *aStruct = (struct arg_struct *) arg;

    pthread_exit(NULL);
}

vector<Mat> processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg) {
    cv::Mat imgCanny;
    cv::Canny(imgGRAY, imgCanny, 180, 120, 3);
    vector<vector<cv::Point>> lineSegments;
    vector<cv::Point> aux;
#ifndef USE_PPHT
    vector <Vec2f> lines;
    cv::HoughLines(imgCanny, lines, 1, CV_PI / 180, 200);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];

        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        Point pt1, pt2;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

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
    std::vector<std::vector<int>> CS;    // index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

    // Call msac function for multiple vanishing point estimation
    msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps);
    for (int v = 0; v < vps.size(); v++) {
        printf("VP %d (%.3f, %.3f, %.3f)", v, vps[v].at<float>(0, 0), vps[v].at<float>(1, 0),
               vps[v].at<float>(2, 0));
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

