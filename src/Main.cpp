/**
 * Main.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include "Main.h"

int index = 0;

int main(int argc, char *argv[]) {

    Mat first_image = loadImage(path_to_first_image);
    Mat second_image = loadImage(path_to_second_image);

    /**
     * Kalibrace fotoaparatu
     * Dle tutorialu dostupneho na https://www.packtpub.com/books/content/learn-computer-vision-applications-open-cv
     */

    vector<string> list_file_name;
    Size border_size(7, 9);

    for (int i = 1; i <= 12; i++) {
        stringstream file_name;
        file_name << "resource/image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        list_file_name.push_back(file_name.str());
    }

    cameraCalibrator.addChessboardPoints(list_file_name, border_size);
    cameraCalibrator.calibrate(first_image.size());

    Mat camera_matrix = cameraCalibrator.getCameraMatrix();

    pnp_registration.setCameraMatrix(camera_matrix);


    /**
     * Robust matcher
     * Dle tutorialu dostupneho na https://www.packtpub.com/books/content/learn-computer-vision-applications-open-cv
     * Dle tutorialu dostupneho na http://docs.opencv.org/3.1.0/dc/d2c/tutorial_real_time_pose.html
     */

    Ptr<FeatureDetector> featureDetector = SURF::create(numKeyPoints);
    vector<KeyPoint> key_points_first_image, key_points_second_image;
    vector<Point2f> detection_points_first_image, detection_points_second_image;
    vector<DMatch> matches;

    robustMatcher.setConfidenceLevel(confidenceLevel);
    robustMatcher.setMinDistanceToEpipolar(min_dist);
    robustMatcher.setRatio(ratioTest);
    robustMatcher.setFeatureDetector(featureDetector);

    Mat fundamental_matrix = robustMatcher.robustMatchRANSAC(first_image, second_image, matches, key_points_first_image,
                                                             key_points_second_image);

    /**
     * Rozklad matic
     */

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = key_points_first_image[it->queryIdx].pt.x;
        float y = key_points_first_image[it->queryIdx].pt.y;
        detection_points_first_image.push_back(Point2f(x, y));

        x = key_points_second_image[it->trainIdx].pt.x;
        y = key_points_second_image[it->trainIdx].pt.y;
        detection_points_second_image.push_back(Point2f(x, y));
    }

    resize(fundamental_matrix, fundamental_matrix, Size(3, 3));
    cv::Mat R1, R2, t, essential_matrix;

    essential_matrix = pnp_registration.getCameraMatrix() * fundamental_matrix * pnp_registration.getCameraMatrix();
    decomposeEssentialMat(essential_matrix, R1, R2, t);

    /**
     * Triangulace
     */

    Mat rotation_translation_vector_first_image, rotation_translation_vector_second_image, result_3D_points,
            rotation_vector_first_image, translation_vector_first_image, found_3D_points;
    rotation_translation_vector_second_image = Mat::eye(3, 4, CV_64FC1);

    double count_positive_z[4] = {0, 0, 0, 0};

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

        Mat triangulation_3D_points, camera_matrix_a, camera_matrix_b;

        camera_matrix_a = camera_matrix * rotation_translation_vector_first_image;
        camera_matrix_b = camera_matrix * rotation_translation_vector_second_image;
        triangulatePoints(camera_matrix_a, camera_matrix_b, detection_points_first_image, detection_points_second_image,
                          found_3D_points);

        transpose(found_3D_points, triangulation_3D_points);
        convertPointsFromHomogeneous(triangulation_3D_points, triangulation_3D_points);

        for (int j = 0; j < triangulation_3D_points.rows; j++) {
            if (triangulation_3D_points.at<float>(j, 2) > 0) {
                count_positive_z[i]++;
            }
        }

        if (result_3D_points.empty() or count_positive_z[i - 1] < count_positive_z[i]) {
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

    cout << list_3D_points_after_triangulation << endl;

    /**
     * Registrace korespondencnich bodu
     * Dle tutorialu dostupneho na http://docs.opencv.org/3.1.0/dc/d2c/tutorial_real_time_pose.html
     */

    namedWindow(WIN_REF_IMAGE_FOR_USER);
    namedWindow(WIN_USER_SELECT_POINT);

    setMouseCallback(WIN_USER_SELECT_POINT, onMouseModelRegistration, 0);

    Mat ref_image = loadImage(path_to_ref_image);

    registration.setRegistrationMax(number_registration);
    vector<Point2f> list_points2d;
    vector<Point3f> list_points3d;

    int previousNumRegistration = registration.getRegistrationCount();
    vector<int> index_of_points;

    while (waitKey(30) < 0) {

        Mat clone_of_ref_image = ref_image.clone();
        Mat clone_frame_with_triangulation = first_image.clone();
        list_points2d = registration.getList2DPoints();
        list_points3d = registration.getList3DPoints();
        if (!end_registration) {

            draw2DPoint(clone_frame_with_triangulation, detection_points_first_image[index], green);
            Point3f point3f = list_3D_points_after_triangulation[index];

            if (previousNumRegistration != registration.getRegistrationCount()) {
                index_of_points.push_back(index);
                registration.register3DPoint(point3f);
                previousNumRegistration = registration.getRegistrationCount();
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
        imshow(WIN_REF_IMAGE_FOR_USER, clone_frame_with_triangulation);
    }

    destroyWindow(WIN_USER_SELECT_POINT);
    destroyWindow(WIN_REF_IMAGE_FOR_USER);

    vector<Point3f> list_3D_points = registration.getList3DPoints();
    vector<Point2f> list_2D_points = registration.getList2DPoints();

    /**
     * Vypocet optickeho stredu
     */

    vector<Mat> vanish_point;
    Mat output_ref_image, gray_ref_image;
    Size size_ref_image;
    double cx, cy;

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

    vector<Point2f> list_vanish_points;
    Point2f center;
    for (int i = 0; i < vanish_point.size(); i++) {
        double x = vanish_point[i].at<float>(0, 0);
        double y = vanish_point[i].at<float>(1, 0);
        if (x < ref_image.cols && y < ref_image.rows)
            list_vanish_points.push_back(Point2f(vanish_point[i].at<float>(0, 0), vanish_point[i].at<float>(1, 0)));
    }


    if (list_vanish_points.size() == 3) {
        Point2f A = list_vanish_points[0];
        Point2f B = list_vanish_points[1];
        Point2f C = list_vanish_points[2];

        Point2f tmp;
        center.x = (A.x + B.x) / 2;
        center.y = (A.y + B.y) / 2;
        tmp.x = C.x;
        tmp.y = C.y;
        line(ref_image, center, tmp, green, 1);
        Line primka1(center.x, center.y, tmp.x, tmp.y);

        center.x = (A.x + C.x) / 2;
        center.y = (A.y + C.y) / 2;
        tmp.x = B.x;
        tmp.y = B.y;
        line(ref_image, center, tmp, green, 1);
        Line primka2(center.x, center.y, tmp.x, tmp.y);

        primka1.getIntersection(primka2, cx, cy);
    } else {
        cx = ref_image.cols / 2 - 0.5;
        cy = ref_image.rows / 2 - 0.5;
    }

    pnp_registration.setOpticalCenter(cx, cy);

    /**
     * Pozice historicke kamery
     */
    Mat inliers_ref_frame;
    vector<Point2f> list_points2d_scene_match;

    pnp_registration.estimatePoseRANSAC(list_3D_points, list_2D_points,
                                        pnp_method, inliers_ref_frame, useExtrinsicGuess, iterationsCount,
                                        reprojectionError, confidence);

    for (int inliers_index = 0; inliers_index < inliers_ref_frame.rows; ++inliers_index) {
        int n = inliers_ref_frame.at<int>(inliers_index);
        Point2f point2d = list_2D_points[n];
        list_points2d_scene_match.push_back(point2d);
    }

    Mat tvect_ref_frame = pnp_registration.getTranslationMatrix();
    Mat rvect_ref_frame = pnp_registration.getRotationMatrix();

    Mat rvect_traspose;
    transpose(rvect_ref_frame, rvect_traspose);
    Mat pos = -rvect_traspose * tvect_ref_frame;

    Mat T(4, 4, rvect_traspose.type());
    T(cv::Range(0, 3), cv::Range(0, 3)) = rvect_traspose;
    T(cv::Range(0, 3), cv::Range(3, 4)) = pos;

    double *p = T.ptr<double>(3);
    p[0] = p[1] = p[2] = 0;
    p[3] = 1;

    cout << "Referencni kamera: " <<  T << endl;

    /**
     * Robust matcher
     */

    Mat current_frame, current_frame_vis, descriptor_first_image;
    key_points_first_image.clear();

    Ptr<FeatureDetector> orb = ORB::create(numKeyPoints);
    orb->detect(first_image, key_points_first_image);
    orb->compute(first_image, key_points_first_image, descriptor_first_image);

    Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

    Ptr<DescriptorMatcher> descriptorMatcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    robustMatcher.setDescriptorMatcher(descriptorMatcher);
    robustMatcher.setRatio(ratioTest);
    robustMatcher.setFeatureDetector(orb);
    robustMatcher.setDescriptorExtractor(orb);


    /*pthread_t fast_robust_matcher_t, robust_matcher_t;
    matcher_struct fast_robust_matcher_arg_struct, robust_matcher_arg_struct;
    pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
    pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) &robust_matcher_arg_struct);*/

    /**
     * Real-time Camera Pose Estimation
     */

    initKalmanFilter(kalmanFilter, nStates, nMeasurements, nInputs, dt);
    Mat measurements(nMeasurements, 1, CV_64F);
    measurements.setTo(Scalar(0));

    VideoCapture cap;
    cap.open(video_read_path);
    bool isFirstImage = true;

    if (!cap.isOpened()) {
        cout << ERROR_OPEN_CAMERA << endl;
        exit(1);
    }

    namedWindow(WIN_REAL_TIME_DEMO);

    while (cap.read(current_frame) && waitKey(30) != 27) {

        current_frame_vis = current_frame.clone();

        if (isFirstImage) {
            //getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points, measurements,                                T);
            cameraCalibrator.calibrate(current_frame_vis.size());
            pnp_detection.setCameraMatrix(cameraCalibrator.getCameraMatrix());
            isFirstImage = false;
        } /*else {
            robust_matcher_arg_struct.current_frame = current_frame;
            robust_matcher_arg_struct.description_first_image = descriptor_first_image;
            robust_matcher_arg_struct.list_3D_points = list_3D_points;
            robust_matcher_arg_struct.measurements = measurements;
            robust_matcher_arg_struct.T = T;

            pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
            pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) &robust_matcher_arg_struct);
            waitKey(20);
        }*/

        getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points, measurements, T);
        imshow(WIN_REAL_TIME_DEMO, current_frame_vis);
    }

    waitKey(0);
    return 0;
}

Mat loadImage(const string path_to_ref_image) {

    Mat image = imread(path_to_ref_image);

    if (!image.data) {
        cout << ERROR_READ_IMAGE << endl;
        exit(1);
    }
    return image;
}

bool getRobustEstimation(Mat current_frame_vis, Mat description_first_image,
                         vector<Point3f> list_3D_points_after_registration, Mat measurements, Mat refT) {

    vector<DMatch> good_matches;
    vector<KeyPoint> key_points_current_frame;
    vector<Point3f> list_points3d_model_match;
    vector<Point2f> list_points2d_scene_match;
    vector<Point2f> list_points2d_inliers;
    Mat inliers_idx;



    robustMatcher.robustMatch(current_frame_vis, good_matches, key_points_current_frame, description_first_image);

    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        Point3f point3d_model = list_3D_points_after_registration[good_matches[match_index].trainIdx];
        Point2f point2d_scene = key_points_current_frame[good_matches[match_index].queryIdx].pt;
        list_points3d_model_match.push_back(point3d_model);
        list_points2d_scene_match.push_back(point2d_scene);
    }

    draw2DPoints(current_frame_vis, list_points2d_scene_match, red);

    bool good_measurement = false;

    if (good_matches.size() > 0) {

        pnp_detection.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnp_method,
                                            inliers_idx, useExtrinsicGuess, iterationsCount, reprojectionError,
                                            confidence);

        for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
            int n = inliers_idx.at<int>(inliers_index);
            Point2f point2d = list_points2d_scene_match[n];
            list_points2d_inliers.push_back(point2d);
        }

        Mat tvect_ref_frame = pnp_detection.getTranslationMatrix();
        Mat rvect_ref_frame = pnp_detection.getRotationMatrix();

        Mat rvect_traspose;

        transpose(rvect_ref_frame, rvect_traspose);
        Mat pos = -rvect_traspose * tvect_ref_frame;

        Mat T(4, 4, rvect_traspose.type());
        T(cv::Range(0, 3), cv::Range(0, 3)) = rvect_traspose;
        T(cv::Range(0, 3), cv::Range(3, 4)) = pos;

        double *p = T.ptr<double>(3);
        p[0] = p[1] = p[2] = 0;
        p[3] = 1;

        Mat revT = Mat(4, 4, CV_64F);
        revT = T.inv() * refT;

        try {
            revT = Mat(4, 4, CV_64F);
            revT = pnp_detection.getProjectionMatrix().inv() * refT;

            double posun_x_t = T.at<float>(0, 3);
            double posun_y_t = T.at<float>(1, 3);

            double posun_x_refT = refT.at<float>(0, 3);
            double posun_y_refT = refT.at<float>(1, 3);

            double posun_x = revT.at<float>(0, 3);
            double posun_y = revT.at<float>(1, 3);

            cout << "X: " << posun_x_t << " " << posun_x_refT << " " << posun_x << endl;
            cout << "Y: " << posun_y_t << " " << posun_y_refT << " " << posun_y << endl;

            if (posun_x < -1.5) {
                cout << "Doprava " << posun_x << endl;
                drawText(current_frame_vis, "Doprava", 10, 10, yellow);
            } else if (posun_x > 1.5) {
                cout << "Doleva " << posun_x << endl;
                drawText(current_frame_vis, "Doleva", 10, 10, yellow);
            } else {
                drawText(current_frame_vis, "Je to OK", 10, 10, yellow);
                cout << "OK " << posun_x << endl;
            }

            if (posun_y > -1.5) {
                cout << "Nahoru " << posun_y << endl;
                drawText(current_frame_vis, "Nahoru", 10, 50, yellow);
            } else if (posun_y < 1.5) {
                cout << "Dolu " << posun_y << endl;
                drawText(current_frame_vis, "Dolu", 10, 50, yellow);
            } else {
                drawText(current_frame_vis, "Je to OK", 10, 50, yellow);
                cout << "OK " << posun_y << endl;
            }


        } catch (Exception e) {
            cout << ERROR_COMPARE_MATRIX << endl;
        }

        if (inliers_idx.rows >= minInliersKalman) {

            Mat translation_measured(3, 1, CV_64F);
            translation_measured = pnp_detection.getTranslationMatrix();

            Mat rotation_measured(3, 3, CV_64F);
            rotation_measured = pnp_detection.getRotationMatrix();

            good_measurement = true;
            fillMeasurements(measurements, translation_measured, rotation_measured);
        }
    }

    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
    pnp_detection.setProjectionMatrix(rotation_estimated, translation_estimated);


    cout << "============================" << endl;


    return good_measurement;
}

bool getLightweightEstimation(Mat current_frame_vis, Mat description_first_image,
                              vector<Point3f> list_3D_points_after_registration, Mat measurements, Mat refT) {

    //TODO: Zjistit co vsechno zde má být

    vector<Point2f> featuresPrevious;
    vector<Point2f> featuresCurrent;
    vector<Point2f> featuresNextPos;

    vector<uchar> featuresFound;
    Mat cImage, lastImgRef, err;

    cvtColor(current_frame_vis, cImage, CV_BGR2GRAY);
    cImage.convertTo(current_frame_vis, CV_8U);
    featuresPrevious = featuresCurrent;
    goodFeaturesToTrack(current_frame_vis, featuresCurrent, 30, 0.01, 30);
/*
    if (!isFirstImage) {
        calcOpticalFlowPyrLK(lastImgRef, current_frame_vis, featuresPrevious, featuresNextPos, featuresFound, err);
        for (size_t i = 0; i < featuresNextPos.size(); i++) {
            if (featuresFound[i]) {
                draw2DPoint(current_frame_vis, featuresPrevious[i], blue);
                draw2DPoint(current_frame_vis, featuresNextPos[i], yellow);
                line(current_frame_vis, featuresPrevious[i], featuresNextPos[i], white, 5);
            }
        }
    }

*/
    lastImgRef = current_frame_vis.clone();


    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;


    bool good_measurement = false;


    if (inliers_idx.rows >= minInliersKalman) {

        Mat translation_measured(3, 1, CV_64F);
        translation_measured = pnp_registration.getTranslationMatrix();

        Mat rotation_measured(3, 3, CV_64F);
        rotation_measured = pnp_registration.getRotationMatrix();

        fillMeasurements(measurements, translation_measured, rotation_measured);
        good_measurement = true;

    }

    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
    pnp_registration.setProjectionMatrix(rotation_estimated, translation_estimated);

    return good_measurement;
}


static void onMouseModelRegistration(int event, int x, int y, int, void *) {

    if (event == EVENT_LBUTTONUP) {

        Point2f point_2d = Point2f((float) x, (float) y);
        bool is_registrable = registration.isRegistration();

        if (is_registrable) {
            registration.register2DPoint(point_2d);
            index++;
            if (registration.getRegistrationCount() == registration.getRegistrationMax()) {
                end_registration = true;
            }
        }
    } else if (event == EVENT_RBUTTONUP) {
        index++;
    }
}


void *fast_robust_matcher(void *arg) {

return NULL;}


void *robust_matcher(void *arg) {
    struct matcher_struct *param = (struct matcher_struct *) arg;

    Mat current_frame = param->current_frame;
    Mat description_first_image = param->description_first_image;
    vector<Point3f> list_3D_points_after_registration = param->list_3D_points_after_registration;
    Mat measurements = param->measurements;
    Mat T = param->T;

    getRobustEstimation(current_frame, description_first_image, list_3D_points_after_registration, measurements, T);

    return NULL;
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

    std::vector<cv::Mat> vps;
    std::vector<std::vector<int>> CS;
    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

    msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps);
    for (int v = 0; v < vps.size(); v++) {
        double vpNorm = cv::norm(vps[v]);
        if (fabs(vpNorm - 1) < 0.001) {
        }
    }

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

