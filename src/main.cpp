//
// Created by acervenka2 on 17.01.2017.
//




#include "main.h"

int index = 0;
vector<Point2f> inliners;

bool getRobustEstimation(Mat current_frame_vis, Mat description_first_image,
                         vector<Point3f> list_3D_points_after_registration,
                         vector<Point2f> list_2D_points_after_registration, int focal, Point2f center,
                         Mat measurements) {

    vector<DMatch> good_matches;
    vector<KeyPoint> key_points_current_frame;
    vector<Point3f> list_points3d_model_match;
    vector<Point2f> list_points2d_scene_match;
    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;

    rmatcher.fastRobustMatch(current_frame_vis, good_matches, key_points_current_frame, description_first_image);

    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        Point3f point3d_model = list_3D_points_after_registration[good_matches[match_index].trainIdx];
        Point2f point2d_scene = key_points_current_frame[good_matches[match_index].queryIdx].pt;
        list_points3d_model_match.push_back(point3d_model);
        list_points2d_scene_match.push_back(point2d_scene);
    }

    if (good_matches.size() > 0) {

        pnp_registration.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnpMethod,
                                            inliers_idx, iterationsCount, reprojectionError, confidence);
        for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
            int n = inliers_idx.at<int>(inliers_index);
            Point2f point2d = list_points2d_scene_match[n];
            list_points2d_inliers.push_back(point2d);
        }

        draw2DPoints(current_frame_vis, list_points2d_inliers, blue);

        if (inliers_idx.rows >= minInliersKalman) {

            Mat translation_measured(3, 1, CV_64F);
            translation_measured = pnp_registration.get_t_matrix();

            Mat rotation_measured(3, 3, CV_64F);
            rotation_measured = pnp_registration.get_R_matrix();


            fillMeasurements(measurements, translation_measured, rotation_measured);
        }
    }

    //getDirection(list_points2d_scene_match, registration_2D_points, focal, center);

    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
    pnp_registration.set_P_matrix(rotation_estimated, translation_estimated);

    /*draw2DPoints(current_frame_vis, list_points2d_scene_match, red);
    draw2DPoints(current_frame_vis, list_points2d_inliers, blue);
    for (int i = 0; i < list_points2d_inliers.size(); i++) {
        line(current_frame_vis, list_points2d_inliers[i], list_points2d_scene_match[i], green);
    }*/

    inliners = list_points2d_inliers;

    return true;
}

bool getLightweightEstimation(Mat current_frame_vis, Mat description_first_image,
                              vector<Point3f> list_3D_points_after_registration,
                              vector<Point2f> list_2D_points_after_registration, int focal, Point2f center,
                              time_t start,
                              Mat measurements) {




    // Mat tresult = T1n − R1n ∗ rvect   10    ∗ T10;

    Mat inliers_idx;
    vector<Point2f> list_points2d_inliers;


    boolean good_measurement = false;

    // GOOD MEASUREMENT
    if (inliers_idx.rows >= minInliersKalman) {

        // Get the measured translation
        Mat translation_measured(3, 1, CV_64F);
        translation_measured = pnp_registration.get_t_matrix();

        // Get the measured rotation
        Mat rotation_measured(3, 3, CV_64F);
        rotation_measured = pnp_registration.get_R_matrix();

        // fill the measurements vector
        fillMeasurements(measurements, translation_measured, rotation_measured);
        good_measurement = true;

    }


    Mat translation_estimated(3, 1, CV_64F);
    Mat rotation_estimated(3, 3, CV_64F);

    // update the Kalman filter with good measurements
    updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
    pnp_registration.set_P_matrix(rotation_estimated, translation_estimated);

    return true;
}


int getDirection(vector<Point2f> list_points2d_scene_match, vector<Point2f> registration_2D_points, int focal,
                 Point2f center) {

    Mat rvect;
    Mat tvect;

    Mat essential = findEssentialMat(list_points2d_scene_match, registration_2D_points, focal, center);
    // correctMatches(essential, list_points2d_scene_match, registration_2D_points,list_points2d_scene_match, registration_2D_points);
    recoverPose(essential, list_points2d_scene_match, registration_2D_points, rvect, tvect, 1, center);

    double xAngle = atan2f(rvect.at<float>(2, 1), rvect.at<float>(2, 2));
    double yAngle = atan2f(-rvect.at<float>(2, 0),
                           sqrtf(rvect.at<float>(2, 1) * rvect.at<float>(2, 1) +
                                 rvect.at<float>(2, 2) * rvect.at<float>(2, 2)));
    double zAngle = atan2f(rvect.at<float>(1, 0), rvect.at<float>(0, 0));

    xAngle = (int) convert_radian_to_degree(xAngle);
    yAngle = (int) convert_radian_to_degree(yAngle);
    zAngle = (int) convert_radian_to_degree(zAngle);

    cout << "xAngle: " << xAngle << "%" << endl;
    cout << "yAngle: " << yAngle << "%" << endl;
    cout << "zAngle: " << zAngle << "%" << endl;
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
    imshow("Right Image (RANSAC)", first_image);
    namedWindow("Left Image (RANSAC)");
    imshow("Left Image (RANSAC)", second_image);*/

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


    draw2DPoints(first_image.clone(), list_2D_points_after_triangulation, blue);
    Mat frame_with_triangulation = first_image.clone();

    /*************************************************************
     *                    * Registration *                       *
     *************************************************************/

    Mat ref_image = imread(path_to_ref_image);

    namedWindow(WIN_REF_IMAGE_FOR_USER);
    namedWindow(WIN_USER_SELECT_POINT);
    setMouseCallback(WIN_USER_SELECT_POINT, onMouseModelRegistration, 0);

    Mat clone_of_ref_image = ref_image.clone();

    if (!ref_image.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    vector<Point2f> list;
    int i = 0;
    while (i < list_3D_points_after_triangulation.size()) {
        list.push_back(list_2D_points_after_triangulation[i]);
        i++;
    }

    registration.setNumMax(number_registration);
    vector<Point2f> list_points2d;
    vector<Point3f> list_points3d;

    int previslyNumRegistration = registration.getNumRegistration();
    while (waitKey(30) < 0) {

        clone_of_ref_image = ref_image.clone();
        list_points2d = registration.get_points2d();
        list_points3d = registration.get_points3d();
        if (!end_registration) {
            drawCounter(clone_of_ref_image, registration.getNumRegistration(), registration.getNumMax(), red);
            draw2DPoint(frame_with_triangulation, list[index], green);
            Point3f point3f = list_3D_points_after_triangulation[index];
            drawQuestion(clone_of_ref_image, point3f, red);
            if (previslyNumRegistration != registration.getNumRegistration()) {
                registration.register3DPoint(point3f);
                previslyNumRegistration = registration.getNumRegistration();
            }

        } else {
            draw2DPoints(clone_of_ref_image, list_points2d, blue);
            break;
        }

        draw2DPoints(clone_of_ref_image, list_points2d, blue);
        imshow(WIN_USER_SELECT_POINT, clone_of_ref_image);
        imshow(WIN_REF_IMAGE_FOR_USER, frame_with_triangulation);
    }

    destroyWindow(WIN_USER_SELECT_POINT);
    destroyWindow(WIN_REF_IMAGE_FOR_USER);

    /*************************************************************
     *                   * Point Estimation *
     *************************************************************/

    double camera_parameters[4];
    camera_parameters[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    camera_parameters[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    camera_parameters[2] = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
    camera_parameters[3] = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

    vector<Point3f> list_3D_points_after_registration = registration.get_points3d();
    vector<Point2f> list_2D_points_after_registration = registration.get_points2d();

    pnp_registration.setMatrixParam(camera_parameters);
    //pnp_registration.estimatePose(list_3D_points_after_registration, list_2D_points_after_registration, pnpMethod);

    double fx = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
    double fy = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

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
    //imshow(WIN_REF_IMAGE_WITH_HOUGH_LINES, output_ref_image);

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

    double cx, cy;
    primka1.getIntersection(primka2, cx, cy);

    //namedWindow(WIN_REF_IMAGE_WITH_VANISH_POINTS);
    //imshow(WIN_REF_IMAGE_WITH_VANISH_POINTS, ref_image);

    for (int i = 0; i < list_3D_points_after_registration.size(); i++) {
        cout << list_3D_points_after_registration[i] << endl;
    }

    pnp_registration.setOpticalCenter(cx, cy);

    cout << "======= Vysledek prvni casti ========" << endl;
    cout << "[ " << cx << " 0 " << fx << " ]" << endl;
    cout << "[ 0 " << cy << " " << fy << " ]" << endl;
    cout << "[ 0 0 1 ]" << endl;

    double width_image = first_image.cols;
    double height_image = first_image.rows;

    double sx = 1.5 * width_image;
    double sy = 1.5 * height_image;

    double focal = (fx * sx) / width_image;
    cout << "Focal lenght: " << focal << endl;
    focal = (fy * sy) / height_image;
    cout << "Focal lenght: " << focal << endl;

    Mat test = first_image.clone();

    vector<Point2f> list_2D_points_first_image;
    vector<Point2f> list_2D_points_ref_image;
    for (int i = 0; i < 8; i++) {
        list_2D_points_first_image.push_back(list[i]);
    }

    for (int i = 0; i < 8; i++) {
        double resize_x = (list_2D_points_after_registration[i].x * first_image.cols) / ref_image.cols;
        double resize_y = (list_2D_points_after_registration[i].y * first_image.rows) / ref_image.rows;
        list_2D_points_ref_image.push_back(Point2f(resize_x, resize_y));
    }

    draw2DPoints(test, list_2D_points_ref_image, red);
    draw2DPoints(test, list_2D_points_first_image, blue);
    for (int i = 0; i < registration.getNumRegistration(); i++) {
        line(test, list_2D_points_ref_image[i], list_2D_points_first_image[i], green);
    }
    namedWindow("TEST");
    imshow("TEST", test);

    /*************************************************************************
     *  4.0 DETECTION
     ************************************************************************/

    initKalmanFilter(kalmanFilter, nStates, nMeasurements, nInputs, dt);
    Mat measurements(nMeasurements, 1, CV_64F);

    measurements.setTo(Scalar(0));

    VideoCapture cap;
    cap.open(video_read_path);

    if (!cap.isOpened()) {
        cout << "Could not open the camera device" << endl;
        return -1;
    }

    Mat current_frame, current_frame_vis;
    Mat detection_model = fundamental;

    Ptr<SURF> detector = SURF::create();
    Ptr<FeatureDetector> orb = ORB::create(numKeyPoints);

    Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

    Ptr<DescriptorMatcher> descriptorMatcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    rmatcher.setDescriptorMatcher(descriptorMatcher);
    rmatcher.setRatio(ratioTest);

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

    rmatcher.setFeatureDetector(orb);
    rmatcher.setDescriptorExtractor(orb);

    vector<Point2f> list_2D_points_ref_image_resize_for_video;
    vector<Point2f> list_2D_points_first_image_resize_for_video;

    //namedWindow("Right Image Homography (RANSAC)");
    //namedWindow("Left Image Homography (RANSAC)");

    std::vector<cv::KeyPoint> keypoints1;
    Ptr<SurfFeatureDetector> surf = SurfFeatureDetector::create(numKeyPoints);
    surf->detect(first_image, keypoints1);
    Ptr<SurfDescriptorExtractor> surfDesc = SurfDescriptorExtractor::create();
    cv::Mat descriptors1;
    surfDesc->compute(first_image, keypoints1, descriptors1);


    while (cap.read(current_frame) && waitKey(30) != 27) {

        current_frame_vis = current_frame.clone();


        for (int i = 0; i < 8; i++) {
            double resize_x_first_image = (list_2D_points_first_image[i].x * current_frame_vis.cols) / first_image.cols;
            double resize_y_first_image = (list_2D_points_first_image[i].y * current_frame_vis.rows) / first_image.rows;
            double resize_x_ref_image = (list_2D_points_ref_image[i].x * current_frame_vis.cols) / first_image.cols;
            double resize_y_ref_image = (list_2D_points_ref_image[i].y * current_frame_vis.rows) / first_image.rows;
            list_2D_points_ref_image_resize_for_video.push_back(Point2f(resize_x_first_image, resize_y_first_image));
            list_2D_points_first_image_resize_for_video.push_back(Point2f(resize_x_ref_image, resize_y_ref_image));
        }

        /*if (isFirstImage) {
            while (!getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points_after_registration,
                                       list_2D_points_after_registration, fx, Point2f(cx, cy), start, measurements));
            isFirstImage = false;
        } else {
            //pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
            //pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) &robust_matcher_arg_struct);
        }*/

        //getInliersPoints(first_image.clone(), current_frame, keypoints1, descriptors1);



        getRobustEstimation(current_frame_vis, descriptor_first_image, list_3D_points_after_registration,
                            list_2D_points_after_registration, focal, Point2f(cx, cy), measurements);


        draw2DPoints(current_frame_vis, list_2D_points_first_image_resize_for_video, red);
        draw2DPoints(current_frame_vis, inliners, yellow);
        for (int i = 0; i < inliners.size(); i++) {
            line(current_frame_vis, list_2D_points_first_image_resize_for_video[i], inliners[i], green);
        }

        /*************************************************************
         *                   * Lucas-Kanade method *
         *************************************************************/
        Mat cImage, current_frame_clone = current_frame.clone();
        cvtColor(current_frame_clone, cImage, CV_BGR2GRAY);
        cImage.convertTo(current_frame_clone, CV_8U);
        featuresPrevious = std::move(featuresCurrent);
        goodFeaturesToTrack(current_frame_clone, featuresCurrent, 30, 0.01, 30);

        if (!isFirstImage) {
            calcOpticalFlowPyrLK(lastImgRef, current_frame_clone, featuresPrevious, featuresNextPos, featuresFound,
                                 err);
            for (size_t i = 0; i < featuresNextPos.size(); i++) {
                if (featuresFound[i]) {
                    //line(current_frame_vis, featuresPrevious[i], featuresNextPos[i], green, 5);
                }
            }
        }

        lastImgRef = current_frame_clone.clone();
        isFirstImage = false;
        imshow(WIN_REAL_TIME_DEMO, current_frame_vis);
    }

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

    struct arg_struct *aStruct = (struct arg_struct *) arg;
    while (!getLightweightEstimation(aStruct->current_frame, aStruct->description_first_image,
                                     aStruct->list_3D_points_after_registration,
                                     aStruct->list_2D_points_after_registration,
                                     aStruct->focal, aStruct->center, aStruct->start, aStruct->measurements)) {};
    pthread_exit(NULL);
}


void *robust_matcher(void *arg) {
    struct arg_struct *aStruct = (struct arg_struct *) arg;
    /*while (!getRobustEstimation(aStruct->current_frame, aStruct->description_first_image,
                                aStruct->list_3D_points_after_registration, aStruct->list_2D_points_after_registration,
                                aStruct->focal, aStruct->center, aStruct->measurements)) {};*/

    pthread_exit(NULL);
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


void getInliersPoints(Mat first_image, Mat second_image, vector<cv::KeyPoint> keypoints1, Mat descriptors1) {

    cv::Mat image1 = first_image.clone();
    cv::Mat image2 = second_image.clone();

    std::vector<cv::KeyPoint> keypoints2;

    Ptr<SurfFeatureDetector> surf = SurfFeatureDetector::create(numKeyPoints);
    surf->detect(image2, keypoints2);

    Ptr<SurfDescriptorExtractor> surfDesc = SurfDescriptorExtractor::create();

    cv::Mat descriptors2;
    surfDesc->compute(image2, keypoints2, descriptors2);

    BFMatcher matcher;

    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    std::vector<cv::DMatch> selMatches;

    std::vector<int> pointIndexes1;
    std::vector<int> pointIndexes2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        pointIndexes1.push_back(it->queryIdx);
        pointIndexes2.push_back(it->trainIdx);
    }

    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
    cv::KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

    cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(selPoints1), cv::Mat(selPoints2), CV_FM_7POINT);

    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1, fundemental, lines1);

    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fundemental, lines2);


    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it) {

        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));

        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }

    std::vector<uchar> inliers(points1.size(), 0);
    fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, 1, 0.98);

    image1 = first_image.clone();
    image2 = second_image.clone();

    std::vector<cv::Point2f> points1In, points2In;
    std::vector<cv::Point2f>::const_iterator itPts = points1.begin();
    std::vector<uchar>::const_iterator itIn = inliers.begin();
    while (itPts != points1.end()) {
        if (*itIn)
            points1In.push_back(*itPts);
        ++itPts;
        ++itIn;
    }

    itPts = points2.begin();
    itIn = inliers.begin();
    while (itPts != points2.end()) {
        if (*itIn)
            points2In.push_back(*itPts);
        ++itPts;
        ++itIn;
    }

    cv::findHomography(cv::Mat(points1In), cv::Mat(points2In), inliers, CV_RANSAC, 1.);

    image1 = first_image.clone();
    image2 = second_image.clone();

    itPts = points1In.begin();
    itIn = inliers.begin();
    while (itPts != points1In.end()) {
        if (*itIn)
            cv::circle(first_image, *itPts, 3, white, 2);
        ++itPts;
        ++itIn;
    }

    itPts = points2In.begin();
    itIn = inliers.begin();
    while (itPts != points2In.end()) {
        if (*itIn)
            cv::circle(second_image, *itPts, 3, white, 2);
        ++itPts;
        ++itIn;
    }

    cv::imshow("Homography", first_image);
    cv::imshow("Homography", second_image);
}


