//
// Created by acervenka2 on 17.01.2017.
//

#include "main.h"

int index = 0;

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
    while (i < list_3D_points_after_triangulation.size()) {
        //registration.register3DPoint(list_3D_points_after_triangulation[i]);
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

    /*for (int i = 0; i < descriptors_of_ref_image.rows; i++) {
        model.add_descriptor(descriptors_of_ref_image.row(i));
    }*/

    vector<Point3f> list_3D_points_after_registration = registration.get_points3d();
    vector<Point2f> registration_2DPoint = registration.get_points2d();

    /*for (int i = 0; i < list_3D_points_after_registration.size(); i++) {
        model.add_correspondence(registration_2DPoint[i], list_3D_points_after_registration[i]);
    }*/

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

    /*
     * [ fx   0  cx ]
     * [  0  fy  cy ]
     * [  0   0   1 ]
     */
    /*Mat camera_matrix_ref = cv::Mat::zeros(3, 3, CV_64FC1);
    camera_matrix_ref.at<double>(0, 0) = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    camera_matrix_ref.at<double>(1, 1) = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    camera_matrix_ref.at<double>(0, 2) = cx;
    camera_matrix_ref.at<double>(1, 2) = cy;
    camera_matrix_ref.at<double>(2, 2) = 1;

    model.set_camera_matrix(camera_matrix_ref);
    model.save("result.yml");*/

    /*************************************************************
     * Calcule Dinstace
     *************************************************************/

    double fx = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    double fy = cameraCalibrator.getCameraMatrix().at<double>(1, 1);

    double avg_f = (fx + fy) / 2;
    double focal_length = 40;
    double m = avg_f / focal_length;

    /*Mat gray, edged;
    vector<vector<Point>> cnts;
    cvtColor(ref_image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    Canny(gray, edged, 35, 125);
    imshow("Distance", edged);

    findContours(edged, cnts, RETR_LIST, CHAIN_APPROX_SIMPLE);*/

    /*double object_width = 24.0;
    double focalLength = 11.0;
    double perWidth = cx;
    double distance = (object_width * focalLength) / perWidth;

    imshow("Distance", edged);


    cout << "Distance:" << distance << endl;*/

    /*************************************************************************
     *  4.0 DETECTION
     ************************************************************************/

    /*fileList.clear();
    for (int i = 1; i <= 20; i++) {
        stringstream str;
        str << "resource/image/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        fileList.push_back(str.str());
        image = imread(str.str(), 0);
    }

    cameraCalibrator.addChessboardPoints(fileList, boardSize);
    cameraCalibrator.calibrate((Size &) image.size);*/

    double cameraParams[4];
    cameraParams[0] = cameraCalibrator.getCameraMatrix().at<double>(0, 0);
    cameraParams[1] = cameraCalibrator.getCameraMatrix().at<double>(1, 1);
    cameraParams[2] = cx;
    cameraParams[3] = cy;

    pnp_detection.setMatrixParam(cameraParams);
    PnPProblem pnp_detection_est(cameraParams);

    initKalmanFilter(kalmanFilter, nStates, nMeasurements, nInputs, dt);
    Mat measurements(nMeasurements, 1, CV_64F);
    measurements.setTo(Scalar(0));
    bool good_measurement = false;
    //vector<Point3f> list_points3D_model;
    /*
     *
     vector<Point3f> list_points3D_model = model.get_points3d();
     Mat descriptors_model = model.get_descriptors();*/

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

    Mat frame, frame_vis, detection_model;

    Mat current_frame = imread(path_to_first_image);
    Mat first_frame = imread(path_to_second_image);

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


    //vanish_point = processImage(msac, numVps, gray_ref_image, output_ref_image);

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_1[good_matches[i].trainIdx].pt);
    }

    Mat H = findHomography(obj, scene, RANSAC);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(current_frame.cols, 0);
    obj_corners[2] = cvPoint(current_frame.cols, current_frame.rows);
    obj_corners[3] = cvPoint(0, current_frame.rows);
    std::vector<Point2f> scene_corners(4);

    line(img_matches, scene_corners[0] + Point2f(current_frame.cols, 0),
         scene_corners[1] + Point2f(current_frame.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f(current_frame.cols, 0),
         scene_corners[2] + Point2f(current_frame.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f(current_frame.cols, 0),
         scene_corners[3] + Point2f(current_frame.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f(current_frame.cols, 0),
         scene_corners[0] + Point2f(current_frame.cols, 0), Scalar(0, 255, 0), 4);

    perspectiveTransform(obj_corners, scene_corners, H);

    imshow("Good Matches & Object detection", img_matches);

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

    pthread_t fast_robust_matcher_t, robust_matcher_t;

    arg_struct fast_robust_matcher_arg_struct;
    fast_robust_matcher_arg_struct.detection_model = detection_model;
    fast_robust_matcher_arg_struct.frame = first_image;
    fast_robust_matcher_arg_struct.list_points3D_model = list_3D_points_after_registration;


    arg_struct *robust_matcher_arg_struct;

    //pthread_create(&fast_robust_matcher_t, NULL, fast_robust_matcher, (void *) &fast_robust_matcher_arg_struct);
    //pthread_create(&robust_matcher_t, NULL, robust_matcher, (void *) robust_matcher_arg_struct);


    while (cap.read(frame) && waitKey(30) != 27) {

        frame_vis = frame.clone();


        good_matches.clear();
        //rmatcher.fastRobustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

        /*list_points3d_model_match.clear();
        list_points2d_scene_match.clear();
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_3D_points_after_registration[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }

        draw2DPoints(frame_vis, list_points2d_scene_match, blue);*/

        rmatcher.robustMatch(frame_vis, good_matches, keypoints_scene, detection_model);

        list_points3d_model_match.clear();
        list_points2d_scene_match.clear();
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            Point3f point3d_model = list_3D_points_after_registration[good_matches[match_index].trainIdx];
            Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }

        Mat inliers_idx;
        vector<Point2f> list_points2d_inliers;

        if (matches.size() > 0) {
            try {
                pnp_detection.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match, pnpMethod,
                                                 inliers_idx, iterationsCount, reprojectionError, confidence);
            } catch (Exception e) {}

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

        //double cx = cameraCalibrator.getCameraMatrix().at<double>(0, 2);
        //double cy = cameraCalibrator.getCameraMatrix().at<double>(1, 2);

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
        updateKalmanFilter(kalmanFilter, measurements, translation_estimated, rotation_estimated);
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

        Mat r, t;
        vector<Point2f> ll = registration.get_points2d();
        vector<Point2f> kk = registration.get_points2d();


        Mat essential = findEssentialMat(ll, kk, cameraCalibrator.getCameraMatrix());
        //Mat essential = findEssentialMat(ll, kk,      cameraCalibrator.getCameraMatrix());
        correctMatches(essential, ll, kk, ll, kk);
        recoverPose(essential, ll, kk, r, t, 1, Point2f(cx, cy));

        double xAngle = atan2f(r.at<float>(2, 1), r.at<float>(2, 2));
        double yAngle = atan2f(-r.at<float>(2, 0),
                               sqrtf(r.at<float>(2, 1) * r.at<float>(2, 1) + r.at<float>(2, 2) * r.at<float>(2, 2)));
        double zAngle = atan2f(r.at<float>(1, 0), r.at<float>(0, 0));

        xAngle = (int) convert_radian_to_degree(xAngle);
        yAngle = (int) convert_radian_to_degree(yAngle);
        zAngle = (int) convert_radian_to_degree(zAngle);

        cout << "xAngle: " << xAngle << "%" << endl;
        cout << "yAngle: " << yAngle << "%" << endl;
        cout << "zAngle: " << zAngle << "%" << endl;

        imshow(WIN_REAL_TIME_DEMO, frame_vis);
    }
    //destroyWindow(WIN_REAL_TIME_DEMO);
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

    struct arg_struct *arg_struct = (struct arg_struct *) arg;
    Mat frame_vis = arg_struct->frame;
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


