/**
 * RobustMatcher.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef ROBUSTMATCHER_H_
#define ROBUSTMATCHER_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class RobustMatcher {

    const string ERROR_KNN_MATCHING = "Error while KNN matching";

private:

    cv::Ptr<cv::FeatureDetector> detector;

    cv::Ptr<cv::DescriptorExtractor> extractor;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    float ratio;

    bool refineF;

    double distance;

    double confidence;

public:
    RobustMatcher() : ratio(0.65f), refineF(true), confidence(0.99), distance(3.0) {

        detector = SURF::create();
        extractor = SURF::create();

        matcher = cv::makePtr<cv::BFMatcher>((int) cv::NORM_HAMMING, false);
    }

    void setFeatureDetector(const cv::Ptr<cv::FeatureDetector> &detect) { detector = detect; }

    void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor> &desc) { extractor = desc; }

    void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher> &match) { matcher = match; }

    void computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);

    void computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    void setMinDistanceToEpipolar(double d) { distance = d; }

    void setConfidenceLevel(double c) { confidence = c; }

    void setRatio(float r) { ratio = r; }

    int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);


    void symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
                      const std::vector<std::vector<cv::DMatch> > &matches2,
                      std::vector<cv::DMatch> &symMatches);

    void robustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                     std::vector<cv::KeyPoint> &keypoints_frame,
                     const cv::Mat &descriptors_model);

    void fastRobustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                         std::vector<cv::KeyPoint> &keypoints_frame,
                         const cv::Mat &descriptors_model);

    Mat robustMatchRANSAC(Mat &image1, Mat &descriptors2, vector<DMatch> &matches, vector<KeyPoint> &key_points1,
                          vector<KeyPoint> &key_points2);

    Mat
    ransacTest(const vector<DMatch> &matches, const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
               vector<DMatch> &outMatches);

    void tryKnnMatch(Mat frame, const Mat &model, vector<vector<DMatch>> matches, int k);
};

#endif
