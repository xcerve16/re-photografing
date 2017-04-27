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

class RobustMatcher {

private:

    cv::Ptr<cv::FeatureDetector> detector;

    cv::Ptr<cv::DescriptorExtractor> extractor;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    cv::Mat descriptors;

    float ratio;

    double distance;

    double confidence;

public:
    RobustMatcher();

    ~RobustMatcher() {};

    void setFeatureDetector(const cv::Ptr<cv::FeatureDetector> &detect) { detector = detect; }

    void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor> &desc) { extractor = desc; }

    void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher> &match) { matcher = match; }

    void setMinDistanceToEpipolar(double d) { distance = d; }

    void setConfidenceLevel(double c) { confidence = c; }

    void setRatio(float r) { ratio = r; }

    int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);


    void symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
                      const std::vector<std::vector<cv::DMatch> > &matches2,
                      std::vector<cv::DMatch> &symMatches);

    void robustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                     std::vector<cv::KeyPoint> &keypoints_frame);

    cv::Mat robustMatchRANSAC(cv::Mat &image1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches,
                              std::vector<cv::KeyPoint> &key_points1,
                              std::vector<cv::KeyPoint> &key_points2, cv::Mat cameraMatrix);

    cv::Mat ransacTest(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1,
                       const std::vector<cv::KeyPoint> &keypoints2,
                       std::vector<cv::DMatch> &outMatches, cv::Mat cameraMatrix);


};

#endif
