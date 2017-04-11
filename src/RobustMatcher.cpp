/**
 * RobustMatcher.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include <opencv/cv.hpp>
#include "RobustMatcher.h"

void RobustMatcher::computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints) {
    detector->detect(image, keypoints);
}

void
RobustMatcher::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    extractor->compute(image, keypoints, descriptors);
}

int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch> > &matches) {
    int removed = 0;
    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator = matches.begin();
         matchIterator != matches.end(); ++matchIterator) {

        if (matchIterator->size() > 1) {
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio) {
                matchIterator->clear();
                removed++;
            }
        } else {
            matchIterator->clear();
            removed++;
        }
    }
    return removed;
}

void RobustMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
                                 const std::vector<std::vector<cv::DMatch> > &matches2,
                                 std::vector<cv::DMatch> &symMatches) {

    for (std::vector<std::vector<cv::DMatch> >::const_iterator
                 matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {

        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2 = matches2.begin();
             matchIterator2 != matches2.end(); ++matchIterator2) {
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;


            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx,
                                                (*matchIterator1)[0].distance));
                break;
            }
        }
    }

}

void RobustMatcher::robustMatch(const cv::Mat &image2, std::vector<cv::DMatch> &good_matches,
                                std::vector<cv::KeyPoint> &key_points2) {


    detector->detect(image2, key_points2);

    cv::Mat descriptors2;
    extractor->compute(image2, key_points2, descriptors2);


    std::vector<std::vector<cv::DMatch>> matches12, matches21;

    matcher->knnMatch(descriptors, descriptors2, matches12, 2);
    matcher->knnMatch(descriptors2, descriptors, matches21, 2);

    cout << "Matcher " << matches12.size() << ", " << matches21.size() << endl;

    ratioTest(matches12);
    ratioTest(matches21);

    cout << "Ratio test " << matches12.size() << ", " << matches21.size() << endl;

    symmetryTest(matches12, matches21, good_matches);
    cout << "Symmetry test " << good_matches.size() << endl;
}

void RobustMatcher::fastRobustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                                    std::vector<cv::KeyPoint> &keypoints_frame,
                                    const cv::Mat &descriptors_model) {

    this->computeKeyPoints(frame, keypoints_frame);

    cv::Mat descriptors_frame;
    this->computeDescriptors(frame, keypoints_frame, descriptors_frame);

    std::vector<std::vector<cv::DMatch> > matches;
    matcher->knnMatch(descriptors_frame, descriptors_model, matches, 2);

    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator = matches.begin();
         matchIterator != matches.end(); ++matchIterator) {
        if (!matchIterator->empty()) good_matches.push_back((*matchIterator)[0]);
    }
}


cv::Mat RobustMatcher::ransacTest(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1,
                                  const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &outMatches) {

    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }

    std::vector<uchar> inliers(points1.size(), 0);
    cv::Mat fundemental = findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, distance,
                                             confidence);

    std::vector<uchar>::const_iterator itIn = inliers.begin();
    std::vector<cv::DMatch>::const_iterator itM = matches.begin();
    for (; itIn != inliers.end(); ++itIn, ++itM) {
        if (*itIn) {
            outMatches.push_back(*itM);
        }
    }

    if (refineF) {
        points1.clear();
        points2.clear();

        for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
             it != outMatches.end(); ++it) {
            float x = keypoints1[it->queryIdx].pt.x;
            float y = keypoints1[it->queryIdx].pt.y;
            points1.push_back(cv::Point2f(x, y));
            x = keypoints2[it->trainIdx].pt.x;
            y = keypoints2[it->trainIdx].pt.y;
            points2.push_back(cv::Point2f(x, y));
        }
        fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), CV_FM_8POINT);
    }

    return fundemental;
}


cv::Mat RobustMatcher::robustMatchRANSAC(cv::Mat &image1, cv::Mat &image2, std::vector<cv::DMatch> &matches,
                                         std::vector<cv::KeyPoint> &key_points1,
                                         std::vector<cv::KeyPoint> &key_points2) {

    detector->detect(image1, key_points1);
    detector->detect(image2, key_points2);

    cv::Mat descriptors1, descriptors2;
    extractor->compute(image1, key_points1, descriptors1);
    extractor->compute(image2, key_points2, descriptors2);

    descriptors = descriptors1.clone();


    BFMatcher matcher;

    std::vector<std::vector<cv::DMatch>> matches1, matches2;
    matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
    matcher.knnMatch(descriptors2, descriptors1, matches2, 2);

    ratioTest(matches1);
    ratioTest(matches2);

    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    cv::Mat fundemental = ransacTest(symMatches, key_points1, key_points2, matches);

    return fundemental;
}





