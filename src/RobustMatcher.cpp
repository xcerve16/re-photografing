/*
 * RobustMatcher.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#include <opencv/cv.hpp>
#include "RobustMatcher.h"

RobustMatcher::~RobustMatcher() {
}

void RobustMatcher::computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints) {
    detector->detect(image, keypoints);
}

void
RobustMatcher::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    extractor->compute(image, keypoints, descriptors);
}

int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch> > &matches) {
    int removed = 0;
    // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator
                 matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator) {
        // if 2 NN has been identified
        if (matchIterator->size() > 1) {
            // check distance ratioTest
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio) {
                matchIterator->clear(); // remove match
                removed++;
            }
        } else { // does not have 2 neighbours
            matchIterator->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

void RobustMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
                                 const std::vector<std::vector<cv::DMatch> > &matches2,
                                 std::vector<cv::DMatch> &symMatches) {

    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::const_iterator
                 matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {

        // ignore deleted matches
        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch> >::const_iterator
                     matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
            // ignore deleted matches
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx ==
                (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx ==
                (*matchIterator1)[0].trainIdx) {
                // add symmetrical match
                symMatches.push_back(
                        cv::DMatch((*matchIterator1)[0].queryIdx,
                                   (*matchIterator1)[0].trainIdx,
                                   (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }

}

void RobustMatcher::robustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                                std::vector<cv::KeyPoint> &keypoints_frame, const cv::Mat &descriptors_model) {

    // 1a. Detection of the ORB features
    this->computeKeyPoints(frame, keypoints_frame);

    // 1b. Extraction of the ORB descriptors
    cv::Mat descriptors_frame;
    this->computeDescriptors(frame, keypoints_frame, descriptors_frame);

    // 2. Match the two image descriptors
    std::vector<std::vector<cv::DMatch> > matches12, matches21;

    // 2a. From image 1 to image 2
    try {
        matcher->knnMatch(descriptors_frame, descriptors_model, matches12, 2); // return 2 nearest neighbours
    } catch (Exception e) {

    }


    // 2b. From image 2 to image 1
    try {
        matcher->knnMatch(descriptors_model, descriptors_frame, matches21,2); // return 2 nearest neighbours
    } catch (Exception e) {

    }

    // 3. Remove matches for which NN ratioTest is > than threshold
    // clean image 1 -> image 2 matches
    ratioTest(matches12);
    // clean image 2 -> image 1 matches
    ratioTest(matches21);

    // 4. Remove non-symmetrical matches
    symmetryTest(matches12, matches21, good_matches);

}

void RobustMatcher::fastRobustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
                                    std::vector<cv::KeyPoint> &keypoints_frame,
                                    const cv::Mat &descriptors_model) {
    good_matches.clear();

    // 1a. Detection of the ORB features
    this->computeKeyPoints(frame, keypoints_frame);

    // 1b. Extraction of the ORB descriptors
    cv::Mat descriptors_frame;
    this->computeDescriptors(frame, keypoints_frame, descriptors_frame);

    // 2. Match the two image descriptors
    std::vector<std::vector<cv::DMatch> > matches;
    matcher->knnMatch(descriptors_frame, descriptors_model, matches, 2);

    // 3. Remove matches for which NN ratioTest is > than threshold
    ratioTest(matches);

    // 4. Fill good matches container
    for (std::vector<std::vector<cv::DMatch> >::iterator
                 matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator) {
        if (!matchIterator->empty()) good_matches.push_back((*matchIterator)[0]);
    }

}


cv::Mat RobustMatcher::ransacTest(const std::vector<cv::DMatch> &matches,
                   const std::vector<cv::KeyPoint> &keypoints1,
                   const std::vector<cv::KeyPoint> &keypoints2,
                   std::vector<cv::DMatch> &outMatches) {

    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {

        // Get the position of left keypoints
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        // Get the position of right keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }

    std::vector<uchar> inliers(points1.size(), 0);
    cv::Mat fundemental = findFundamentalMat(
            cv::Mat(points1), cv::Mat(points2), // matching points
            inliers,      // match status (inlier ou outlier)
            CV_FM_RANSAC, // RANSAC method
            distance,     // distance to epipolar line
            confidence);  // confidence probability

    // extract the surviving (inliers) matches
    std::vector<uchar>::const_iterator itIn = inliers.begin();
    std::vector<cv::DMatch>::const_iterator itM = matches.begin();
    // for all matches
    for (; itIn != inliers.end(); ++itIn, ++itM) {

        if (*itIn) { // it is a valid match

            outMatches.push_back(*itM);
        }
    }

    //cout << "Number of matched points (after cleaning): " << outMatches.size() << std::endl;

    if (refineF) {
        // The F matrix will be recomputed with all accepted matches

        // Convert keypoints into Point2f for final F computation
        points1.clear();
        points2.clear();

        for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
             it != outMatches.end(); ++it) {

            // Get the position of left keypoints
            float x = keypoints1[it->queryIdx].pt.x;
            float y = keypoints1[it->queryIdx].pt.y;
            points1.push_back(cv::Point2f(x, y));
            // Get the position of right keypoints
            x = keypoints2[it->trainIdx].pt.x;
            y = keypoints2[it->trainIdx].pt.y;
            points2.push_back(cv::Point2f(x, y));
        }

        // Compute 8-point F from all accepted matches
        fundemental = cv::findFundamentalMat(
                cv::Mat(points1), cv::Mat(points2), // matching points
                CV_FM_8POINT); // 8-point method
    }

    return fundemental;
}


cv::Mat RobustMatcher::matchSimple(cv::Mat &image1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches,
                    std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2) {

    detector->detect(image1, keypoints1);

    cv::Mat descriptors1;
    extractor->compute(image1, keypoints1, descriptors1);

    BFMatcher matcher;

    std::vector<std::vector<cv::DMatch>> matches1;
    matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
    std::vector<std::vector<cv::DMatch>> matches2;
    matcher.knnMatch(descriptors2, descriptors1, matches2, 2);

    int removed = ratioTest(matches1);
    removed = ratioTest(matches2);

    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    cv::Mat fundemental = ransacTest(symMatches, keypoints1, keypoints2, matches);
    return fundemental;
}

cv::Mat RobustMatcher::match(cv::Mat &image1, cv::Mat &image2, std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints1,
      std::vector<cv::KeyPoint> &keypoints2) {

    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    cv::Mat descriptors1, descriptors2;
    extractor->compute(image1, keypoints1, descriptors1);
    extractor->compute(image2, keypoints2, descriptors2);

    BFMatcher matcher;
    std::vector<std::vector<cv::DMatch>> matches1;
    matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
    std::vector<std::vector<cv::DMatch>> matches2;
    matcher.knnMatch(descriptors2, descriptors1, matches2, 2);

    int removed = ratioTest(matches1);
    removed = ratioTest(matches2);

    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    cv::Mat fundemental = ransacTest(symMatches, keypoints1, keypoints2, matches);
    return fundemental;
}



