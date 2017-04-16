/**
 * Utils.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

#include "PnPProblem.h"

void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);

void draw2DPoint(cv::Mat image, cv::Point2f &points, cv::Scalar color);

cv::Mat rot2euler(const cv::Mat & rotationMatrix);

cv::Mat euler2rot(const cv::Mat & euler);




#endif
