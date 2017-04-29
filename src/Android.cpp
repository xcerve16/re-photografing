//
// Created by acervenka2 on 30.04.2017.
//

#include "Android.h"

int main(int argc, char *argv[]) {

    process.initReconstruction(cv::imread(path_to_first_image), cv::imread(path_to_second_image),
                               cv::imread(path_to_ref_image), cv::Point2f(273, 339), cv::Point2f(585, 548),
                               cv::Point2f(0, 0),
                               cv::Point2f(0, 0));
    process.processReconstruction();
    cv::Mat out;
    process.registrationPoints(0, 0, out);
    process.nextPoint();
    process.initNavigation();
}