//
// Created by Yishi Wang on 2019-02-07.
//

#ifndef MATCHIMAGES_FEATUREDETECTION_H
#define MATCHIMAGES_FEATUREDETECTION_H

#include <opencv2/opencv.hpp>

bool checkLocalMaximum(cv::Mat &, int, int);

void findKeyPoints(cv::Mat &, std::vector<cv::KeyPoint> &);

#endif //MATCHIMAGES_FEATUREDETECTION_H
