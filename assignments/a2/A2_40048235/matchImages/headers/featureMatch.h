//
// Created by Yishi Wang on 2019-02-07.
//

#ifndef MATCHIMAGES_FEATUREMATCH_H
#define MATCHIMAGES_FEATUREMATCH_H

#include "../headers/featureDescription.h"

using std::vector;
using cv::Mat;

void findMatch(vector<Feature> &, vector<Feature> &, vector<cv::DMatch> &, Mat &);

float SSD(Feature &feature1, Feature &feature2);

#endif //MATCHIMAGES_FEATUREMATCH_H
