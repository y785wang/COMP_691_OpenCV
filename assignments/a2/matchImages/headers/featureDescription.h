//
// Created by Yishi Wang on 2019-02-07.
//

#ifndef MATCHIMAGES_FEATUREDESCRIPTION_H
#define MATCHIMAGES_FEATUREDESCRIPTION_H

#include <opencv2/opencv.hpp>

using cv::Mat;
using cv::KeyPoint;
using std::vector;

/**
 * Store relative info for a feature
 */
class Feature {
  private:
    Mat orientationHistogram;
    Mat descriptor;
    Mat grayImage;
    KeyPoint keyPoint;
  public:
    Feature(Mat &, KeyPoint &);
    int L(int, int);
    int L(int, int, Mat &);
    float calculateMagnitude(int, int);
    float calculateMagnitude(int, int, Mat &);
    float calculateTheta(int, int);
    float calculateTheta(int, int, Mat &);
    void calculateDescriptor(int);
    Mat getDescriptor();
};

void findFeatures(Mat &, vector<KeyPoint> &, vector<Feature> &);

#endif //MATCHIMAGES_FEATUREDESCRIPTION_H
