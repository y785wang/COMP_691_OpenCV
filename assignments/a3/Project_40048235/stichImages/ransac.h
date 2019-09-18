//
// Created by Yishi Wang on 2019-03-24.
//

#ifndef STICHIMAGES_RANSAC_H
#define STICHIMAGES_RANSAC_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void project(float x1, float y1, Mat &hom, float& x2, float& y2);

int computeInlierCount(Mat &hom, vector<DMatch> dMatches, float inlierThreshold,
                        vector<KeyPoint> &keyPoints_1, vector<KeyPoint> &keyPoints_2,
                        vector<DMatch> &goodDMatches, bool generateDMatches);

void myRANSAC(vector<DMatch> dMatches, int numIterations, vector<KeyPoint> keyPoints_1,
              vector<KeyPoint> keyPoints_2, Mat &hom, Mat &homInv, Mat &image_1, Mat &image_2);

#endif //STICHIMAGES_RANSAC_H
