//
// Created by Yishi Wang on 2019-03-24.
//

#ifndef STICHIMAGES_STITCH_H
#define STICHIMAGES_STITCH_H

#include <opencv2/opencv.hpp>

using namespace cv;

void stitch(Mat &, Mat &, Mat &, Mat &, Mat &);

void updateStitchedImageSize(float &xMin, float &xMax, float &yMin, float &yMax, float x, float y);

void copyImageOne(Mat &imageOne, Mat &stitchedImage, int diffX, int diffY);

void copyImageTwo(Mat &imageTwo, Mat &stitchedImage, Mat &homInv, int diffX, int diffY);

#endif //STICHIMAGES_STITCH_H
