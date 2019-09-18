//
// Created by Yishi Wang on 2019-02-07.
//

#include "featureDetection.h"

using namespace cv;
using namespace std;


/**
 * Do the non-maximum suppression. Check all 8 points around a given
 * pixel.
 * @param c is the gray image
 * @param x is the pixel x coordinate
 * @param y is the pixel y coordinate
 * @return true if the give pixel is the max compare from its neighbour, false otherwise
 */
bool checkLocalMaximum(Mat &c, int x, int y) {
    float response = c.at<float>(x, y);
    return (0 != response &&
            response >= c.at<float>(x - 1, y - 1) &&
            response >= c.at<float>(x - 1, y) &&
            response >= c.at<float>(x - 1, y + 1) &&
            response >= c.at<float>(x, y - 1) &&
            response >= c.at<float>(x, y + 1) &&
            response >= c.at<float>(x + 1, y - 1) &&
            response >= c.at<float>(x + 1, y) &&
            response >= c.at<float>(x + 1, y + 1));
}


/**
 * Find all KeyPoints for the image
 * @param grayImage is the gray image
 * @param keyPoints is the place where all KeyPoints are stored
 */
void findKeyPoints(Mat &grayImage, vector<cv::KeyPoint> &keyPoints) {

    Mat Ix, Iy, IxIy, IxIx, IyIy;
    int depth = CV_32F;
    int scale = 1;
    int delta = 0;

    //// Step 1A
    // calculate gradient
    Scharr(grayImage, Ix, depth , 1, 0, scale, delta, BORDER_DEFAULT);
    Scharr(grayImage, Iy, depth , 0, 1, scale, delta, BORDER_DEFAULT);
//    Sobel(originalGrayImage, Ix, depth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//    Sobel(originalGrayImage, Iy, depth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    //// Step 1B
    IxIx = Ix.mul(Ix);
    IyIy = Iy.mul(Iy);
    IxIy = Ix.mul(Iy);

    // apply Gaussian to each
    Mat GIxIx, GIyIy, GIxIy;
    GaussianBlur(IxIx, GIxIx, Size(5, 5), 0, 0, BORDER_DEFAULT);
    GaussianBlur(IyIy, GIyIy, Size(5, 5), 0, 0, BORDER_DEFAULT);
    GaussianBlur(IxIy, GIxIy, Size(5, 5), 0, 0, BORDER_DEFAULT);

    // generate all corners that satisfy the threshold
    float threshold = 275000;
    int counter = 0; // for self test purpose
    Mat c = Mat::zeros(grayImage.rows, grayImage.cols, CV_32F); // response matrix
    for (int i = 0; i < grayImage.rows; ++i) {
        for (int j = 0; j < grayImage.cols; ++j) {
            float Ha = GIxIx.at<float>(i, j);
            float Hb = GIxIy.at<float>(i, j);
            float Hc = Hb;
            float Hd = GIyIy.at<float>(i, j);
            //// Step 1C
            float response;
            if (0 == Ha + Hd) {
                response = 0;
            } else {
                response = 1.0f * (Ha * Hd - Hb * Hc) / (Ha + Hd);
            }
            //// Step 1D
            if (response > threshold) {
                c.at<float>(i, j) = response;
                ++counter;
            }
        }
    }
//    cout << "key points:                " << counter << endl;
    counter = 0;

    // remove all corners which is not local maximum, ignore the border for convenience
    for (int i = 1; i < c.rows-1; ++i) {
        for (int j = 1; j < c.cols-1; ++j) {
            if (i < 8 || i > c.rows-8 || j < 8 || j > c.cols-8) continue;
            if (checkLocalMaximum(c, i, j)) {
                // to make it simple, ignore all keyPoint near border
                ++counter;
                KeyPoint keyPoint(j, i, 15);
                keyPoint.response = c.at<float>(i, j);
                keyPoints.push_back(keyPoint);
            }
        }
    }
//    cout << "key points local max only: " << counter << endl << endl;
}