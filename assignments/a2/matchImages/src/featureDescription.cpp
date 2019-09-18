//
// Created by Yishi Wang on 2019-02-07.
//

#include "../headers/featureDescription.h"

using namespace cv;
using namespace std;


/**
 * Ctor
 * @param grayImage is the gray image
 * @param keyPoint is the key point for this feature
 */
Feature::Feature(Mat &grayImage, KeyPoint &keyPoint) {
    this->grayImage = grayImage;
    this->keyPoint = keyPoint;
    orientationHistogram = Mat::zeros(16, 8, CV_32FC1);
};


/**
 * Get a original gray image value for a given coordinate
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @return the gray value
 */
int Feature::L(int x, int y) { return (int)grayImage.at<uchar>(y, x); }


/**
 * Get a cut window value from original gray image for a givem corrdinate
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @param window18 is the cut windows from original gray image
 * @return the gray value
 */
int Feature::L(int x, int y, Mat &window18) { return (int)window18.at<uchar>(y, x); }


/**
 * Calculate the magnitude from original gray image
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @return the magnitude
 */
float Feature::calculateMagnitude(int x, int y) {
    return (float)sqrt(pow(L(x+1, y) - L(x-1, y), 2) + pow(L(x, y+1) - L(x, y-1), 2));
}


/**
 * Calculate the magnitude from the cut window
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @param window18 is the window image
 * @return the magnitude
 */
float Feature::calculateMagnitude(int x, int y, Mat &window18) {
    return (float)sqrt(pow(L(x+1, y, window18) - L(x-1, y, window18), 2) + pow(L(x, y+1, window18) - L(x, y-1, window18), 2));
}


/**
 * Calculate the relative theta corresponding to the key point from the original gray image
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @return the theta of range 0 ~ 360
 */
float Feature::calculateTheta(int x, int y) {
    // calculate theta for a pixel
    float vertical = L(x, y+1) - L(x, y-1); // vertical vector points down
    float horizontal = L(x+1, y) - L(x-1, y); // horizontal vector points right
    float theta = 0;
    if (0 != horizontal) theta = (float)(atan(1.0 * vertical / horizontal) * 180 / M_PI);
    if (horizontal < 0) theta += 180;
    if (horizontal >= 0 && vertical < 0) theta += 360;

    // calculate theta between pixel and key point
    float rHorizontal = y - (int)keyPoint.pt.y;
    float rVertical   = x - (int)keyPoint.pt.x;
    float rTheta = 0;
    if (0 != rHorizontal) rTheta = (float)(atan(1.0 * rVertical / rHorizontal) * 180 / M_PI);
    if (rHorizontal < 0) rTheta += 180;
    if (rHorizontal >= 0 && rVertical < 0) rTheta += 360;

    // calculate the relative theta for a pixel relative to the key point
    float relativeTheta = theta - rTheta;
    if (relativeTheta < 0) relativeTheta += 360;

    return relativeTheta;
}


/**
 * Calculate the theta from the cut window
 * @param x is the x coordinate
 * @param y is the y coordinate
 * @param window18 is the window image
 * @return the theta of range 0 ~ 360
 */
float Feature::calculateTheta(int x, int y, Mat &window18) {
    float vertical = L(x, y+1, window18) - L(x, y-1, window18); // vertical vector points down
    float horizontal = L(x+1, y, window18) - L(x-1, y, window18); // horizontal vector points right
    auto theta = (float)(atan2(vertical, horizontal) * 180 / M_PI);
    if (theta < 0) theta += 360;

    return theta;
}


/**
 * Generate the descriptor for a KeyPoint
 */
void Feature::calculateDescriptor (int method) {
    // method = 1; // cut a 18*18 window out
    // method = 2; // use original gray image directly

    if (1 == method) {
        // top-left cell position of 18*18 window
        int x18 = (int)keyPoint.pt.x-9;
        int y18 = (int)keyPoint.pt.y-9;
        Mat image18 = Mat(grayImage, Rect(x18, y18, 18, 18));

        //// scale invariance
        auto gX = getGaussianKernel(19, 0, CV_32F);
        auto gY = getGaussianKernel(19, 0, CV_32F);
//        Mat gFilter = gX * gY.t();
//        for (int i = 0; i < 18; ++i) {
//            for (int j = 0; j < 18; ++j) {
//                image18.at<float>(i, j) *= gFilter.at<float>(i, j);
//            }
//        }
        Mat kernel = getGaussianKernel(3, 0.5, CV_32F);
        filter2D(image18, image18, image18.depth(), kernel);

        Mat magnitudes(16, 16, CV_32F);
        Mat thetas(16, 16, CV_32F);
        float dominateMagnitude = 0;
        float dominateTheta = 0;
        for(int i = 0; i < 4; ++i) { // col
            for (int j = 0; j < 4; ++j) { // row
                // top-left cell position of 4*4 grid
                int x4 = x18 + i*4 + 1;
                int y4 = y18 + j*4 + 1;
                for(int x = x4; x < x4+4; ++x) {
                    for (int y = y4; y < y4+4; ++y) {
                        float magnitude = calculateMagnitude(x-x18, y-y18, image18);
                        magnitudes.at<float>(y-y18-1, x-x18-1) = magnitude;
                        float theta = calculateTheta(x-x18, y-y18, image18);
                        thetas.at<float>(y-y18-1, x-x18-1) = theta;
                        if (magnitude > dominateMagnitude) {
                            dominateMagnitude = magnitude;
                            dominateTheta = theta;
                        }
                    }
                }
            }
        }

//        cout << dominateTheta << endl;

        //// rotate invariance
        for(int i = 0; i < 4; ++i) { // col
            for (int j = 0; j < 4; ++j) { // row
                // top-left cell position of 4*4 grid
                int x4 = x18 + i*4 + 1;
                int y4 = y18 + j*4 + 1;
                for(int x = x4; x < x4+4; ++x) {
                    for (int y = y4; y < y4+4; ++y) {
                        float theta = thetas.at<float>(y-y18-1, x-x18-1) - ((int)(dominateTheta / 10) * 10 + 5);
                        if (theta < 0) theta += 360;
                        thetas.at<float>(y-y18-1, x-x18-1) = theta;
                    }
                }
            }
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int x4 = i * 4;
                int y4 = j * 4;
                for (int x = x4; x < x4+4; ++x) {
                    for (int y = y4; y < y4+4; ++y) {
                        int position = (int)(thetas.at<float>(y, x) / 45);
                        if (8 == position) position = 0;
                        orientationHistogram.at<float>(i*4+j, position) += magnitudes.at<float>(y, x);
                    }
                }
            }
        }
    } else {
        // top-left cell position of 16*16 windows
        Mat magnitudes(16, 16, CV_32F);
        Mat thetas(16, 16, CV_32F);
        int x16 = (int) keyPoint.pt.x - 8;
        int y16 = (int) keyPoint.pt.y - 8;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {

                // top-left cell position of 4*4 grid
                int x4 = x16 + i * 4;
                int y4 = y16 + j * 4;
                for (int x = x4; x < x4 + 4; ++x) {
                    for (int y = y4; y < y4 + 4; ++y) {
                        float magnitude = calculateMagnitude(x, y);
                        magnitudes.at<float>(y - y16, x - x16) = magnitude;
                        float theta = calculateTheta(x, y);
                        thetas.at<float>(y - y16, x - x16) = theta;
                    }
                }
            }
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int x4 = i * 4;
                int y4 = j * 4;
                for (int x = x4; x < x4 + 4; ++x) {
                    for (int y = y4; y < y4 + 4; ++y) {
                        int position = (int) (thetas.at<float>(y, x) / 45);
                        if (8 == position) position = 0;
                        orientationHistogram.at<float>(i * 4 + j, position) += magnitudes.at<float>(y, x);
                    }
                }
            }
        }
    }


    //// contrast invariant
//    float summation = 0;
//    for (int i = 0; i < 16; ++i) {
//        for (int j = 0; j < 8; ++j) {
//            summation += pow(orientationHistogram.at<float>(i, j), 2);
//        }
//    }
//    for (int i = 0; i < 16; ++i) {
//        for (int j = 0; j < 8; ++j) {
//            float temp = orientationHistogram.at<float>(i, j);
//            temp /= sqrt(summation);
//            orientationHistogram.at<float>(i, j) = (temp > 0.2f) ? 0.2f : temp;
//        }
//    }

    for (int i = 0; i < 16; ++i) {
        float summation = 0;
        for (int j = 0; j < 8; ++j) {
            summation += pow(orientationHistogram.at<float>(i, j), 2);;
        }
        float threshold = sqrt(summation) * 0.2f;
        summation = 0;
        for (int j = 0; j < 8; ++j) {
            float temp = orientationHistogram.at<float>(i, j);
            if (temp > threshold) temp = threshold;
            summation += temp;
            orientationHistogram.at<float>(i, j) = temp;
        }
        summation = sqrt(summation);
        for (int j = 0; j < 8; ++j) {
            orientationHistogram.at<float>(i, j) /= summation;
        }
    }

    descriptor = orientationHistogram.clone();
//    normalize(orientationHistogram, descriptor, 0, 1, NORM_MINMAX, CV_32F);
}


/**
 * Get the feature descriptor
 * @return
 */
Mat Feature::getDescriptor() { return descriptor; }


/**
 * Find features for all given key points
 * @param grayImage is the original gray image
 * @param keyPoints are all the key points
 * @param features are all the features
 */
void findFeatures(Mat &grayImage, vector<cv::KeyPoint> &keyPoints, vector<Feature> & features) {
    int maxX = grayImage.cols - 9;
    int maxY = grayImage.rows - 9;
    for (KeyPoint keyPoint : keyPoints) {
        // for convenient purpose, ignore the keyPoint near the border
        int x = (int)keyPoint.pt.x;
        int y = (int)keyPoint.pt.y;
        if (9 < x && x < maxX && 9 < y && y < maxY) {
            Feature feature = Feature(grayImage, keyPoint);
            feature.calculateDescriptor(1);
            features.push_back(feature);
        }
    }
}
