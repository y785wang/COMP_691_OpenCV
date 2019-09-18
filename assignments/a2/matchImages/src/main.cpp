#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "../headers/featureDetection.h"
#include "../headers/featureDescription.h"
#include "../headers/featureMatch.h"

using namespace std;
using namespace cv;


/**
 * Main function
 * @return 0 if run success
 */
int main() {


    //// load image one
    Mat colorImage1, grayImage1;
    colorImage1 = imread("images/img.png", IMREAD_COLOR);
    cvtColor(colorImage1, grayImage1, COLOR_BGR2GRAY);


    //// generate all KeyPoints for image one
    vector<KeyPoint> keyPoints1;
    findKeyPoints(grayImage1, keyPoints1);

    //// find all features for image one
//    vector<Feature> features1;
//    findFeatures(grayImage1, keyPoints1, features1);

    //// draw all KeyPoints on image one
//    Mat cornerImage1;
//    drawKeypoints(colorImage1, keyPoints1, cornerImage1, -1, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    imshow("cornerImage1", cornerImage1);



     //// for self test purpose, generate some rotate images
//    Point2f src_center(colorImage1.cols/2.0F, colorImage1.rows/2.0F);
//    Mat rot_mat = getRotationMatrix2D(src_center, 10, 1.0);
//    Mat dst;
//    warpAffine(colorImage1, dst, rot_mat, colorImage1.size());
//    imwrite("images/img_rotate_10°.png", dst);



    //// load image 2
    Mat colorImage2, grayImage2;
    colorImage2 = imread("images/img_rotate_10°.png", IMREAD_COLOR);
//    colorImage2 = imread("images/img_scale_70%.png", IMREAD_COLOR);
//    colorImage2 = imread("images/img_rotate_contrast.png", IMREAD_COLOR);
//    colorImage2 = imread("images/img_cut.png", IMREAD_COLOR);
//    colorImage2 = imread("images/img_half.png", IMREAD_COLOR);
    cvtColor(colorImage2, grayImage2, COLOR_BGR2GRAY);

    //// generate all KeyPoints for image two
    vector<KeyPoint> keyPoints2;
    findKeyPoints(grayImage2, keyPoints2);

    //// find all features for image two
//    vector<Feature> features2;
//    findFeatures(grayImage2, keyPoints2, features2);

    //// draw all KeyPoints on image two
//    Mat cornerImage2;
//    drawKeypoints(colorImage2, keyPoints2, cornerImage2);
//    imshow("cornerImage2", cornerImage2);



    //// find all matches, draw match lines
//    Mat resultImage;
//    vector<DMatch> dMatches;
//    findMatch(features1, features2, dMatches, resultImage);
//    drawMatches(colorImage1, keyPoints1, colorImage2, keyPoints2, dMatches, resultImage);
//    imshow("matchedImage", resultImage);




//    Mat colorImage1, colorImage2;
      Mat resultImage;
//    colorImage1 = imread("images/img.png", IMREAD_COLOR);
//    colorImage2 = imread("images/img_rotate_10°.png", IMREAD_COLOR);
//
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//    //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
//    //cv::Ptr<Feature2D> f2d = ORB::create();
//    // you get the picture, i hope..
//
//    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect( colorImage1, keypoints_1 );
    f2d->detect( colorImage2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute( colorImage1, keypoints_1, descriptors_1 );
    f2d->compute( colorImage2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    drawMatches(colorImage1, keypoints_1, colorImage2, keypoints_2, matches, resultImage);
    imshow("matchedImage", resultImage);

    waitKey(0);

    return 0;
}