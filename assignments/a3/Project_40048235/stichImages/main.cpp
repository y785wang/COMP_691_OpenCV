#include <iostream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "ransac.h"
#include "stitch.h"

using namespace std;
using namespace cv;

int main() {

    //// step 0: load two images
    String imagePath_1 = "project_images/Rainier1.png";
//    String imagePath_1 = "s54623.png";
    String imagePath_2 = "project_images/Rainier2.png";

//    String imagePath_1 = "self_images/d3.jpg";
//    String imagePath_1 = "self_images/self321.png";
//    String imagePath_2 = "self_images/d1.jpg";
//    String imagePath_2 = "self_images/self345.png";

//    String imagePath_1 = "project_images/Hanging1.png";
//    String imagePath_2 = "project_images/Hanging2.png";


    Mat image_1, image_2;
    image_1 = imread(imagePath_1, IMREAD_COLOR);
    image_2 = imread(imagePath_2, IMREAD_COLOR);
//    image_1 = imread("project_images/Boxes.png", IMREAD_COLOR);
//    image_2 = imread("project_images/Boxes.png", IMREAD_COLOR);

    // down-sampling
//    Mat temp_1, temp_11, temp_2, temp_22;
//    pyrDown(image_1, temp_1);
//    pyrDown(temp_1, temp_11);
//    pyrDown(temp_11, image_1);
//    imwrite("self_images/d5.jpg", image_1);
//    pyrDown(image_2, temp_2);
//    pyrDown(temp_2, temp_22);
//    pyrDown(temp_22, image_2);
//    imwrite("self_images/d4.jpg", image_2);
//    return 0;




    //// step 1: detect the interest points
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(500, 3, 0.15, 10, 1.6);
//    cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
//    cv::Ptr<Feature2D> f2d = ORB::create();

    std::vector<KeyPoint> keyPoints_1, keyPoints_2;
    f2d->detect(image_1, keyPoints_1);
    f2d->detect(image_2, keyPoints_2);
    Mat temp_1, temp_2;
    drawKeypoints(image_1, keyPoints_1, temp_1);
    drawKeypoints(image_2, keyPoints_2, temp_2);
    imshow("1b", temp_1);
//    imwrite("1b.png", temp_1);
    imshow("1c", temp_2);
//    imwrite("1c.png", temp_2);
//    waitKey(0);
//    return 0;




    //// step 2: compute descriptors, match them
    Mat descriptors_1, descriptors_2, matchResult;
    f2d->compute(image_1, keyPoints_1, descriptors_1);
    f2d->compute(image_2, keyPoints_2, descriptors_2);
    cout << "image_1 key points: " << keyPoints_1.size() << endl;
    cout << "image_2 key points: " << keyPoints_2.size() << endl;
    BFMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    drawMatches(image_1, keyPoints_1, image_2, keyPoints_2, matches, matchResult);
    imshow("2.png", matchResult);
//    imwrite("2.png", matchResult);
//    waitKey(0);
//    return 0;




    //// step 3: compute the homography
    Mat hom;
    Mat homInv;
    myRANSAC(matches, 50000, keyPoints_1, keyPoints_2, hom, homInv, image_1, image_2);




    //// step 4: stitch the images together
    Mat stitchedImage;
    stitch(image_1, image_2, hom, homInv, stitchedImage);
    imshow("4", stitchedImage);
//    imwrite("4.png", stitchedImage);
//    imwrite("self_images/self321345.png", stitchedImage);


    waitKey(0);

    return 0;
}