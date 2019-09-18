//
// Created by Yishi Wang on 2019-03-24.
//

#include "ransac.h"
#include <opencv2/opencv.hpp>

//// Step 3A
void project(float x1, float y1, Mat &hom, float& x2, float& y2) {
    double w = hom.at<double>(2, 0) * x1 + hom.at<double>(2, 1) * y1 + hom.at<double>(2, 2);
    x2 = (float)((hom.at<double>(0, 0) * x1 + hom.at<double>(0, 1) * y1 + hom.at<double>(0, 2)) / w);
    y2 = (float)((hom.at<double>(1, 0) * x1 + hom.at<double>(1, 1) * y1 + hom.at<double>(1, 2)) / w);

//    vector<float> input;
//    input.push_back(x1);
//    input.push_back(x2);
//    input.push_back(1);
//    Mat output = hom * Mat(input);
//    x2 = output.at<float>(0, 0) / output.at<float>(0, 2);
//    y2 = output.at<float>(0, 1) / output.at<float>(0, 2);

//    vector<Point2f> inputArray, outputArray;
//    inputArray.emplace_back(Point2f(x1, y1));
//    perspectiveTransform(inputArray, outputArray, H);
//    if (outputArray[0].x != x2 || outputArray[0].y != y2) {
//        cout << w << endl;
//        cout << H << endl;
//        cout << inputArray[0] << outputArray[0] << ", " << x2 << " " << y2 << endl;
//    }

}

//// Step 3B
int computeInlierCount(Mat &hom, vector<DMatch> dMatches, float inlierThreshold,
        vector<KeyPoint> &keyPoints_1, vector<KeyPoint> &keyPoints_2,
        vector<DMatch> &goodDMatches, bool generateDMatches) {

    int numMatches = 0;

    for (DMatch dMatch : dMatches) {
        if (0 == hom.rows || 0 == hom.type()) continue;

        float x, y; // coordinates for image 2, obtains by using project function
        int index_1 = dMatch.queryIdx; // index for image 1
        int index_2 = dMatch.trainIdx; // index for image 2
        project(keyPoints_1[index_1].pt.x, keyPoints_1[index_1].pt.y, hom, x, y);

        // find distance
        float dX = (keyPoints_2[index_2].pt.x - x);
        float dY = (keyPoints_2[index_2].pt.y - y);
        float distance = sqrt(dX*dX + dY*dY);

        // compare with the inlier threshold
        if (distance < inlierThreshold) {
            ++numMatches;
            if (generateDMatches) goodDMatches.push_back(dMatch);
        }
    }

    return numMatches;
}


//// Step 3C
void myRANSAC(vector<DMatch> dMatches, int numIterations, vector<KeyPoint> keyPoints_1,
        vector<KeyPoint> keyPoints_2, Mat &hom, Mat &homInv, Mat &image_1, Mat &image_2) {

    int maxInlierCount = 0;
    vector<DMatch> emptyDMatch;
    RNG rng;

    //// Step 3Ca
    for(int i = 0; i < numIterations; ++i) {
        Mat newHom;
        vector<DMatch> matches2;
        vector<Point2f> fourPoints_1; // used for calculate newHom
        vector<Point2f> fourPoints_2; // used for calculate newHom

        // generate four indices for four random points
        int indices[] = {-1, -1, -1, -1};
        for (int j = 0; j < 4; ++j) {
            int tempIndex = rng.uniform(0, int(dMatches.size()));
            // avoiding duplicate random number
            for (int index : indices) {
                if (tempIndex == index) {
                    j--;
                    break;
                }
            }
            indices[j] = tempIndex;
        }

        // locate four points in image one, and their corresponding points in image two
        for (int j = 0; j < 4; ++j) {
            fourPoints_1.push_back(keyPoints_1[dMatches[indices[j]].queryIdx].pt);
            fourPoints_2.push_back(keyPoints_2[dMatches[indices[j]].trainIdx].pt);
            matches2.emplace_back(DMatch(j , j , 0));
        }

        // find hom and its corresponding inlier count
        newHom = findHomography(fourPoints_1, fourPoints_2, 0); // 0 - a regular method using all the points
        int inlierCount = computeInlierCount(newHom, dMatches, 4, keyPoints_1, keyPoints_2, emptyDMatch, false);

        // store the max inlier count
        if (inlierCount > maxInlierCount) {
            maxInlierCount = inlierCount;
            hom = newHom;
        }
    }

    homInv = hom.inv();

    //// Step 3Cb
    // use best hom, find best matches, draw them out
    Mat goodMatchResult;
    vector<DMatch> goodDMatches;
    computeInlierCount(hom, dMatches, 2, keyPoints_1, keyPoints_2, goodDMatches, true);
    drawMatches(image_1, keyPoints_1, image_2, keyPoints_2, goodDMatches, goodMatchResult);
    //// Step 3Cc
    imshow("3.png", goodMatchResult);
//    imwrite("3.png", goodMatchResult);
    cout << "goodDMatch bhgtes: " << goodDMatches.size() << endl;
//    waitKey(0);
}
