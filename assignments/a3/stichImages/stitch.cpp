//
// Created by Yishi Wang on 2019-03-24.
//

#include "stitch.h"
#include "ransac.h"
#include <iostream>

using namespace std;

//// Step 4A
void stitch(Mat &image_1, Mat &image_2, Mat &hom, Mat &homInv, Mat &stitchedImage) {
    //// Step 4Aa
    float rows_2 = image_2.rows;
    float cols_2 = image_2.cols;
//    cout << cols_2 << ", " << rows_2 << endl;
    float x, y;
    float xMin = 0;
    float xMax = cols_2;
    float yMin = 0;
    float yMax = rows_2;
    project(0, 0, homInv, x, y);
//    cout << x << " " << y << endl;
    updateStitchedImageSize(xMin, xMax, yMin, yMax, x, y);

    project(0, rows_2, homInv, x, y);
//    cout << x << " " << y << endl;
    updateStitchedImageSize(xMin, xMax, yMin, yMax, x, y);

    project(cols_2, 0, homInv, x, y);
//    cout << x << " " << y << endl;
    updateStitchedImageSize(xMin, xMax, yMin, yMax, x, y);

    project(cols_2, rows_2, homInv, x, y);
//    cout << x << " " << y << endl;
    updateStitchedImageSize(xMin, xMax, yMin, yMax, x, y);
//    cout << xMin << " " << xMax << " " << yMin << " " << yMax << endl;

    updateStitchedImageSize(xMin, xMax, yMin, yMax, image_1.cols, image_1.rows);

    float diffX = 0 - xMin;
    float diffY = 0 - yMin;
//    cout << diffX << " " << diffY << endl;
//    cout << "new size: " << xMax-xMin  << " " << yMax-yMin << endl;

    stitchedImage = Mat::zeros((int)(yMax-yMin), (int)(xMax-xMin), CV_8UC3);
    //// Step 4Ab
    copyImageOne(image_1, stitchedImage, (int)diffX, (int)diffY);
    //// Step 4Ac
    copyImageTwo(image_2, stitchedImage, hom, (int)diffX, (int)diffY);
}

void updateStitchedImageSize(float &xMin, float &xMax, float &yMin, float &yMax, float x, float y) {
    if (x < xMin) {
        xMin = x;
    } else if (x > xMax) {
        xMax = x;
    }
    if (y < yMin) {
        yMin = y;
    } else if (y > yMax) {
        yMax = y;
    }
}


void copyImageOne(Mat &imageOne, Mat &stitchedImage, int diffX, int diffY) {
    for (int row = 0; row < imageOne.rows; ++row) {
        for (int col = 0; col < imageOne.cols; ++col) {
            stitchedImage.at<Vec3b>(row+diffY, col+diffX) = imageOne.at<Vec3b>(row, col);
        }
    }
}

void copyImageTwo(Mat &imageTwo, Mat &stitchedImage, Mat &hom, int diffX, int diffY) {

    vector<Point2f> verticalArtifactLine;
//    Mat originalStitchedImage = stitchedImage.clone();

    int maxCommonX = 0;                     // the max x coordinate value for common area
    int maxCommonY = 0;
    int minCommonX = stitchedImage.cols;
    int minCommonY = stitchedImage.rows;

    // variables to determine which image is left side and which image is right side
    int maxX_1 = 0;        // max x coordinate for image 1
    int minX_1 = 10000;    // min x coordinate for image 1
    int maxX_2 = 0;        // max x coordinate for image 2
    int minX_2 = 10000;    // min x coordinate for image 2

    for (int row_s = 0; row_s < stitchedImage.rows; ++row_s) {
        for (int col_s = 0; col_s < stitchedImage.cols; ++col_s) {
            float col_2, row_2;
            project(col_s - diffX, row_s - diffY, hom, col_2, row_2);
            Vec3b fOriginal = stitchedImage.at<Vec3b>(row_s, col_s);
            if (0 != fOriginal[0] || 0 != fOriginal[1] || 0 != fOriginal[2]) {
                if (col_s < minX_1) {
                    minX_1 = col_s;
                } else if (col_s > maxX_1) {
                    maxX_1 = col_s;
                }
                if (0 <= col_2 && col_2 < imageTwo.cols-1 && 0 <= row_2 && row_2 < imageTwo.rows-1) {
//                    stitchedImage.at<Vec3b>(row_s, col_s) = Vec3b(0,0,255); // set comman area to red
                    if (col_s < minCommonX) {
                        minCommonX = col_s;
                    } else if (col_s > maxCommonX) {
                        maxCommonX = col_s;
                    }
                    if (row_s < minCommonY) {
                        minCommonY = row_s;
                    } else if (row_s > maxCommonY) {
                        maxCommonY = row_s;
                    }
                }
            }
            if (0 <= col_2 && col_2 < imageTwo.cols && 0 <= row_2 && row_2 < imageTwo.rows) {
                if (col_s < minX_2) {
                    minX_2 = col_s;
                } else if (col_s > maxX_2) {
                    maxX_2 = col_s;
                }
            }
        }
    }

//    cout << minCommonX << " " << minCommonY << ", " << maxCommonX << ", " << maxCommonY << endl;
//    cout << minX_1 << ", " << maxX_1 << endl;
//    cout << minX_2 << ", " << maxX_2 << endl;

    int leftX = minCommonX;
    int deltaX = maxCommonX - minCommonX + 2;
    for (int row_s = 0; row_s < stitchedImage.rows; ++row_s) {
        for (int col_s = 0; col_s < stitchedImage.cols; ++col_s) {
            float col_2, row_2;
            project(col_s-diffX, row_s-diffY, hom, col_2, row_2);
            // Bilinear interpolation
            // https://en.wikipedia.org/wiki/Bilinear_interpolation
            // Q12(x1, y2)  Q22(x2, y2)
            //
            //
            // Q11(x1, y1)  Q21(x2, y1)
            float x = col_2;
            float y = row_2;
            int x1 = (int)floor(col_2);
            int x2 = (int)ceil(col_2);
            int y1 = (int)floor(row_2);
            int y2 = (int)ceil(row_2);

            if (0 <= x1 && x1 < imageTwo.cols && 0 <= y1 && y1 < imageTwo.rows &&
                0 <= x2 && x2 < imageTwo.cols && 0 <= y2 && y2 < imageTwo.rows) {
                Vec3b fQ11 = imageTwo.at<Vec3b>(y1, x1);
                Vec3b fQ12 = imageTwo.at<Vec3b>(y1, x2);
                Vec3b fQ21 = imageTwo.at<Vec3b>(y2, x1);
                Vec3b fQ22 = imageTwo.at<Vec3b>(y2, x2);
                Vec3b fxy = (1.0 / ((x2-x1)*(y2-y1))) * (fQ11*(x2-x)*(y2-y) + fQ21*(x-x1)*(y2-y) + fQ12*(x2-x)*(y-y1) + fQ22*(x-x1)*(y-y1));
                Vec3b fOriginal = stitchedImage.at<Vec3b>(row_s, col_s);
//                Vec3b fOriginal = originalStitchedImage.at<Vec3b>(row_s, col_s);


                // simple blending, half 0.5 weight for both image
//                if (0 == fOriginal[0] && 0 == fOriginal[1] && 0 == fOriginal[2]) {
//                    stitchedImage.at<Vec3b>(row_s, col_s) = fxy;
//                } else {
//                    stitchedImage.at<Vec3b>(row_s, col_s) = fxy*0.5 + fOriginal*0.5;
//                }


                // blending by using weight, line by line
                if (0 == fOriginal[0] && 0 == fOriginal[1] && 0 == fOriginal[2]) {
                    stitchedImage.at<Vec3b>(row_s, col_s) = fxy;
                } else {
                    float imageRightWeight = (col_s-leftX) * 1.0f / deltaX;
                    float imageLeftWeight = 1 - imageRightWeight;
                    // TODO: deal with ambiguous cases
                    // TODO: determine left or right based on the center point position
                    if (minX_1 - 15 < minX_2) {
                        stitchedImage.at<Vec3b>(row_s, col_s) = fOriginal * imageLeftWeight + fxy * imageRightWeight;
                    } else {
                        stitchedImage.at<Vec3b>(row_s, col_s) = fOriginal * imageRightWeight + fxy * imageLeftWeight;
                    }
                }
            }

        }
    }
}
