#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
 * Calculate the difference between two images
 * @param originalImage is the original image
 * @param demosaicImage is the demosaic image
 * @param diffImage is the difference image
 */
void diff(Mat &originalImage, Mat &demosaicImage, Mat &diffImage) {

    originalImage.convertTo(originalImage, CV_32FC3);
    demosaicImage.convertTo(demosaicImage, CV_32FC3);
    diffImage.convertTo(diffImage, CV_32FC3);

    for (int i = 0; i < originalImage.rows; ++i) {
        for (int j = 0; j < originalImage.cols; ++j) {
            Vec3f pixelOriginal = originalImage.at<Vec3f>(i, j);
            Vec3f pixelDemosaic = demosaicImage.at<Vec3f>(i, j);
            Vec3f pixelDiff;
            pixelDiff[0] = sqrt(abs(pixelOriginal[0]*pixelOriginal[0] - pixelDemosaic[0]*pixelDemosaic[0]));
            pixelDiff[1] = sqrt(abs(pixelOriginal[1]*pixelOriginal[1] - pixelDemosaic[1]*pixelDemosaic[1]));
            pixelDiff[2] = sqrt(abs(pixelOriginal[2]*pixelOriginal[2] - pixelDemosaic[2]*pixelDemosaic[2]));
            diffImage.at<Vec3f>(i, j) = pixelDiff;
        }
    }

    originalImage.convertTo(originalImage, CV_8UC3);
    demosaicImage.convertTo(demosaicImage, CV_8UC3);
    diffImage.convertTo(diffImage, CV_8UC3);
}


/**
 * For assignment part one
 * @param originalImage is the original rgb image
 * @param blueImage is the deparatd blue channel image
 * @param greenImage is the separated green channel image
 * @param redImage is the separated red channel image
 */
void partOne(Mat &originalImage, Mat &blueImage, Mat &greenImage, Mat &redImage) {
    Mat demosaicImage;
    // combine three channels to one rgb image
    vector<Mat> rgbImages;
    rgbImages.push_back(blueImage);
    rgbImages.push_back(greenImage);
    rgbImages.push_back(redImage);
    merge(rgbImages, demosaicImage);

    // calculate the square difference
    Mat diffImage = Mat(originalImage.rows, originalImage.cols, 16);
    diff(originalImage, demosaicImage, diffImage);

    // concat images
    Mat concatImage;
    hconcat(originalImage, demosaicImage, concatImage);
    hconcat(concatImage, diffImage, concatImage);

    namedWindow( "Part 1 image", WINDOW_AUTOSIZE );
    imshow("Part 1 image", concatImage);
    imwrite("combined_image_2_p1.jpg", concatImage);
    cout << "p1" << sum(diffImage) << endl;
}


/**
 * part two, do the b-r and g-r first, median filter, then add r back
 * @param originalImage is the original image
 * @param blueImage is the separated blue channel image
 * @param greenImage is the separated green channel image
 * @param redImage is the separated red channel image
 */
void partTwo(Mat &originalImage, Mat &blueImage, Mat &greenImage, Mat &redImage) {
    Mat newB, newG, newR, finalB, finalG, demosaicImage;
    Mat blurB, blurG;
    blueImage.convertTo(newB, CV_32F);
    greenImage.convertTo(newG, CV_32F);
    redImage.convertTo(newR, CV_32F);
    newB -= newR;
    newG -= newR;
    medianBlur(newB, blurB, 5);
    medianBlur(newG, blurG, 5);
    blurB += newR;
    blurG += newR;
    blurB.convertTo(finalB, CV_8U);
    blurG.convertTo(finalG, CV_8U);
    vector<Mat> rgb;
    rgb.push_back(finalB);
    rgb.push_back(finalG);
    rgb.push_back(redImage);
    merge(rgb, demosaicImage);

    // calculate diff
    Mat diffImage = Mat(originalImage.rows, originalImage.cols, 16);
    diff(originalImage, demosaicImage, diffImage);

    // concat images
    Mat concatImage;
    hconcat(originalImage, demosaicImage, concatImage);
    hconcat(concatImage, diffImage, concatImage);

    namedWindow( "Part 2 image", WINDOW_AUTOSIZE );
    imshow("Part 2 image", concatImage);
    imwrite("combined_image_2_p2.jpg", concatImage);
    cout << "p2" << sum(diffImage) << endl;
}


/**
 * Main function
 * @return
 */
int main() {

    // setup the image set path
    String imageDirectory = "image_set/";
    String bayerImageNames[3] = {"crayons_mosaic.bmp", "oldwell_mosaic.bmp", "pencils_mosaic.bmp"};
    String rgbImageNames[3] = {"crayons.jpg", "oldwell.jpg", "pencils.jpg"};
    int testImageIndex = 1; // 0 ~ 2

    // read images
    Mat originalImage = imread(imageDirectory + rgbImageNames[testImageIndex], IMREAD_COLOR);
    Mat bayerImage = imread(imageDirectory + bayerImageNames[testImageIndex], IMREAD_GRAYSCALE);
    if (!originalImage.data || !bayerImage.data) {
        cout << "ERROR: Could not load input image." << endl;
        return -1;
    }

    // separate the b/g/r channels
    Mat partialBlueImage = Mat::zeros(bayerImage.rows, bayerImage.cols, 0);
    for (int i = 0; i < bayerImage.rows; ++i) {
        for (int j = 0; j < bayerImage.cols; ++j) {
            if (0 == i % 2 && 0 == j % 2) {
                partialBlueImage.at<uchar>(i, j) = bayerImage.at<uchar>(i, j);
            }
        }
    }
    Mat partialGreenImage = Mat::zeros(bayerImage.rows, bayerImage.cols, 0);
    for (int i = 0; i < bayerImage.rows; ++i) {
        for (int j = 0; j < bayerImage.cols; ++j) {
            if (1 == i % 2 && 1 == j % 2) {
                partialGreenImage.at<uchar>(i, j) = bayerImage.at<uchar>(i, j);
            }
        }
    }
    Mat partialRedImage = Mat::zeros(bayerImage.rows, bayerImage.cols, 0);
    for (int i = 0; i < bayerImage.rows; ++i) {
        for (int j = 0; j < bayerImage.cols; ++j) {
            if ((0 == i % 2 && 1 == j % 2) || (1 == i % 2 && 0 == j % 2)) {
                partialRedImage.at<uchar>(i, j) = bayerImage.at<uchar>(i, j);
            }
        }
    }

    // initialize three kernels
    double blueKernelData[9]  = {1.0/4, 1.0/2, 1.0/4, 1.0/2, 1.0, 1.0/2, 1.0/4, 1.0/2, 1.0/4};
    double greenKernelData[9] = {1.0/4, 1.0/2, 1.0/4, 1.0/2, 1.0, 1.0/2, 1.0/4, 1.0/2, 1.0/4};
    double redKernelData[9]   = {0, 1.0/4, 0, 1.0/4, 1.0, 1.0/4, 0, 1.0/4, 0};

    // generate kernel matrix
    Mat blueKernel  = Mat(3, 3, CV_64F, blueKernelData);
    Mat greenKernel = Mat(3, 3, CV_64F, greenKernelData);
    Mat redKernel   = Mat(3, 3, CV_64F, redKernelData);

    // init relative demosaic images
    Mat demosaicImage, blueImage, greenImage, redImage;
    blueImage  = Mat(bayerImage.rows, bayerImage.cols, 0);
    greenImage = Mat(bayerImage.rows, bayerImage.cols, 0);
    redImage   = Mat(bayerImage.rows, bayerImage.cols, 0);

    // apply kernel to corresponding channel
    filter2D(partialBlueImage,  blueImage,  0, blueKernel,  Point( -1, -1 ), 0, BORDER_DEFAULT);
    filter2D(partialGreenImage, greenImage, 0, greenKernel, Point( -1, -1 ), 0, BORDER_DEFAULT);
    filter2D(partialRedImage,   redImage,   0, redKernel,   Point( -1, -1 ), 0, BORDER_DEFAULT);

    partOne(originalImage, blueImage, greenImage, redImage);
    partTwo(originalImage, blueImage, greenImage, redImage);

    waitKey(0);

    return 0;
}
