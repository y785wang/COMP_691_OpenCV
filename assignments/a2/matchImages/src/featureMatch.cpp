//
// Created by Yishi Wang on 2019-02-07.
//

#include "../headers/featureMatch.h"

using namespace cv;
using namespace std;


/**
 * Find matches for all features
 * @param features1 are all features in image one
 * @param features2 are all features in image two
 * @param dMatches are all feature matches between image one and imge two
 * @param resultImage is the output image
 */
void findMatch(vector<Feature> &features1, vector<Feature> &features2, vector<DMatch> &dMatches, Mat &resultImage) {
    // self test purpose
    float minThreshold = 10000;
    float maxThreshold = 0;

    // SSD threshold
    float threshold = 785; // for rotate_10°
//    float threshold = 1170; // for scaling_70%
//    float threshold = 850; // for rotate_contrast,
//    float threshold = 1000; // for cut, half;

    for (int i = 0; i < features1.size(); ++i) {
        float bestDiff = -1;
        float secondBestDiff = -1;
        int bestIndex = 0;
        int secondBestIndex = 0;
        for (int j = 0; j < features2.size(); ++j) {
            float diff = SSD(features1[i], features2[j]);
            // self test purpose
            if (0 != diff && diff < minThreshold) minThreshold = diff;
            if (diff > maxThreshold) maxThreshold = diff;

            if (-1 == bestDiff) {
                bestDiff = diff;
                bestIndex = j;
            } else if (diff < bestDiff) {
                secondBestDiff = bestDiff;
                bestDiff = diff;
                secondBestIndex = bestIndex;
                bestIndex = j;
            } else if (-1 == secondBestDiff || diff < secondBestDiff) {
                secondBestDiff = diff;
                secondBestIndex = j;
            }
        }

        float numerator = SSD(features1[i], features2[bestIndex]);
        float denominator = SSD(features1[i], features2[secondBestIndex]);
//        cout << numerator << ", " << denominator << endl;
        if (numerator > threshold || denominator > threshold) continue;

        float ratio;
        if (0 == denominator) {
            ratio = 0;
        } else {
            ratio = numerator / denominator;
        }
        DMatch dMatch;
        if (-1 != bestDiff) { // find at least one match
            if (-1 == secondBestDiff || ratio < 0.5) {
                dMatch = DMatch(i, bestIndex, 1);
                dMatches.push_back(dMatch);
            }
        }
    }
    cout << "threshold range: " << minThreshold << ", " << maxThreshold << endl;
}


/**
 *
 * @param feature1
 * @param feature2
 * @return
 */
float SSD(Feature &feature1, Feature &feature2) {
    Mat descriptor1 = feature1.getDescriptor();
    Mat descriptor2 = feature2.getDescriptor();
    Mat diff = descriptor1 - descriptor2;
    float sum = 0;
    for (int i = 0; i < diff.rows; ++i) {
        for (int j = 0; j < diff.cols; ++j) {
            sum += pow(diff.at<float>(i, j), 2); // SSD, sum of square difference
        }
    }
    return sum;
}