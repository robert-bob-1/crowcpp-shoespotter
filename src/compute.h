#ifndef COMPUTE_H
#define COMPUTE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include "utils.h"

// Structures for computed shoe properties
struct ShoeColor {
    float red;
    float green;
    float blue;
};

// Method to get shoe metadata
ShoeColor computeShoeColorRGB(cv::Mat image) {
    ShoeColor shoeColor;
    shoeColor.red = 0.0;
    shoeColor.green = 0.0;
    shoeColor.blue = 0.0;

    for (int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
            // if pixel is black skip it as it's not part of the shoe, its just the background
            if (image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) {
                continue;
            }
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            shoeColor.red += pixel[2];
            shoeColor.green += pixel[1];
            shoeColor.blue += pixel[0];
        }
    }

    // Compute percentages of color
    double totalShoeColorValues = shoeColor.red + shoeColor.green + shoeColor.blue;

    if (totalShoeColorValues < 0) {
        throw std::runtime_error("No shoe colors were detected.");
    }

    shoeColor.red = (shoeColor.red / totalShoeColorValues) * 100;
    shoeColor.green = (shoeColor.green / totalShoeColorValues) * 100;
    shoeColor.blue = (shoeColor.blue / totalShoeColorValues) * 100;

    std::cout << "Shoe color: red " << shoeColor.red << "%, green " << shoeColor.green << "%, blue" << shoeColor.blue << "%." << std::endl;

    return shoeColor;
}

// Structure for dominant color
struct DominantColor {
    cv::Vec3b color;
    float percentage;

    bool operator==(const DominantColor& other) const {
        return color == other.color && percentage == other.percentage;
    }
};

std::vector<DominantColor> computeDominantColors(cv::Mat image, int k = 4) {
    // std::vector<cv::Vec3b> dominantColors(k);
    cv::Mat resizedImage;
    image.convertTo(resizedImage, CV_32F);
    resizedImage = resizedImage.reshape(1, resizedImage.total());

    cv::Mat labels, centers;
    cv::kmeans(resizedImage, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    int totalPixels = labels.rows;
    std::vector<int> totalPixelsPerCentroid(k, 0);
    for (int i = 0; i < labels.rows; ++i) {
        totalPixelsPerCentroid[labels.at<int>(i)]++;
    }

    std::vector<DominantColor> dominantColors(k);
    for (int i = 0; i < k; ++i) {
        dominantColors[i].color = centers.at<cv::Vec3f>(i);
        std::cout << "totalPixelsPerCentroid[" << i << "]: " << totalPixelsPerCentroid[i] << " total pixels " << totalPixels << std::endl;
        dominantColors[i].percentage = (1.0f * totalPixelsPerCentroid[i] / totalPixels) * 100;
        std::cout << "Color " << i << ": " << dominantColors[i].color << " percentage: " << dominantColors[i].percentage << std::endl;
    }

    // display the colors
    cv::Mat colorSwatch(100, 100 * k, CV_8UC3);
    for (int i = 0; i < k; ++i) {
        colorSwatch.colRange(i * 100, (i + 1) * 100) = dominantColors[i].color;
        std::cout << "Color " << i << ": " << dominantColors[i].color << " percentage: " << dominantColors[i].percentage << std::endl;
    }

    cv::imshow("Dominant Colors", colorSwatch);
    cv::waitKey(0);

    return dominantColors;
}

std::vector<cv::Mat> computeRGBHistograms(cv::Mat image) {
    std::vector<cv::Mat> histograms;
    std::vector<cv::Mat> bgrChannels;
    cv::split(image, bgrChannels);

    // Mask to ignore white background - adapt to background color
    cv::Mat mask;
    cv::inRange(image, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
    // Invert the mask so only the shoe is highlighted, not the background
    cv::bitwise_not(mask, mask);
    // cv::imshow("Mask", mask);
    // cv::waitKey(0);

    for (int i = 0; i < 3; i++) {
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};

        cv::calcHist(&bgrChannels[i], 1, 0, mask, hist, 1, &histSize, &histRange);

        // display histogram for testing purposes
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound((double)hist_w/histSize);
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        // // Normalize the histogram
        cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, CV_32F);
        std::cout << "Histogram size: " << hist.size() << std::endl;
        std::cout << "Mat type: " << hist.type() << std::endl;
        // // Draw the histogram
        // for (int j = 0; j < histSize; j++) {
        //     cv::line(histImage,
        //         cv::Point(j*bin_w, hist_h - cvRound(hist.at<float>(j))),
        //         cv::Point((j+1)*bin_w, hist_h - cvRound(hist.at<float>(j+1))),
        //         cv::Scalar(255, 0, 0), 2, 8, 0);
        // }
        // // Display the histogram image
        // cv::imshow("Histogram " + std::to_string(i), histImage);
        // cv::waitKey(0);

        histograms.push_back(hist);
    }

    return histograms;
}

// Function to compute the LBP histogram
cv::Mat computeLBPHistogram(cv::Mat image, int numPatterns = 256) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    std::cout << "check if image was not modified accidentally" << std::endl;
    std::cout << "image channels should be 3: " << image.channels() << std::endl;

    cv::Mat lbpImage = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            unsigned char code = 0;

            code |= (gray.at<uchar>(i-1, j-1) > center) << 7;
            code |= (gray.at<uchar>(i-1, j)   > center) << 6;
            code |= (gray.at<uchar>(i-1, j+1) > center) << 5;
            code |= (gray.at<uchar>(i,   j+1) > center) << 4;
            code |= (gray.at<uchar>(i+1, j+1) > center) << 3;
            code |= (gray.at<uchar>(i+1, j)   > center) << 2;
            code |= (gray.at<uchar>(i+1, j-1) > center) << 1;
            code |= (gray.at<uchar>(i,   j-1) > center) << 0;

            lbpImage.at<uchar>(i, j) = code;
        }
    }

    // Set the number of bins to the number of patterns (usually 256)
    int histSize = numPatterns;
    float range[] = { 0, (float)numPatterns };
    const float* histRange = { range };

    cv::Mat hist;
    cv::calcHist(&lbpImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Normalize the histogram
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    std::cout << "LBP histogram size: " << hist.size() << std::endl;
    std::cout << "Mat type: " << hist.type() << std::endl;

    // // Test display the lbpimage
    // cv::imshow("LBP Image", lbpImage);
    // cv::waitKey(0);

    return hist;
}

cv::Mat computeHOGFeatures(cv::Mat image) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Set parameters for HOG descriptors
    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(64, 128);  // Use a fixed window size
    hog.blockSize = cv::Size(16, 16);
    hog.blockStride = cv::Size(8, 8);
    hog.cellSize = cv::Size(8, 8);
    hog.nbins = 9;

    // Resize the image to match the HOG window size
    cv::resize(grayImage, grayImage, hog.winSize);

    std::vector<float> descriptors;
    hog.compute(grayImage, descriptors);

    // Convert descriptors to Mat and normalize
    cv::Mat hogFeatures(descriptors, true);
    hogFeatures = hogFeatures.reshape(1, 1);  // Make it a single row matrix
    cv::normalize(hogFeatures, hogFeatures, 0, 1, cv::NORM_MINMAX);

    std::cout << "HOG features size: " << hogFeatures.size() << std::endl;
    std::cout << "Mat type: " << hogFeatures.type() << std::endl;

    return hogFeatures;
}

struct ShoeProperties {
    std::vector<cv::Mat> rgbHistograms;
    cv::Mat lbpHistogram;
    cv::Mat hogFeatures;
};

ShoeProperties computeShoeFeatures(cv::Mat image) {
    ShoeProperties shoeFeatures;
    shoeFeatures.rgbHistograms = computeRGBHistograms(image);
    shoeFeatures.lbpHistogram = computeLBPHistogram(image);
    shoeFeatures.hogFeatures = computeHOGFeatures(image);

    return shoeFeatures;
}

double computeDistance(const cv::Mat& mat1, const cv::Mat& mat2) {
    return cv::norm(mat1, mat2, cv::NORM_L2);
}

double computeCosineSimilarity(const cv::Mat& mat1, const cv::Mat& mat2) {
    double dotProduct = mat1.dot(mat2);
    double norm1 = cv::norm(mat1);
    double norm2 = cv::norm(mat2);
    return dotProduct / (norm1 * norm2);
}

    // Test view gray image
    // cv::imshow("Gray Image", grayImage);
    // cv::waitKey(0);

    // // Set parameters for hog descriptors
    // cv::HOGDescriptor hog;
    // hog.winSize = cv::Size(128, 128);
    // hog.blockSize = cv::Size(16, 16);
    // hog.blockStride = cv::Size(8, 8);
    // hog.cellSize = cv::Size(8, 8);
    // hog.nbins = 9;

    // std::vector<float> descriptors;
    // hog.compute(grayImage, descriptors);

    // // Normalize Hog vector to make it more robust to lighting variation
    // cv::Mat hogFeatures(descriptors);
    // hogFeatures = hogFeatures.reshape(1, 1);
    // cv::normalize(hogFeatures, hogFeatures);

    // // Calculate gradients gx, gy
    // Mat gx, gy;
    // Sobel(grayImage, gx, CV_32F, 1, 0, 1);
    // Sobel(grayImage, gy, CV_32F, 0, 1, 1);

    // // C++ Calculate gradient magnitude and direction (in degrees)
    // Mat mag, angle;
    // cartToPolar(gx, gy, mag, angle, 1);

    // visualizeHOG(image, hog, descriptors);

    // HOG parameters (matching your Python code)
#endif