#ifndef EVALUATE_H
#define EVALUATE_H

#include <pqxx/pqxx>
#include <iostream>

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


#endif