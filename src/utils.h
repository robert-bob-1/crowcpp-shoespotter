#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include "crow.h"

struct ImageResponse {
    cv::Mat image;
    int statusCode;
    std::string errorMessage;
};

// Define utils functions
void showMat(cv::Mat image) {
    // // Resize the image
    // int targetWidth = 800;
    // int targetHeight = 600;
    // cv::Size targetSize(targetWidth, targetHeight);
    // cv::resize(image, image, targetSize);

    // Set window position
    int posX = 100;
    int posY = 100;

    // Create named window and move it to position
    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::moveWindow("Display window", posX, posY);

    // Display the image
    cv::imshow("Display window", image);
    cv::waitKey(0);
    // cv::destroyWindow("Display window");
}

ImageResponse convertImageRequestToMat(const crow::request& req) {
    ImageResponse response;
    response.statusCode = 200; // Default status code

    // Check if the request contains a file
    if (req.body.empty()) {
        response.statusCode = 400;
        response.errorMessage = "No file uploaded";
        return response;
    }

    // Find the position of the start of the image data
    size_t imageDataStart = req.body.find("\r\n\r\n") + 4;
    if (imageDataStart == std::string::npos) {
        response.statusCode = 400;
        response.errorMessage = "Invalid image data";
        return response;
    }

    // Extract image data from body
    std::string imageData = req.body.substr(imageDataStart);
    if (imageData.empty()) {
        response.statusCode = 400;
        response.errorMessage = "Invalid image data after extraction";
        return response;
    }
    // Convert the image data from string to vector of bytes
    std::vector<uchar> imageDataVec(imageData.begin(), imageData.end());

    // Decode the image data using OpenCV
    response.image = imdecode(imageDataVec, cv::IMREAD_COLOR);

    // Check if the image was successfully decoded
    if (response.image.empty()) {
        response.statusCode = 400;
        response.errorMessage = "Failed to decode image data";
    }

    return response;
}

struct ImageAndIdResponse {
    cv::Mat image;
    int id;
    int statusCode;
    std::string errorMessage;
};

ImageAndIdResponse convertImageAndIdRequestToMat(const crow::request& req) {
    ImageAndIdResponse response;
    response.statusCode = 200; // Default status code

    // Check if the request contains a file
    if (req.body.empty()) {
        response.statusCode = 400;
        response.errorMessage = "No file uploaded";
        return response;
    }

    // Find the position of the start of the image data
    size_t imageDataStart = req.body.find("\r\n\r\n") + 4;
    if (imageDataStart == std::string::npos) {
        response.statusCode = 400;
        response.errorMessage = "Invalid image data";
        return response;
    }

    CROW_LOG_INFO << "before substr";


    // Extract image data from body
    std::string imageData = req.body.substr(imageDataStart);
    if (imageData.empty()) {
        response.statusCode = 400;
        response.errorMessage = "Invalid image data after extraction";
        return response;
    }

    CROW_LOG_INFO << "before string to vector of bytes";

    // Convert the image data from string to vector of bytes
    std::vector<uchar> imageDataVec(imageData.begin(), imageData.end());

    // Decode the image data using OpenCV
    response.image = imdecode(imageDataVec, cv::IMREAD_COLOR);

    // Check if the image was successfully decoded
    if (response.image.empty()) {
        response.statusCode = 400;
        response.errorMessage = "Failed to decode image data";
    }

    CROW_LOG_INFO << "before idString";

    // Extract the id from the request
    std::string idString = req.url_params.get("id");
    CROW_LOG_INFO << "idString: " << idString;
    if (idString.empty()) {
        response.statusCode = 400;
        response.errorMessage = "No id provided";
        return response;
    }

    response.id = std::stoi(idString);

    return response;
}

bool isImageBlurry(cv::Mat image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat laplacianImage;
    cv::Laplacian(grayImage, laplacianImage, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImage, mean, stddev);

    double focusMeasure = stddev.val[0] * stddev.val[0];
    CROW_LOG_INFO << "Focus measure: " << focusMeasure;

    return focusMeasure < 100;
}

#endif