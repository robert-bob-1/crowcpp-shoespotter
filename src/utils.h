#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <pqxx/pqxx>
#include <sstream>
#include "crow.h"
#include "crow/middlewares/cors.h"

struct ImageResponse {
    cv::Mat image;
    int statusCode;
    std::string errorMessage;
};

// Simply display a single image in a window
void showMat(cv::Mat image, std::string windowName = "Display window") {
    // Set window position
    int posX = 100;
    int posY = 100;

    // Create named window and move it to position
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::moveWindow(windowName, posX, posY);

    // Display the image
    cv::imshow(windowName, image);
    cv::waitKey(0);
    // cv::destroyWindow("Display window");
}

// Display multiple Mat images in a matrix style in a single window
void showMats(std::vector<cv::Mat> images, std::string windowName = "Display window", int sizeOfImage = 200) {
    // Set window position
    int posX = 100;
    int posY = 100;

    // Create named window and move it to position
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::moveWindow(windowName, posX, posY);

    // Calculate the matrix size
    int matrixSize = ceil(sqrt(images.size()));
    // Set window size
    int windowWidth = matrixSize * sizeOfImage;
    int windowHeight = matrixSize * sizeOfImage;

    cv::Mat matrixImage = cv::Mat::zeros(windowWidth, windowHeight, CV_8UC3);

    // Add images to the matrix
    int imageIndex = 0;
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            if (imageIndex >= images.size()) {
                break;
            }

            cv::Mat image = images[imageIndex];
            cv::resize(image, image, cv::Size(sizeOfImage, sizeOfImage));

            cv::Rect currentImageInsertWindow(j * sizeOfImage, i * sizeOfImage, sizeOfImage, sizeOfImage);
            image.copyTo(matrixImage(currentImageInsertWindow));

            imageIndex++;
        }
    }

    // Display the matrix image
    cv::imshow(windowName, matrixImage);
    cv::waitKey(0);
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

struct ImagesResponse {
    std::vector<cv::Mat> images;
    int statusCode;
    std::string errorMessage;
};

ImagesResponse convertImagesRequestToMat(const crow::request& req) {
    ImagesResponse response;
    response.statusCode = 200; // Default status code

    // Check if the request contains a file
    if (req.body.empty()) {
        response.statusCode = 400;
        response.errorMessage = "No file uploaded";
        return response;
    }

    // Find the boundary string from the request headers
    auto contentTypeIter = req.headers.find("content-type");
    if (contentTypeIter == req.headers.end() ||
        contentTypeIter->second.find("multipart/form-data; boundary=") == std::string::npos) {
        response.statusCode = 400;
        response.errorMessage = "Invalid Content-Type header";
        return response;
    }

    // 30 is the length of "multipart/form-data; boundary="
    std::string boundary = "--" + contentTypeIter->second.substr(30);
    CROW_LOG_INFO << "Boundary: " << boundary;
    // Print req.body to a file for logging/debugging
    std::ofstream file("req_body.txt");
    file << req.body;
    file.close();

    // Find image data blocks using boundary
    size_t imageStart = 0;
    size_t imageEnd = req.body.find(boundary, imageStart);

    while (imageEnd != std::string::npos) {
        imageStart = req.body.find("\r\n\r\n", imageEnd) + 4;

        if (imageStart == std::string::npos) {
            response.statusCode = 400;
            response.errorMessage = "Invalid image data format";
            return response;
        }

        imageEnd = req.body.find(boundary, imageStart);

        // Handle the last part of the body
        if (imageEnd == std::string::npos) {
            imageEnd = req.body.size();
        }

        size_t imageDataSize = imageEnd - imageStart;

        // Create a Mat header for the image data
        cv::Mat imageDataMat = cv::Mat(1, imageDataSize, CV_8UC1, (void*)(req.body.data() + imageStart));

        // Decode the image
        cv::Mat image = cv::imdecode(imageDataMat, cv::IMREAD_COLOR);

        // Check if decoding was successful
        if (image.empty()) {
            response.statusCode = 400;
            response.errorMessage = "Image decoding failed";
            return response;
        }

        response.images.push_back(image);
        imageStart = imageEnd + boundary.size(); // Move to the next boundary
    }

    return response;
}


using ShoeClassification = std::pair<std::string, double>;

struct ImageAndClassification {
    cv::Mat image;
    std::vector<ShoeClassification> classificationData;
};

std::vector<std::string> splitString(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

ImageAndClassification convertRequestToImageAndClassification(const crow::request& req) {
    ImageAndClassification imageAndClassification;

    crow::multipart::message multiPartMessage(req);

    std::ofstream outputFile("multiPartMessage.txt");
    outputFile << multiPartMessage.dump() << std::endl;
    outputFile.close();

    // iterate through part_map
    auto iterator = multiPartMessage.part_map.begin();

    while (iterator != multiPartMessage.part_map.end()) {
        // ignore first iterator
        if (iterator == multiPartMessage.part_map.begin()) {
            iterator++;
            continue;
        }
        std::cout << "Part name: " << iterator->first << std::endl;
        std::cout << "Part body: " << iterator->second.body << std::endl;
        std::cout << "Part headers: " << iterator->second.headers.size() << std::endl;
        iterator++;
    }

    // Get image part from body
    auto imagePart = multiPartMessage.part_map.find("file");
    if (imagePart == multiPartMessage.part_map.end()) {
        std::cerr << "file part not found" << std::endl;
        return imageAndClassification;
    } else if (imagePart->second.body.empty()) {
        std::cerr << "file body is empty" << std::endl;
        return imageAndClassification;
    }

    std::cout << "Image part found" << std::endl;

    // Decode the image from the crow::multipart::mp_map::iterator
    // Convert to a byte stream and then decode using cv imdecode
    std::vector<uchar> imageData(imagePart->second.body.begin(), imagePart->second.body.end());
    imageAndClassification.image = cv::imdecode(imageData, cv::IMREAD_COLOR);
    // Test print
    // showMat(imageAndClassification.image);

    // Get classification data part from body
    auto classificationDataPart = multiPartMessage.part_map.find("classification_data");
    if (classificationDataPart == multiPartMessage.part_map.end()) {
        std::cerr << "classification_data part not found" << std::endl;
        return imageAndClassification;
    } else if (classificationDataPart->second.body.empty()) {
        std::cerr << "classification_data body is empty" << std::endl;
        return imageAndClassification;
    }

    std::string jsonString = classificationDataPart->second.body;

    // std::cout << "Classification data part found: " << jsonString << std::endl;

    // Initialize the classification data vector
    imageAndClassification.classificationData = std::vector<ShoeClassification>();

    // Parse the classification data JSON
    crow::json::type stringType = crow::json::type::String;
    crow::json::rvalue jsonValue = crow::json::load(jsonString);

    // Convert JSON data to vector of pairs
    // Construct a vector with all types and confidence scores and then sort it
    std::vector<std::pair<std::string, double>> unsortedClassificationData;

    for (const auto& key : jsonValue.keys()) {
        std::string keyString = key;
        double value = jsonValue[key].d();
        // Ignore values less than
        if (value > 0.10)
            unsortedClassificationData.push_back(std::make_pair(keyString, value));
    }

    // Sort the classification data by confidence score
    // Use the default sort function  and declare a lambda function to compare the pairs
    std::sort(unsortedClassificationData.begin(), unsortedClassificationData.end(),
        [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
            return a.second > b.second;
        });

    imageAndClassification.classificationData = unsortedClassificationData;

    // Test print classificationData
    for (const auto& [key, value] : imageAndClassification.classificationData) {
        std::cout << "Key: " << key << " Value: " << value << std::endl;
    }

    return imageAndClassification;
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


cv::Mat preprocessImages(cv::Mat image) {
    // Resize the image
    int targetWidth = 128;
    int targetHeight = 128;
    cv::Size targetSize(targetWidth, targetHeight);
    cv::resize(image, image, targetSize);

    return image;
}

cv::Mat binarystringToMat(pqxx::binarystring binary, int rows = -1, int cols = -1, int type = CV_32F) {
    // in case rows and cols are not specified we assume we are dealing with a rgb histogram which has the following sizes
    if (rows == -1) {
        rows = binary.size() / sizeof(float);
    }
    if (cols == -1) {
        cols = 1;
    }
    cv::Mat hogDescriptor = cv::Mat(
        rows,
        cols,
        type,
        (void*)binary.data()
    );

}

#endif