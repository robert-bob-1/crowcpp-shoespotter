#include "crow.h"
#include <opencv2/opencv.hpp>

#include "database.h"
#include "evaluate.h"
#include "utils.h"

using namespace cv;



int main()
{
    crow::SimpleApp app; //define your crow application
    //set logging
    crow::logger::setLogLevel(crow::LogLevel::INFO);

    //define your endpoint at the root directory
    CROW_ROUTE(app, "/")([](){
        CROW_LOG_INFO << "running / route";
        Mat image = imread("../img.webp", IMREAD_COLOR);
        if (image.empty()) {
            printf("Could not open or find the image\n");
            return "unable to open image";
        }
        CROW_LOG_INFO << "read image";

        // Open the image asynchronously
        std::thread displayThread([image]() {
            namedWindow("Display window", WINDOW_AUTOSIZE);
            imshow("Display window", image);
            waitKey(0);
            destroyWindow("Display window");
        });
        displayThread.detach();
        return "";
    });

    CROW_ROUTE(app, "/upload")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            CROW_LOG_DEBUG << "upload method has received body: \n" << req.body.data();

            ImageResponse imageResponse = convertImageRequestToMat(req);
            Mat image = imageResponse.image;
            if (image.empty()) {
                return crow::response(imageResponse.statusCode, imageResponse.errorMessage);
            }

            if (!isImageBlurry(image)) {
                showMat(image);
            }

            return crow::response("Image uploaded and processed successfully");
    });

    // POST
    // Method to compute image properties and save them to db
    // Input: segmented image with a shoe, and a black background
    // Effects: computes shoe properties and checks against those in database
    CROW_ROUTE(app, "/compute-properties-and-save")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            CROW_LOG_INFO << "evaluate method has received body: \n" << req.body.data();

            ImageAndIdResponse imageAndIdResponse = convertImageAndIdRequestToMat(req);

            int id = imageAndIdResponse.id;
            Mat image = imageAndIdResponse.image;
            CROW_LOG_INFO << "ID: " << id;
            if (image.empty()) {
                CROW_LOG_INFO << "Image is empty";
                return crow::response(imageAndIdResponse.statusCode, imageAndIdResponse.errorMessage);
            }
            showMat(image);

            // Compute shoe properties
            ShoeColor shoeColors = computeShoeColorRGB(image);

            // Save shoe properties to database using libpqxx
            try {
                saveShoeProperties(id, shoeColors);
            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
                return crow::response(500, e.what());
            }

            return crow::response("Shoe properties computed successfully");
    });

    // POST
    // Method to compute image parameters
    // Input: segmented image with a shoe, and a black background
    // Effects: computes shoe properties and checks against those in database
    CROW_ROUTE(app, "/evaluate")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            CROW_LOG_INFO << "evaluate method has received body: \n" << req.body.data();

            ImageResponse imageResponse = convertImageRequestToMat(req);
            Mat image = imageResponse.image;
            if (image.empty()) {
                CROW_LOG_INFO << "Image is empty";
                return crow::response(imageResponse.statusCode, imageResponse.errorMessage);
            }

            showMat(image);

            // Compute shoe properties
            ShoeColor shoeColors = computeShoeColorRGB(image);

            return crow::response("Shoe properties computed successfully");
    });

    CROW_ROUTE(app, "/test-db")
        .methods(crow::HTTPMethod::Get)([](){ // Capture the 'conn' variable in the lambda's capture list
            try {
                testGetShoeMetadata();
            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
            }

            return "Test route";
    });

    //set the port, set the app to run on multiple threads, and run the app
    app.bindaddr("127.0.0.1").port(8081).multithreaded().run();
}