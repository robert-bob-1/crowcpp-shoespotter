#include "crow.h"
#include <opencv2/opencv.hpp>

#include "compare.h"
#include "compute.h"
#include "database_features.h"
#include "database_shoes.h"
#include "evaluate.h"
#include "service.h"
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
            CROW_LOG_INFO << "Image size: " << image.size();
            CROW_LOG_INFO << "ID: " << id;
            if (image.empty()) {
                CROW_LOG_INFO << "Image is empty";
                return crow::response(imageAndIdResponse.statusCode, imageAndIdResponse.errorMessage);
            }

            // Compute shoe properties and save them to database
            try {
                cv::Mat resizedImage = preprocessImages(image);
                // showMat(resizedImage);

                std::vector<cv::Mat> RGBHistograms = computeRGBHistograms(resizedImage);
                cv::Mat lbpHistogram = computeLBPHistogram(resizedImage);
                cv::Mat hogDescriptor = computeHOGFeatures(resizedImage);

                saveShoeProperties(id, RGBHistograms, lbpHistogram, hogDescriptor);
            } catch (const std::exception &e) {
                return crow::response(500, e.what());
            }

            return crow::response("Shoe properties computed successfully");
    });

    // POST
    // Method to compute image properties and save them to db
    // Input: segmented image with a shoe, and a black background
    // Effects: computes shoe properties and checks against those in database
    CROW_ROUTE(app, "/compare-shoe-images")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            CROW_LOG_INFO << "evaluate method has received body: \n" << req.body.data();

            ImagesResponse imagesResponse = convertImagesRequestToMat(req);

            std::vector<Mat> images = imagesResponse.images;
            if (images.empty()) {
                CROW_LOG_INFO << "Images are empty";
                return crow::response(imagesResponse.statusCode, imagesResponse.errorMessage);
            }

            // stretch them to a fixed size 656x656
            for (int i = 0; i < images.size(); i++) {
                images[i] = preprocessImages(images[i]);
                std::cout << "image" << i << "size after preprocessing" << images[i].size() << std::endl;
            }


            // showMat(images[0]);
            // showMat(images[1]);

            // Compute shoe properties
            std::vector<std::vector<cv::Mat>> histograms;
            std::vector<cv::Mat> lbpHistograms;
            std::vector<cv::Mat> hogDescriptors;
            for (int i = 0; i < images.size(); i++) {
                auto histogramsForImage = computeRGBHistograms(images[i]);
                histograms.push_back(histogramsForImage);
                std::cout << "Computed histograms for image " << i << std::endl;

                auto lbpHistogramsForImage = computeLBPHistogram(images[i]);
                lbpHistograms.push_back(lbpHistogramsForImage);
                std::cout << "Computed LBP histograms for image " << i << std::endl;

                auto hogDescriptorForImage = computeHOGFeatures(images[i]);
                hogDescriptors.push_back(hogDescriptorForImage);
                std::cout << "Computed HOG descriptor for image " << i << std::endl;
            }

            // Compare first image and find most similar image from other images
            int mostSimilarImage = -1;
            std::cout << "Comparing image 0 with other images " << images.size() << std::endl;
            std::cout << "images.size() " << images.size() << std::endl;
            // Calculate similarity for each color channel
            for (int i = 1; i < images.size(); ++i) {
                std::cout << "Comparing image " << i << " with image 0" << std::endl;
                // Comparing color histogram
                for (int channel = 0; channel < 3; channel++) {
                    double correlation = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_CORREL);
                    double chiSquareDistance = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_CHISQR);
                    double intersection = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_INTERSECT);

                    std::cout << "Channel " << channel << " Similarity:" << std::endl;
                    std::cout << "  Correlation: " << correlation << std::endl;
                    std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                    std::cout << "  Intersection: " << intersection << std::endl;
                }

                // Comparing lbp
                double correlation = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_CORREL);
                double chiSquareDistance = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_CHISQR);
                double intersection = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_INTERSECT);

                std::cout << "LBP Similarity:" << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << "  Intersection: " << intersection << std::endl;

                std::cout << std::endl;

                // std::cout << "HOG Similarity:" << std::endl;
                // std::cout << "  Correlation: " << correlationHOG << std::endl;
                // std::cout << "  Chi-Square Distance: " << chiSquareDistanceHOG << std::endl;
                // std::cout << "  Intersection: " << intersectionHOG << std::endl;

                double distance = computeDistance(hogDescriptors[0], hogDescriptors[i]);
                double similarity = computeCosineSimilarity(hogDescriptors[0], hogDescriptors[i]);
                correlation = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_CORREL);
                chiSquareDistance = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_CHISQR);
                intersection = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_INTERSECT);

                std::cout << "HOG Similarity:" << std::endl;
                std::cout << "  Distance: " << distance << std::endl;
                std::cout << "  Cosine similarity: " << similarity << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << "  Intersection: " << intersection << std::endl;

                std::cout << std::endl;
            }


            return crow::response("Shoe properties computed successfully");
    });

    // GET
    // Test method to get hog descriptors by id and test their similarity
    CROW_ROUTE(app, "/test-hog-similarity")
        .methods(crow::HTTPMethod::Get)([](){
            try {
                int img1Id = 202;
                int img2Id = img1Id + 1;
                std::vector<cv::Mat> rgbHistograms1 = getRGBHistogramsByShoeImageId(img1Id);
                cv::Mat lbpHistogram1 = getLBPFeaturesByShoeImageId(img1Id);
                cv::Mat hogDescriptor1 = getHOGFeaturesByShoeImageId(img1Id);
                // check dimensions of matrices
                std::cout << "RGB Histograms size: " << rgbHistograms1[0].size() << std::endl;
                std::cout << "LBP Histogram size: " << lbpHistogram1.size() << std::endl;
                std::cout << "HOG Descriptor size: " << hogDescriptor1.size() << std::endl;

                std::vector<cv::Mat> rgbHistograms2 = getRGBHistogramsByShoeImageId(img2Id);
                cv::Mat lbpHistogram2 = getLBPFeaturesByShoeImageId(img2Id);
                cv::Mat hogDescriptor2 = getHOGFeaturesByShoeImageId(img2Id);

                // Compare shoe properties
                for (int channel = 0; channel < 3; channel++) {
                    double correlation = cv::compareHist(rgbHistograms1[channel], rgbHistograms2[channel], cv::HISTCMP_CORREL);
                    double chiSquareDistance = cv::compareHist(rgbHistograms1[channel], rgbHistograms2[channel], cv::HISTCMP_CHISQR);
                    double intersection = cv::compareHist(rgbHistograms1[channel], rgbHistograms2[channel], cv::HISTCMP_INTERSECT);

                    std::cout << "Channel " << channel << " Similarity:" << std::endl;
                    std::cout << "  Correlation: " << correlation << std::endl;
                    std::cout << "  Intersection: " << intersection << std::endl;
                    std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                }

                double correlation = cv::compareHist(lbpHistogram1, lbpHistogram2, cv::HISTCMP_CORREL);
                double chiSquareDistance = cv::compareHist(lbpHistogram1, lbpHistogram2, cv::HISTCMP_CHISQR);

                std::cout << "LBP Similarity:" << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << std::endl;

                correlation = cv::compareHist(hogDescriptor1, hogDescriptor2, cv::HISTCMP_CORREL);
                chiSquareDistance = cv::compareHist(hogDescriptor1, hogDescriptor2, cv::HISTCMP_CHISQR);

                std::cout << "HOG Similarity:" << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << std::endl;


            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
            }

            return "Test route";
    });

    // POST
    // Method to test saving and retrieving shoe properties from database
    CROW_ROUTE(app, "/test-save-retrieve")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            int image1Id = 196;
            int image2Id = image1Id + 1;
            CROW_LOG_INFO << "evaluate method has received body: \n" << req.body.data();

            ImagesResponse imagesResponse = convertImagesRequestToMat(req);

            std::vector<Mat> images = imagesResponse.images;
            if (images.empty()) {
                CROW_LOG_INFO << "Images are empty";
                return crow::response(imagesResponse.statusCode, imagesResponse.errorMessage);
            }

            // stretch them to a fixed size 656x656
            for (int i = 0; i < images.size(); i++) {
                images[i] = preprocessImages(images[i]);
                std::cout << "image" << i << "size after preprocessing" << images[i].size() << std::endl;
            }

            // Compute shoe properties
            std::vector<std::vector<cv::Mat>> histograms;
            std::vector<cv::Mat> lbpHistograms;
            std::vector<cv::Mat> hogDescriptors;
            for (int i = 0; i < images.size(); i++) {
                auto histogramsForImage = computeRGBHistograms(images[i]);
                histograms.push_back(histogramsForImage);
                std::cout << "Computed histograms for image " << i << std::endl;
                if (i == 0) saveColorHistograms(image1Id, histogramsForImage);
                else saveColorHistograms(image2Id, histogramsForImage);

                auto lbpHistogramsForImage = computeLBPHistogram(images[i]);
                lbpHistograms.push_back(lbpHistogramsForImage);
                std::cout << "Computed LBP histograms for image " << i << std::endl;
                if (i == 0) saveLBPFeatures(image1Id, lbpHistogramsForImage);
                else saveLBPFeatures(image2Id, lbpHistogramsForImage);

                auto hogDescriptorForImage = computeHOGFeatures(images[i]);
                hogDescriptors.push_back(hogDescriptorForImage);
                std::cout << "Computed HOG descriptor for image " << i << std::endl;
                if (i == 0) saveHOGFeatures(image1Id, hogDescriptorForImage);
                else saveHOGFeatures(image2Id, hogDescriptorForImage);
            }

            // Extract saved images properties
            std::vector<cv::Mat> savedRGBHistograms1 = getRGBHistogramsByShoeImageId(image1Id);
            cv::Mat savedLBPFeatures1 = getLBPFeaturesByShoeImageId(image1Id);
            cv::Mat savedHOGFeatures1 = getHOGFeaturesByShoeImageId(image1Id);

            std::vector<cv::Mat> savedRGBHistograms2 = getRGBHistogramsByShoeImageId(image2Id);
            cv::Mat savedLBPFeatures2 = getLBPFeaturesByShoeImageId(image2Id);
            cv::Mat savedHOGFeatures2 = getHOGFeaturesByShoeImageId(image2Id);

            // Compare first image and find most similar image from other images
            int mostSimilarImage = -1;
            std::cout << "Comparing image 0 with other images " << images.size() << std::endl;
            std::cout << "images.size() " << images.size() << std::endl;
            // Calculate similarity for each color channel
            for (int i = 1; i < images.size(); ++i) {
                std::cout << "Comparing image " << i << " with image 0" << std::endl;
                // Comparing color histogram
                for (int channel = 0; channel < 3; channel++) {
                    double correlation = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_CORREL);
                    double chiSquareDistance = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_CHISQR);
                    double intersection = cv::compareHist(histograms[0][channel], histograms[i][channel], cv::HISTCMP_INTERSECT);

                    std::cout << "Channel " << channel << " Similarity:" << std::endl;
                    std::cout << "  Correlation: " << correlation << std::endl;
                    std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                    std::cout << "  Intersection: " << intersection << std::endl;
                }

                // Comparing lbp
                double correlation = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_CORREL);
                double chiSquareDistance = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_CHISQR);
                double intersection = cv::compareHist(lbpHistograms[0], lbpHistograms[i], cv::HISTCMP_INTERSECT);

                std::cout << "LBP Similarity:" << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << "  Intersection: " << intersection << std::endl;

                std::cout << std::endl;

                double distance = computeDistance(hogDescriptors[0], hogDescriptors[i]);
                double similarity = computeCosineSimilarity(hogDescriptors[0], hogDescriptors[i]);
                correlation = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_CORREL);
                chiSquareDistance = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_CHISQR);
                intersection = cv::compareHist(hogDescriptors[0], hogDescriptors[i], cv::HISTCMP_INTERSECT);

                std::cout << "HOG Similarity:" << std::endl;
                std::cout << "  Distance: " << distance << std::endl;
                std::cout << "  Cosine similarity: " << similarity << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
                std::cout << "  Intersection: " << intersection << std::endl;

                std::cout << std::endl;
            }

            // show comparisons for saved images
            for (int channel = 0; channel < 3; channel++) {
                double correlation = cv::compareHist(savedRGBHistograms1[channel], savedRGBHistograms2[channel], cv::HISTCMP_CORREL);
                double chiSquareDistance = cv::compareHist(savedRGBHistograms1[channel], savedRGBHistograms2[channel], cv::HISTCMP_CHISQR);
                double intersection = cv::compareHist(savedRGBHistograms1[channel], savedRGBHistograms2[channel], cv::HISTCMP_INTERSECT);

                std::cout << "Channel " << channel << " Similarity:" << std::endl;
                std::cout << "  Correlation: " << correlation << std::endl;
                std::cout << "  Intersection: " << intersection << std::endl;
                std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
            }

            double correlation = cv::compareHist(savedLBPFeatures1, savedLBPFeatures2, cv::HISTCMP_CORREL);
            double chiSquareDistance = cv::compareHist(savedLBPFeatures1, savedLBPFeatures2, cv::HISTCMP_CHISQR);
            double intersection = cv::compareHist(savedLBPFeatures1, savedLBPFeatures2, cv::HISTCMP_INTERSECT);

            std::cout << "LBP Similarity:" << std::endl;
            std::cout << "  Correlation: " << correlation << std::endl;
            std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;

            correlation = cv::compareHist(savedHOGFeatures1, savedHOGFeatures2, cv::HISTCMP_CORREL);
            chiSquareDistance = cv::compareHist(savedHOGFeatures1, savedHOGFeatures2, cv::HISTCMP_CHISQR);
            intersection = cv::compareHist(savedHOGFeatures1, savedHOGFeatures2, cv::HISTCMP_INTERSECT);

            std::cout << "HOG Similarity:" << std::endl;
            std::cout << "  Correlation: " << correlation << std::endl;
            std::cout << "  Chi-Square Distance: " << chiSquareDistance << std::endl;
            std::cout << "  Intersection: " << intersection << std::endl;


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

            // Preprocess shoe image
            cv::Mat resizedImage = preprocessImages(image);

            // Compute shoe properties
            ShoeProperties inputShoeFeatures = computeShoeFeatures(resizedImage);

            ShoePropertiesList allShoeProperties = getShoeProperties();

            // Compare shoe properties and return most similar pairs of shoes
            // Return vector of id and confidence score for the x most similar pairs
            int nrPairsToDetect = 5;
            std::vector<std::pair<int, float>> mostSimilarShoes = compareShoeProperties(allShoeProperties, inputShoeFeatures, nrPairsToDetect);

            // Test display most similar shoes
            for (int i = 0; i < mostSimilarShoes.size(); i++) {
                std::cout << "Similar shoe number: " << i+1 << " with id: " << mostSimilarShoes[i].first << " with confidence: " << mostSimilarShoes[i].second << std::endl;
            }
            cv::Mat similarImage1 = getShoeImageByID(mostSimilarShoes[0].first);
            cv::Mat similarImage2 = getShoeImageByID(mostSimilarShoes[1].first);
            cv::Mat similarImage3 = getShoeImageByID(mostSimilarShoes[2].first);
            cv::Mat similarImage4 = getShoeImageByID(mostSimilarShoes[3].first);
            cv::Mat similarImage5 = getShoeImageByID(mostSimilarShoes[4].first);
            std::vector<cv::Mat> similarImages = {
                similarImage1,
                similarImage2,
                similarImage3,
                similarImage4,
                similarImage5
            };
            showMats(similarImages, "Most similar shoes");



            return crow::response("Shoe evaluation completed.");
    });

    // POST
    // Method to evaluate image using all known properties
    // Input: segmented image with a shoe and a white backgound
    //        json with shoe classification data
    // Output: id of most similar shoes in database
    CROW_ROUTE(app, "/evaluate/all-properties")
        .methods(crow::HTTPMethod::Post)([](const crow::request& req){
            // CROW_LOG_INFO << "evaluate method has received body: \n" << req.body.data();

            ImageAndClassification imageResponse = convertRequestToImageAndClassification(req);
            Mat image = imageResponse.image;
            if (image.empty()) {
                CROW_LOG_INFO << "Image is empty";
                return crow::response(500, "Image is empty");
            }

            // // Preprocess shoe image
            // cv::Mat resizedImage = preprocessImages(image);

            // // Compute shoe properties
            // ShoeFeatures inputShoeFeatures = computeShoeFeatures(resizedImage);

            // ShoePropertiesList allShoeProperties = getShoeProperties();
            // std::vector<int> shoeImageIds = allShoeProperties.shoeImageIds;
            // std::vector<std::vector<cv::Mat>> rgbHistograms = allShoeProperties.RGBHistograms;
            // std::vector<cv::Mat> lbpHistogram = allShoeProperties.LBPHistograms;
            // std::vector<cv::Mat> hogFeatures = allShoeProperties.HOGFeatures;

            // std::cout << "Shoe Image IDs size: " << shoeImageIds.size() << std::endl;
            // std::cout << "RGB Histograms size: " << rgbHistograms.size() << std::endl;
            // std::cout << "LBP Histogram size: " << lbpHistogram.size() << std::endl;
            // std::cout << "HOG Features size: " << hogFeatures.size() << std::endl;


            // // Compare shoe properties
            // double weightRGB = 0.3;
            // double weightLBP = 0.5;
            // double weightHOG = 0.2;

            // double maximumCorrelation = 0.0;
            // int mostCorrelatedShoe = -1;

            // double maximumColorCorrelation = 0.0;
            // int mostCorrelatedColorShoe = -1;

            // double maximumLBPcorrelation = 0.0;
            // int mostCorrelatedLBPShoe = -1;

            // double maximumHOGcorrelation = 0.0;
            // int mostCorrelatedHOGShoe = -1;

            // for (int i = 0; i < rgbHistograms.size(); i++) {
            //     double totalCorrelation = 0.0;
            //     double totalColorCorrelation = 0.0;

            //     std::vector<double> correlationRGB;
            //     for (int channel = 0; channel < 3; channel++) {
            //         double channelCorrelation = cv::compareHist(inputShoeFeatures.rgbHistograms[channel], rgbHistograms[i][channel], cv::HISTCMP_CORREL);
            //         correlationRGB.push_back(channelCorrelation);

            //         totalColorCorrelation += channelCorrelation;
            //     }
            //     totalColorCorrelation /= 3;

            //     double lbpCorrelation = cv::compareHist(inputShoeFeatures.lbpHistogram, lbpHistogram[i], cv::HISTCMP_CORREL);
            //     double hogCorrelation = cv::compareHist(inputShoeFeatures.hogFeatures, hogFeatures[i], cv::HISTCMP_CORREL);

            //     totalCorrelation =
            //         weightRGB * totalColorCorrelation +
            //         weightLBP * lbpCorrelation +
            //         weightHOG * hogCorrelation;

            //     if (totalCorrelation > maximumCorrelation) {
            //         maximumCorrelation = totalCorrelation;
            //         mostCorrelatedShoe = shoeImageIds[i];
            //     }

            //     if (totalColorCorrelation > maximumColorCorrelation) {
            //         maximumColorCorrelation = totalColorCorrelation;
            //         mostCorrelatedColorShoe = shoeImageIds[i];
            //     }

            //     if (lbpCorrelation > maximumLBPcorrelation) {
            //         maximumLBPcorrelation = lbpCorrelation;
            //         mostCorrelatedLBPShoe = shoeImageIds[i];
            //     }

            //     if (hogCorrelation > maximumHOGcorrelation) {
            //         maximumHOGcorrelation = hogCorrelation;
            //         mostCorrelatedHOGShoe = shoeImageIds[i];
            //     }
            // }

            // std::cout << "Total correlation: " << maximumCorrelation << std::endl;
            // std::cout << "Total correlation shoeImageID: " << mostCorrelatedShoe << std::endl;

            // std::cout << "RGB correlation: " << maximumColorCorrelation << std::endl;
            // std::cout << "RGB correlation shoeImageID: " << mostCorrelatedColorShoe << std::endl;

            // std::cout << "LBP correlation: " << maximumLBPcorrelation << std::endl;
            // std::cout << "LBP correlation shoeImageID: " << mostCorrelatedLBPShoe << std::endl;

            // std::cout << "HOG correlation: " << maximumHOGcorrelation << std::endl;
            // std::cout << "HOG correlation shoeImageID: " << mostCorrelatedHOGShoe << std::endl;

            // // Display results
            // // cv::Mat totalCorrelationImage = getShoeImageByRGBHistogramID(mostCorrelatedShoe);
            // // cv::Mat colorCorrelationImage = getShoeImageByRGBHistogramID(mostCorrelatedColorShoe);
            // // cv::Mat lbpCorrelationImage = getShoeImageByRGBHistogramID(mostCorrelatedLBPShoe);
            // // cv::Mat hogCorrelationImage = getShoeImageByRGBHistogramID(mostCorrelatedHOGShoe);

            // cv::Mat totalCorrelationImage = getShoeImageByID(mostCorrelatedShoe);
            // cv::Mat colorCorrelationImage = getShoeImageByID(mostCorrelatedColorShoe);
            // cv::Mat lbpCorrelationImage = getShoeImageByID(mostCorrelatedLBPShoe);
            // cv::Mat hogCorrelationImage = getShoeImageByID(mostCorrelatedHOGShoe);

            // showMat(totalCorrelationImage, "Most correlated shoe");
            // showMat(colorCorrelationImage, "Most correlated color shoe");
            // showMat(lbpCorrelationImage, "Most correlated LBP shoe");
            // showMat(hogCorrelationImage, "Most correlated HOG shoe");

            // std::cout << "Most correlated shoe: " << mostCorrelatedShoe << std::endl;

            return crow::response("Shoe evaluation completed.");
    });

    // GET
    // Method to recalculate the histograms of each image in database
    // Input: None
    // Effects: Recalculates the histograms of each image in the database
    CROW_ROUTE(app, "/recalculate-histograms")
        .methods(crow::HTTPMethod::Get)([](){
            try {
                // recalculateHistograms();
            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
            }

            return "Recalculated histograms";
    });

    CROW_ROUTE(app, "/test-db")
        .methods(crow::HTTPMethod::Get)([](){ // Capture the 'conn' variable in the lambda's capture list
            try {
                // testGetShoeMetadata();
            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
            }

            return "Test route";
    });

    CROW_ROUTE(app, "/test-get-shoe-image")
        .methods(crow::HTTPMethod::Get)([](){
            try {
                cv::Mat shoeImage = getShoeImageByID(653);
                showMat(shoeImage);
            } catch (const std::exception &e) {
                CROW_LOG_ERROR << e.what();
            }

            return "Test route";
    });

    //set the port, set the app to run on multiple threads, and run the app
    app.bindaddr("127.0.0.1").port(8081).multithreaded().run();
}