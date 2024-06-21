#ifndef COMPARE_H
#define COMPARE_H

#include "database_features.h"
#include "database_shoes.h"


// Return an ordered list of the most similar shoe images with their similarity scores
// based on dominant colors
std::vector<std::pair<int, float>> getShoeImagesWithSimilarDominantColors(std::vector<DominantColor> inputColors) {
    std::vector<std::pair<int, float>> shoeImages;

    // Get all images and respective dominant colors
    std::map<int, std::vector<DominantColor>> shoeDominantColors = getShoeImagesWithDominantColors();

    // Compare dominant colors
    for (const auto& [shoeImageId, dominantColors] : shoeDominantColors) {
        // Associate the most similar dominant colors with the input colors
        // Obtain pairs of dominant colors
        std::vector<DominantColor> inputColorsCopy = inputColors;
        std::vector<std::pair<DominantColor, DominantColor>> dominantColorPairs;
        for (DominantColor dominantColor : dominantColors) {
            float minimumDistance = std::numeric_limits<float>::max();
            DominantColor mostSimilarInputDominantColor;
            for (DominantColor inputColor : inputColors) {
                float distance = cv::norm(dominantColor.color, inputColor.color);
                if (distance < minimumDistance) {
                    minimumDistance = distance;
                    mostSimilarInputDominantColor = inputColor;
                }
            }

            // Add the pair to the list
            dominantColorPairs.push_back(std::make_pair(dominantColor, mostSimilarInputDominantColor));
            inputColorsCopy.erase(
                std::remove(inputColorsCopy.begin(),
                    inputColorsCopy.end(), mostSimilarInputDominantColor), inputColorsCopy.end()
            );
        }

        // Compute similarity score, taking into account the frequency of the dominant colors
        // The bigger the difference in frequency, the bigger the totalDifference
        float totalDifference = 0.0;
        for (const auto& [dominantColor, mostSimilarInputDominantColor] : dominantColorPairs) {
            // float frequencyDifference;
            // if (dominantColor.percentage > mostSimilarInputDominantColor.percentage)
            //     frequencyDifference = dominantColor.percentage / mostSimilarInputDominantColor.percentage;
            // else
            //     frequencyDifference = mostSimilarInputDominantColor.percentage / dominantColor.percentage;

            totalDifference += cv::norm(dominantColor.color, mostSimilarInputDominantColor.color);
        }

        shoeImages.push_back(std::make_pair(shoeImageId, totalDifference));
    }

    // Sort the shoe images by similarity score
    std::sort(shoeImages.begin(), shoeImages.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });


    return shoeImages;
}

// Function to add a id, confidence_score pair to a sorted vector of pairs.
// Also checks if vector size is less than the max input size.
void addShoeImageToVector(std::vector<std::pair<int, float>>& shoeImages, std::pair<int, float> shoeImage, int maxInputSize = 5) {
    if (shoeImages.empty()) {
        shoeImages.push_back(shoeImage);
        return;
    }

    bool isShoeImageVectorFilled = shoeImages.size() >= maxInputSize;

    for (int i = 0; i < shoeImages.size(); i++) {
        if (shoeImage.second > shoeImages[i].second) {
            shoeImages.insert(shoeImages.begin() + i, shoeImage);
            if (isShoeImageVectorFilled) {
                shoeImages.pop_back();
            }
            return;
        }
    }
}


// Return an ordered list of the most similar shoe images with their id and similarity score
std::vector<std::pair<int, float>> compareShoeProperties(
    ShoePropertiesList allShoesProperties,
    ShoeProperties inputShoeFeatures,
    int nrOfSimilarShoes = 5
) {
    std::vector<int> shoeImageIds = allShoesProperties.shoeImageIds;
    std::vector<std::vector<cv::Mat>> rgbHistograms = allShoesProperties.RGBHistograms;
    std::vector<cv::Mat> lbpHistogram = allShoesProperties.LBPHistograms;
    std::vector<cv::Mat> hogFeatures = allShoesProperties.HOGFeatures;

    // std::cout << "Shoe Image IDs size: " << shoeImageIds.size() << std::endl;
    // std::cout << "RGB Histograms size: " << rgbHistograms.size() << std::endl;
    // std::cout << "LBP Histogram size: " << lbpHistogram.size() << std::endl;
    // std::cout << "HOG Features size: " << hogFeatures.size() << std::endl;


    std::vector<std::pair<int, float>> similarShoeImages;

    // Compare shoe properties
    double weightRGB = 0.3;
    double weightLBP = 0.3;
    double weightHOG = 0.4;

    double maximumCorrelation = 0.0;
    int mostCorrelatedShoe = -1;

    double maximumColorCorrelation = 0.0;
    int mostCorrelatedColorShoe = -1;
    cv::Mat mostCorrelatedColorShoeRHistogram;

    double maximumLBPcorrelation = 0.0;
    int mostCorrelatedLBPShoe = -1;

    double maximumHOGcorrelation = 0.0;
    int mostCorrelatedHOGShoe = -1;

    for (int i = 0; i < rgbHistograms.size(); i++) {
        double totalCorrelation = 0.0;
        double totalColorCorrelation = 0.0;

        std::vector<double> correlationRGB;
        for (int channel = 0; channel < 3; channel++) {
            double channelCorrelation = cv::compareHist(inputShoeFeatures.rgbHistograms[channel], rgbHistograms[i][channel], cv::HISTCMP_CORREL);
            correlationRGB.push_back(channelCorrelation);

            totalColorCorrelation += channelCorrelation;
        }
        totalColorCorrelation /= 3;

        double lbpCorrelation = cv::compareHist(inputShoeFeatures.lbpHistogram, lbpHistogram[i], cv::HISTCMP_CORREL);
        double hogCorrelation = cv::compareHist(inputShoeFeatures.hogFeatures, hogFeatures[i], cv::HISTCMP_CORREL);

        totalCorrelation =
            weightRGB * totalColorCorrelation +
            weightLBP * lbpCorrelation +
            weightHOG * hogCorrelation;

        addShoeImageToVector(similarShoeImages, std::make_pair(shoeImageIds[i], totalCorrelation), nrOfSimilarShoes);

        if (totalCorrelation > maximumCorrelation) {
            maximumCorrelation = totalCorrelation;
            mostCorrelatedShoe = shoeImageIds[i];
        }

        if (totalColorCorrelation > maximumColorCorrelation) {
            maximumColorCorrelation = totalColorCorrelation;
            mostCorrelatedColorShoe = shoeImageIds[i];
            mostCorrelatedColorShoeRHistogram = rgbHistograms[i][0];
        }

        if (lbpCorrelation > maximumLBPcorrelation) {
            maximumLBPcorrelation = lbpCorrelation;
            mostCorrelatedLBPShoe = shoeImageIds[i];
        }

        if (hogCorrelation > maximumHOGcorrelation) {
            maximumHOGcorrelation = hogCorrelation;
            mostCorrelatedHOGShoe = shoeImageIds[i];
        }

    }

    std::cout << "Total correlation: " << maximumCorrelation << std::endl;
    std::cout << "Total correlation shoeImageID: " << mostCorrelatedShoe << std::endl;

    std::cout << "RGB correlation: " << maximumColorCorrelation << std::endl;
    std::cout << "RGB correlation shoeImageID: " << mostCorrelatedColorShoe << std::endl;

    std::cout << "LBP correlation: " << maximumLBPcorrelation << std::endl;
    std::cout << "LBP correlation shoeImageID: " << mostCorrelatedLBPShoe << std::endl;

    std::cout << "HOG correlation: " << maximumHOGcorrelation << std::endl;
    std::cout << "HOG correlation shoeImageID: " << mostCorrelatedHOGShoe << std::endl;

    // Display some results that are not returned like hog/lbp/color correlated shoe images
    // cv::Mat totalCorrelationImage = getShoeImageByID(mostCorrelatedShoe);
    // cv::Mat colorCorrelationImage = getShoeImageByID(mostCorrelatedColorShoe);
    // cv::Mat lbpCorrelationImage = getShoeImageByID(mostCorrelatedLBPShoe);
    // cv::Mat hogCorrelationImage = getShoeImageByID(mostCorrelatedHOGShoe);
    // showMat(totalCorrelationImage, "Most correlated shoe");
    // showMat(colorCorrelationImage, "Most correlated color shoe");
    // showMat(lbpCorrelationImage, "Most correlated LBP shoe");
    // showMat(hogCorrelationImage, "Most correlated HOG shoe");
    // std::cout << "Most correlated shoe: " << mostCorrelatedShoe << std::endl;

    return similarShoeImages;
}

#endif // !COMPARE_H