#ifndef EVALUATE_H
#define EVALUATE_H

#include <pqxx/pqxx>
#include <iostream>

#include "compare.h"
#include "compute.h"
#include "database_features.h"
#include "database_shoes.h"

struct ShoeImageSimilarity {
    int shoeImageId;
    double similarityScore;
};


int evaluateShoeColor(ShoeColor shoecolor) {
    // Compare shoe color with shoes in database
    // If shoe color is within a certain threshold, return the shoe id
    // Else return -1
    std::vector<int> shoeIds = getShoeImagesWithSimilarColor(shoecolor);
    std::cout << "Shoe IDs with similar color: ";
    for (int i = 0; i < shoeIds.size(); i++) {
        std::cout << shoeIds[i] << " ";
    }
    return 1;
}

std::vector<int> evaluateShoeDominantColors(std::vector<DominantColor> dominantColors) {
    // Compare shoe dominant colors with shoe images in database

    std::vector<std::pair<int, float>> shoeImageId_and_similarityScore = getShoeImagesWithSimilarDominantColors(dominantColors);

    std::vector<int> shoeImageIds;
    for (int i = 0; i < shoeImageId_and_similarityScore.size(); i++) {
        std::cout << "Shoe ID: " << shoeImageId_and_similarityScore[i].first << " Similarity score: " << shoeImageId_and_similarityScore[i].second << std::endl;
        shoeImageIds.push_back(shoeImageId_and_similarityScore[i].first);
    }

    return shoeImageIds;
}

#endif