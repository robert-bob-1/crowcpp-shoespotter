#ifndef DATABASE_SHOES_H
#define DATABASE_SHOES_H

#include <iostream>
#include <pqxx/pqxx>
#include "database_features.h"
#include "utils.h"


void getShoeImages() {
    try {
        pqxx::work txn(conn);

        for (auto row: txn.exec("SELECT * FROM public.evaluate_shoeimage")) {
            std::cout << row[0].as<int>() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

std::vector<int> getShoeImagesWithSimilarColor(ShoeColor shoecolor) {
    std::vector<int> shoeIds;

    if (!conn.is_open()) {
        std::cerr << "Can't open database" << std::endl;
        return shoeIds;
    }

    // Create a transactional object to execute the query
    pqxx::work txn(conn);

    std::string dbQuery = R"(
        SELECT shoe_image_id,
                SQRT(POWER(percentage_red - $1, 2) +
                    POWER(percentage_green - $2, 2) +
                    POWER(percentage_blue - $3, 2)) AS distance
        FROM public.evaluate_shoeproperties
        ORDER BY distance
        LIMIT 5;
    )";

    // Execute the query
    pqxx::result res = txn.exec_params(dbQuery, shoecolor.red, shoecolor.green, shoecolor.blue);

    // Process the results
    for (const auto& row : res) {
        shoeIds.push_back(row[0].as<int>());
    }

    return shoeIds;
}

std::map<int, std::vector<DominantColor>> getShoeImagesWithDominantColors() {
    if (!conn.is_open()) {
        throw std::runtime_error("Can't open database");
    }

    // Create a transactional object to execute the query
    pqxx::work txn(conn);

    std::string dbQuery = R"(
        SELECT shoe_image_id, red, green, blue, frequency_percentage
        FROM public.evaluate_shoedominantcolor
        ORDER BY shoe_image_id;
    )";

    // Execute the query
    pqxx::result res = txn.exec(dbQuery);

    std::map<int, std::vector<DominantColor>> shoeImageColors;
    // Process the results
    for (const auto& row : res) {
        int shoeImageId = row[0].as<int>();
        DominantColor dominantColor;
        dominantColor.color[2] = row["red"].as<int>();
        dominantColor.color[1] = row["green"].as<int>();
        dominantColor.color[0] = row["blue"].as<int>();
        dominantColor.percentage = row["frequency_percentage"].as<float>();

        shoeImageColors[shoeImageId].push_back(dominantColor);
    }

    return shoeImageColors;
}

cv::Mat getShoeImageByRGBHistogramID(int rgbHistId) {
    if (!conn.is_open()) {
        throw std::runtime_error("Can't open database");
    }

    pqxx::work txn(conn);

    // Execute the query
    pqxx::result res = txn.exec_params(
        R"(
            SELECT image
            FROM public.evaluate_shoeimage as im
            JOIN public.evaluate_shoehistograms as hist ON im.id = hist.shoe_image_id
            WHERE hist.id = $1;
        )",
        rgbHistId
    );

    // Process result and convert image from binary to Mat
    cv::Mat shoeImage;
    if (!res.empty()) {
        pqxx::binarystring imageBinary = res[0]["image"].as<pqxx::binarystring>();
        int rows = 640;
        int cols = 640;
        shoeImage = cv::Mat(rows, cols, CV_8UC3, (void*)imageBinary.data());
        // Test display image
        // showMat(shoeImage);
        return shoeImage.clone();
    } else {
        throw std::runtime_error("No shoe image found with the given RGB histogram ID");
    }
}

cv::Mat getShoeImageByID(int id) {
    if (!conn.is_open()) {
        throw std::runtime_error("Can't open database");
    }

    pqxx::work txn(conn);

    // Execute the query
    pqxx::result res = txn.exec_params(
        R"(
            SELECT image
            FROM public.evaluate_shoeimage
            WHERE id = $1;
        )",
        id
    );

    // Process result and convert image from binary to Mat
    cv::Mat shoeImage;
    if (!res.empty()) {
        pqxx::binarystring imageBinary = res[0]["image"].as<pqxx::binarystring>();

        // Convert the binary data to a vector of data which is then decoded into a matrix
        std::vector<uchar> data(imageBinary.size());
        std::memcpy(data.data(), imageBinary.data(), imageBinary.size());
        shoeImage = cv::imdecode(data, cv::IMREAD_COLOR);

        if (shoeImage.empty()) {
            throw std::runtime_error("Failed to decode image");
        }

        return shoeImage.clone();
    } else {
        throw std::runtime_error("No shoe image found with given ID");
    }
}

#endif // DATABASE_SHOES_H